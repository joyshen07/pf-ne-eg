from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import scipy
import pickle


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto probability simplex"""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def generate_sparse_matrix(d: int, sparsity: float, random_seed: int = None) -> np.ndarray:
    """Generate sparse random matrix for matrix game"""
    if random_seed is not None:
        np.random.seed(random_seed)
    # Generate mask of where to place non-zero entries
    mask = np.random.rand(d, d) < sparsity
    # Generate uniform values in [-1, 1] for those entries
    mat = np.random.uniform(-1, 1, size=(d, d)) * mask
    return mat


class SaddlePointProblem(ABC):
    """Base class for saddle point problems: min_x max_y f(x, y)"""

    def __init__(self, dim_x: int, dim_y: int, seed: int = 42):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim = dim_x + dim_y  # Total dimension
        self.seed = seed
        np.random.seed(seed)
        self.x_opt = None
        self.y_opt = None
        self.metrics = {}
        self._setup()

    @abstractmethod
    def _setup(self):
        """Initialize problem-specific parameters"""
        pass

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unconstrained problem - no projection needed"""
        return x, y

    @abstractmethod
    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute objective function value f(x, y)"""
        pass

    @abstractmethod
    def grad_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient with respect to x"""
        pass

    @abstractmethod
    def grad_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient with respect to y"""
        pass

    def gradient(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute both gradients"""
        return self.grad_x(x, y), self.grad_y(x, y)

    # def operator(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    #     """Monotone operator F(z) = [∇_x f(x,y); -∇_y f(x,y)]"""
    #     gx, gy = self.gradient(x, y)
    #     return np.concatenate([gx, -gy])

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial point"""
        x0 = np.random.randn(self.dim_x)
        y0 = np.random.randn(self.dim_y)
        return x0, y0

    def saddle_point_gap(self, x: np.ndarray, y: np.ndarray, obj_val: float = None) -> float:
        """Compute duality gap or other measure of optimality"""
        return np.nan  # Override in subclass if needed

    def natural_residual(self, x: np.ndarray, y: np.ndarray, step_size: float = 1.0,
                         gx: np.ndarray = None, gy: np.ndarray = None) -> float:
        """Compute natural residual: ||z - Proj(z - α*F(z))||"""
        # Use cached gradients if provided
        if gx is None or gy is None:
            gx, gy = self.gradient(x, y)

        x_temp = x - step_size * gx
        y_temp = y + step_size * gy
        x_proj, y_proj = self.project(x_temp, y_temp)
        residual_x = np.sum((x - x_proj)**2)
        residual_y = np.sum((y - y_proj)**2)

        return np.sqrt(residual_x + residual_y)

    def dist_to_opt(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance to the optimal solution"""
        if self.x_opt is None:
            return np.nan
        return np.sqrt(np.sum((x - self.x_opt) ** 2) + np.sum((y - self.y_opt) ** 2))

    def lower_bound(self, x: np.ndarray, gx: np.ndarray, obj: float) -> float:
        """Compute lower bound of the relaxation based on linear underestimate"""
        return np.nan


class MatrixGameProblem(SaddlePointProblem):
    """Zero-sum matrix game with simplex constraints (using barrier approximation)
    min_x max_y x^T M y where x, y are on simplices"""

    def __init__(self, dim_x: int, dim_y: int, seed: int = 42,
                 sparsity: float = 1.0, matrix: np.ndarray = None):
        """Initialize matrix game problem

        Args:
            dim_x: Dimension of x (rows of M)
            dim_y: Dimension of y (columns of M)
            seed: Random seed
            sparsity: Probability κ that a matrix entry is nonzero (0 < κ <= 1)
            matrix: Optional pre-specified matrix M. If None, generates random sparse matrix
        """
        self.sparsity = sparsity
        self.custom_matrix = matrix
        super().__init__(dim_x, dim_y, seed)
        self.metrics = {'sp_gap': self.saddle_point_gap}

    def _setup(self):
        self.M = generate_sparse_matrix(self.dim_x, self.sparsity, self.seed)
        # self.mu = 0.1  # Barrier parameter

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project onto probability simplices"""
        return project_simplex(x), project_simplex(y)

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        # # Add log barrier to enforce simplex constraints approximately
        # barrier_x = -self.mu * np.sum(np.log(np.maximum(x, 1e-10)))
        # barrier_y = self.mu * np.sum(np.log(np.maximum(y, 1e-10)))
        return x @ self.M @ y  # + barrier_x + barrier_y

    def grad_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.M @ y  # - self.mu / np.maximum(x, 1e-10)

    def grad_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.M.T @ x  # + self.mu / np.maximum(y, 1e-10)

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start with uniform distribution"""
        x0 = np.ones(self.dim_x) / self.dim_x
        y0 = np.ones(self.dim_y) / self.dim_y
        return x0, y0

    def saddle_point_gap(self, x: np.ndarray, y: np.ndarray, obj_val: float = None) -> float:
        """Compute saddle point gap: max_y f(x,y) - min_x f(x,y)

        For matrix game f(x,y) = x^T M y on simplices:
        - max_y x^T M y = max_i (M^T x)_i  (select best pure strategy for y)
        - min_x x^T M y = min_j (M y)_j     (select best pure strategy for x)
        """
        # max_y x^T M y subject to y on simplex
        # Solution: y concentrates on argmax_j (M^T x)_j
        max_val = np.max(self.M.T @ x)

        # min_x x^T M y subject to x on simplex
        # Solution: x concentrates on argmin_i (M y)_i
        min_val = np.min(self.M @ y)

        return max_val - min_val


class MESP(SaddlePointProblem):
    """Abstract class for MESP relaxations"""

    def _setup(self):
        # read data from file
        self.C = scipy.io.loadmat(f'../data/data{self.d:d}.mat')['C']

    def project_capped_simplex(self, x):
        # make projection onto domain of x
        ext_x = np.concatenate((x, x - 1))
        ind_x = np.argsort(ext_x)
        sorted_x = ext_x[ind_x]
        is_x_minus_1 = np.concatenate((np.zeros(self.d), np.ones(self.d)))[ind_x]
        sorted_x = np.concatenate((sorted_x, np.array([np.inf])))
        i = 2 * self.d - 1
        sum_x = -self.s
        denom = 0
        while i >= 0 and not (denom > 0 and sorted_x[i] <= sum_x / denom <= sorted_x[i + 1]):
            if is_x_minus_1[i] == 1:
                sum_x -= sorted_x[i]
                denom -= 1
            else:
                sum_x += sorted_x[i]
                denom += 1
            i -= 1
        nu = sum_x / denom
        res = np.minimum(np.maximum(x - nu, np.zeros(self.d)), np.ones(self.d))
        return res

    def lower_bound(self, x: np.ndarray, y: np.ndarray, obj: float = None) -> float:
        """Compute lower bound of the relaxation based on linear underestimate"""
        if obj is None:
            obj = self.objective(x, y)
        gx, _ = self.gradient(x, y)
        x_tmp = np.zeros(self.d)
        x_tmp[np.argpartition(gx, self.s)[:self.s]] = 1  # smallest s elements
        lb = obj + gx @ (x_tmp - x)
        if self.opt_val is None:
            return lb
        else:
            return self.opt_val - lb


class LinxDoubleScaling(MESP):
    """Double-scaling applied to the linx relaxation of the MESP."""

    def __init__(self, d: int, s: int):
        self.d = d
        self.s = s
        super().__init__(d, 2 * d)
        self.L_inv = None
        self.opt_val = None
        self.metrics = {'lb_diff': self.lower_bound}

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project onto probability simplices"""
        return self.project_capped_simplex(x), y

    def split_lg(self, lg):
        lg0 = lg[:self.d]
        g0 = np.exp(lg0)
        lg1 = lg[-self.d:]
        g1 = np.exp(lg1)
        return g0, g1, lg0, lg1

    def matrix_L(self, x, lg):
        g0, g1, _, _ = self.split_lg(lg)
        return np.diag(g1) @ self.C @ np.diag(x * g0) @ self.C @ np.diag(g1) + np.diag((1 - x))

    def objective(self, x: np.ndarray, lg: np.ndarray) -> float:
        _, _, lg0, lg1 = self.split_lg(lg)
        return -.5 * np.linalg.slogdet(self.matrix_L(x, lg))[1] + .5 * x @ lg0 + x @ lg1

    def grad_x(self, x: np.ndarray, lg: np.ndarray) -> np.ndarray:
        # gradient wrt x
        # -.5 * g0 \circ diag(C @ Diag(g1) @ L^{-1} @ Diag(g1) @ C) + .5 * diag(L^{-1}) + .5 lg0 + .5 * lg1
        # see our notes
        g0, g1, lg0, lg1 = self.split_lg(lg)
        return -.5 * (g0 * np.diag(self.C @ np.diag(g1) @ self.L_inv @ np.diag(g1) @ self.C) - np.diag(self.L_inv)) \
            + .5 * lg0 + lg1

    def grad_y(self, x: np.ndarray, lg: np.ndarray) -> np.ndarray:
        # gradient wrt the log of scaling vector
        # grad_lg0 = -.5 * g0 \circ x \circ diag(C @ Diag(g1) @ L^{-1} @ Diag(g1) @ C) + .5 * x
        # grad_lg1 = (1 - x) \circ diag(L^{-1}) + x - 1
        # see our notes
        g0, g1, _, _ = self.split_lg(lg)
        grad_lg0 = -.5 * g0 * x * np.diag(self.C @ np.diag(g1) @ self.L_inv @ np.diag(g1) @ self.C) + .5 * x
        # grad_lg0 *= 2
        # grad_lg0 = np.zeros(self.data.d)
        grad_lg1 = (1 - x) * np.diag(self.L_inv) - (1 - x)
        # grad_lg1 = np.zeros(self.data.d)
        return np.hstack([grad_lg0, grad_lg1])

    def gradient(self, x: np.ndarray, lg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute both gradients"""
        self.L_inv = np.linalg.inv(self.matrix_L(x, lg))
        return self.grad_x(x, lg), self.grad_y(x, lg)

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start with uniform distribution"""
        x = np.ones(self.d) * self.s / self.d
        lg = np.zeros(2 * self.d)  # initialize log of scaling vector
        return x, lg

    def get_optimal_solution(self):
        """Obtain proxy optimal solution from saved file"""
        filename = f'output/proxy-{self.d}-{self.s}-iter100000.pkl'
        try:
            with open(filename, 'rb') as f:
                z_proxy = pickle.load(f)
                self.x_opt = z_proxy['x']
                self.y_opt = z_proxy['y']
                if self.opt_val is None:
                    self.opt_val = self.lower_bound(self.x_opt, self.y_opt)
            print("Loaded proxy successfully.")
        except FileNotFoundError:
            print(f"{filename} not found. ")


class GammaStar(MESP):
    """Gamma^* relaxation of the MESP."""

    def __init__(self, d: int, s: int):
        self.d = d
        self.s = s
        super().__init__(d, 2)
        self.metrics = {'sp_gap': self.saddle_point_gap}

    def _setup(self):
        # read data from file
        self.C = scipy.io.loadmat(f'../data/data{self.d:d}.mat')['C']
        self.V = np.linalg.cholesky(self.C)
        self.W = np.linalg.pinv(self.V).T
        self.log_det_C = np.linalg.slogdet(self.C)[1]

    def project_2d_simplex(self, alpha: np.ndarray) -> np.ndarray:
        p0 = np.array([1.0, 0.0])
        p1 = np.array([0.0, 1.0])
        direction = p1 - p0  # direction (-1, 1)

        # projection parameter onto infinite line
        t = np.dot(alpha - p0, direction) / np.dot(direction, direction)

        # clamp to segment [0, 1]
        t = np.clip(t, 0.0, 1.0)

        # projected point on segment
        return p0 + t * direction

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.project_capped_simplex(x), self.project_2d_simplex(y)

    def find_k(self, lmbd: np.ndarray, s: int = None) -> tuple[int, float]:
        s = self.s if s is None else s
        sum_lmbd = np.sum(lmbd)
        for i in range(s):
            nu = sum_lmbd / (s - i)
            if nu >= lmbd[i] - 1e-10:
                return i, nu
            sum_lmbd -= lmbd[i]

    def obj_n_grad_core(self, X: np.ndarray, s: int):
        # Eigen decomposition
        lmbd, U = np.linalg.eigh(X)
        lmbd = lmbd[::-1]  # eigenvalues in descending order

        # Find k and nu
        k, nu = self.find_k(lmbd, s)

        # Compute objective
        obj_val = -np.sum(np.log(lmbd[:k])) - (s - k) * np.log(nu)
        # obj_val = 0
        # for i in range(k):
        #     obj_val += -np.log(lmbd[i])
        # obj_val += -np.log(nu) * (s - k)

        # Compute gradient
        y = np.zeros(self.d)
        y[:k] = -1 / lmbd[:k]
        y[k:] = -1 / nu
        y = y[::-1]  # reverse to match original indexing
        # for j in range(k):
        #     y[-1 - j] = -1 / (lmbd[j])
        # for j in range(k, self.d):
        #     y[-1 - j] = -1 / (nu)

        # Reconstruct Y
        Y = U @ np.diag(y) @ U.T
        # self.dual = s + np.sum(np.log(-y[-s:]))

        return obj_val, Y

    def x_to_X(self, x: np.ndarray) -> np.ndarray:
        X = self.V.T @ np.diag(x) @ self.V
        # X = sum(x[i] * self.aux.V_square[i] for i in range(self.d))
        return X

    def obj_n_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        obj_val, Y = self.obj_n_grad_core(self.x_to_X(x), self.s)
        grad = np.diag(self.V @ Y @ self.V.T)
        return obj_val, grad

    def x_to_X_compl(self, x: np.ndarray) -> np.ndarray:
        X = self.W.T @ np.diag(1 - x) @ self.W
        return X

    def obj_n_grad_compl(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        obj_val, Y = self.obj_n_grad_core(self.x_to_X_compl(x), self.d - self.s)
        grad = -np.diag(self.W @ Y @ self.W.T)
        return obj_val - self.log_det_C, grad

    def objective(self, x: np.ndarray, alpha: np.ndarray) -> float:
        _, objs = self.gradient(x, alpha)
        return alpha @ objs

    def gradient(self, x: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute both gradients"""
        # Get individual objectives and gradients for Gamma and Gamma^c
        obj1, grad1 = self.obj_n_grad(x)
        obj2, grad2 = self.obj_n_grad_compl(x)

        # Store for diagnostics
        objs = np.array([obj1, obj2])

        return alpha[0] * grad1 + alpha[1] * grad2, objs

    def grad_x(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return self.gradient(x, alpha)[0]

    def grad_y(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return self.gradient(x, alpha)[1]

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial point"""
        x0 = np.ones(self.d) * self.s / self.d
        y0 = np.array([.5, .5])
        return x0, y0

    def saddle_point_gap(self, x: np.ndarray, alpha: np.ndarray, obj_val: float = None) -> float:
        """Approximate saddle point gap"""

        # Compute objectives and gradients once
        grad_x, objs = self.gradient(x, alpha)

        # Compute the maximizer x of the linearization
        x_tmp = np.zeros(self.d)
        idx = np.argpartition(grad_x, self.s)[:self.s]
        x_tmp[idx] = 1

        # Dual value
        dual_val = objs @ alpha + (x_tmp - x) @ grad_x

        # Record primal-dual gap
        return max(objs) - dual_val
