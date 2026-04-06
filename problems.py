from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import scipy
import pickle
from sklearn.datasets import make_classification


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto probability simplex"""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    cond = u * np.arange(1, n + 1) > (cssv - 1)
    if not np.any(cond):
        theta = 0.0
    else:
        rho = np.where(cond)[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def project_l2_ball(x: np.ndarray, radius: float) -> np.ndarray:
    x = np.asarray(x)
    norm = np.linalg.norm(x)
    return x if norm <= radius else x / norm * radius


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
        self.metrics = {'nat_res': self.natural_residual}
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

    def saddle_point_gap(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute duality gap or other measure of optimality"""
        return np.nan  # Override in subclass if needed

    def natural_residual(self, x: np.ndarray, y: np.ndarray, step_size: float = 0.01,
                         gx: np.ndarray = None, gy: np.ndarray = None) -> float:
        """Compute normalized natural residual: 1/α * ||z - Proj(z - α*F(z))||"""
        # Use cached gradients if provided
        if gx is None or gy is None:
            gx, gy = self.gradient(x, y)

        x_temp = x - step_size * gx
        y_temp = y + step_size * gy
        x_proj, y_proj = self.project(x_temp, y_temp)
        residual_x = np.sum((x - x_proj)**2)
        residual_y = np.sum((y - y_proj)**2)

        return np.sqrt(residual_x + residual_y) / step_size

    def dist_to_opt(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance to the optimal solution"""
        if self.x_opt is None:
            return np.nan
        return np.sqrt(np.sum((x - self.x_opt) ** 2) + np.sum((y - self.y_opt) ** 2))

    def lower_bound(self, x: np.ndarray, gx: np.ndarray, obj: float) -> float:
        """Compute lower bound of the relaxation based on linear underestimate"""
        return np.nan


class MatrixGameProblem(SaddlePointProblem):
    """Zero-sum matrix game with simplex constraints
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

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project onto probability simplices"""
        return project_simplex(x), project_simplex(y)

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        return x @ self.M @ y

    def grad_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.M @ y

    def grad_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.M.T @ x

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start with uniform distribution"""
        x0 = np.ones(self.dim_x) / self.dim_x
        y0 = np.ones(self.dim_y) / self.dim_y
        return self.project(x0, y0)

    def saddle_point_gap(self, x: np.ndarray, y: np.ndarray) -> float:
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


class Bilinear(SaddlePointProblem):
    """Bilinear saddle point problem
    min_x max_y x^T M y where x is unconstrained, y is in L_inf norm ball"""

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
        super().__init__(dim_x, dim_y, seed)

    def _setup(self):
        self.M = np.random.randn(self.dim_x, self.dim_y)

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project onto probability simplices"""
        return x, np.clip(y, -1, 1)

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        return x @ self.M @ y

    def grad_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.M @ y

    def grad_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.M.T @ x

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start with uniform distribution"""
        x0 = np.random.randn(self.dim_x) / self.dim_x
        y0 = np.random.randn(self.dim_y) / self.dim_y
        return self.project(x0, y0)


class LASSO(SaddlePointProblem):
    """LASSO as a saddle point problem
    min_x max_y {1/2 * ||A x - b||_2^2 + x^T y}
    where x is unconstrained, ||y||_inf <= lmd"""

    def __init__(self, dim_x: int, dim_y: int, seed: int = 42, lmd: float = 1.0,
                 sparsity: float = 1.0, matrix: np.ndarray = None):
        """Initialize matrix game problem

        Args:
            dim_x: Dimension of x (rows of M)
            dim_y: Dimension of y (columns of M)
            seed: Random seed
            sparsity: Probability κ that a matrix entry is nonzero (0 < κ <= 1)
            matrix: Optional pre-specified matrix M. If None, generates random sparse matrix
        """
        self.lmd = lmd
        self.sparsity = sparsity
        super().__init__(dim_x, dim_y, seed)

    def _setup(self):
        self.A = np.random.randn(self.dim_y, self.dim_x)
        # Normalize columns to have unit norm (standard for Lasso)
        self.A /= np.linalg.norm(self.A, axis=0)

        # Create a true sparse signal x
        x_true = np.zeros(self.dim_x)
        indices = np.random.choice(self.dim_x, int(self.dim_x * self.sparsity), replace=False)
        x_true[indices] = np.random.randn(len(indices))

        # Generate observations with some noise
        self.b = self.A @ x_true + 0.05 * np.random.randn(self.dim_y)

        # Standard rule for picking lambda
        self.lmd = 0.1 * np.max(np.abs(self.A.T @ self.b))

    def project(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project onto probability simplices"""
        return x, np.clip(y, -self.lmd, self.lmd)

    def objective(self, x: np.ndarray, y: np.ndarray) -> float:
        residual = self.A @ x - self.b
        return 0.5 * np.sum(residual ** 2) + self.lmd * y.T @ self.A @ x

    def grad_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.A.T @ (self.A @ x - self.b) + self.A.T @ y

    def grad_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.A @ x

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start with uniform distribution"""
        x0 = np.zeros(self.dim_x)
        y0 = np.zeros(self.dim_y)
        return self.project(x0, y0)


class GroupFairnessClassification(SaddlePointProblem):
    """Group fairness classification problem"""

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, lmd: float = 0,
                 n_groups: int = 5, n_samples_per_group: int = 100, n_features: int = 100):
        """Initialize matrix game problem

        Args:
        """
        self.X = X
        self.y = y
        self.reg_lambda = lmd
        self.n_groups = n_groups
        self.n_samples_per_group = n_samples_per_group
        self.n_features = n_features
        dim_q = self.n_groups
        dim_theta = self.n_features
        super().__init__(dim_q, dim_theta)

    def _setup(self):
        """
        Generates a synthetic dataset for minimax fairness testing
        """
        group_features = []
        group_labels = []

        for i in range(self.n_groups):
            # We vary the difficulty and class balance for each group
            # Group 0 might be easy (linearly separable),
            # while Group 2 might be noisy and imbalanced.
            X, y = make_classification(
                n_samples=self.n_samples_per_group,
                n_features=self.n_features - 1,
                n_informative=self.n_features - 3,
                n_redundant=2,
                flip_y=0.1 * (i ** 2) / self.n_groups ** 2,  # Escalating label noise
                weights=[0.5 + (0.1 * i / self.n_groups), 0.5 - (0.1 * i / self.n_groups)],  # Shifting class balance
                random_state=self.seed + i
            )

            # Add a bias term (column of 1s) if you want the model to have an intercept
            X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])

            group_features.append(X_with_bias)
            group_labels.append(y.reshape(-1))

        self.X, self.y = group_features, group_labels

    def project(self, theta: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project onto probability simplices"""
        return theta, project_simplex(q)

    def objective(self, theta: np.ndarray, q: np.ndarray) -> float:
        # Compute individual group losses
        losses = np.zeros(self.n_groups)
        for i in range(self.n_groups):
            # Using y in {-1, 1} for the classic logistic form
            # L = mean(log(1 + exp(-y * X * theta)))
            y_raw = 2 * self.y[i] - 1  # Convert {0, 1} to {-1, 1}
            margins = y_raw * (self.X[i] @ theta)
            # log(1 + exp(-x)) is stable via:
            # losses[i] = np.mean(np.logaddexp(0, -margins))
            losses[i] = np.mean(np.exp(-margins))  # exponential loss

        # Compute total objective
        # Phi(theta, q) = sum(q_i * L_i) + reg
        weighted_loss = np.sum(q.flatten() * losses)
        reg_term = 0.5 * self.reg_lambda * np.sum(theta**2)
        objective = weighted_loss + reg_term

        return objective

    def grad_x(self, theta: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Individual group gradients
        grads_per_group = []

        for i in range(self.n_groups):
            y_raw = 2 * self.y[i] - 1
            z = y_raw * (self.X[i] @ theta)
            # Gradient of log(1+exp(-z)) is -1 / (1 + exp(z))
            # p_weights = -y_raw / (1 + np.exp(z))
            p_weights = -y_raw * np.exp(-z)  # exponential loss
            grad_per_group = (self.X[i].T @ p_weights) / len(self.y[i])
            grads_per_group.append(grad_per_group)

        # Gradient wrt theta
        # Weighted average of group gradients + reg
        grad_theta = np.zeros_like(theta)
        for i in range(self.n_groups):
            grad_theta += q[i] * grads_per_group[i]
        grad_theta += self.reg_lambda * theta

        return grad_theta

    def grad_y(self, theta: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Compute individual group losses
        losses = np.zeros(self.n_groups)
        for i in range(self.n_groups):
            # Using y in {-1, 1} for the classic logistic form
            # L = mean(log(1 + exp(-y * X * theta)))
            y_raw = 2 * self.y[i] - 1  # Convert {0, 1} to {-1, 1}
            margins = y_raw * (self.X[i] @ theta)
            # log(1 + exp(-x)) is stable via:
            # losses[i] = np.mean(np.logaddexp(0, -margins))
            losses[i] = np.mean(np.exp(-margins))  # exponential loss
        # Gradient wrt q
        # Partial wrt q_i is just L_i
        return losses.reshape(-1)

    def initial_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start with uniform distribution"""
        theta0 = np.zeros(self.n_features)
        q0 = np.ones(self.n_groups) / self.n_groups
        return self.project(theta0, q0)


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

    def saddle_point_gap(self, x: np.ndarray, alpha: np.ndarray) -> float:
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
