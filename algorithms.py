from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import time

from scipy._lib.cobyqa import problem

from problems import SaddlePointProblem


DEFAULT_METRIC_PRIORITY = [
    'sp_gap',
    'lb_diff',
    'dist_to_opt',
    'nat_res'
]


def select_convergence_metric(problem: SaddlePointProblem, algo) -> str:
    for m in DEFAULT_METRIC_PRIORITY:
        if m in problem.metrics:
            return f'avg_{m}' if algo.track_average else m


def compute_local_lip(x: np.ndarray, y: np.ndarray,
                      x_tilde: np.ndarray, y_tilde: np.ndarray,
                      gx: np.ndarray, gy: np.ndarray,
                      gx_tilde: np.ndarray, gy_tilde: np.ndarray) -> float:
    """Estimate local Lipschitz constant L_t

    L_t = ||F(z_{t+1/2}) - F(z_t)|| / ||z_{t+1/2} - z_t||

    Args:
        x, y: Current point z_t
        x_tilde, y_tilde: Extrapolated point z_{t+1/2}
        gx, gy: Gradient at z_t
        gx_tilde, gy_tilde: Gradient at z_{t+1/2}

    Returns:
        Local Lipschitz constant estimate, or np.nan if undefined
    """
    # ||F(z_{t+1/2}) - F(z_t)|| where F = [gx; -gy]
    F_diff_x = gx_tilde - gx
    F_diff_y = -gy_tilde - (-gy)
    F_diff_norm = np.sqrt(np.sum(F_diff_x ** 2) + np.sum(F_diff_y ** 2))

    # ||z_{t+1/2} - z_t||
    z_diff_x = x_tilde - x
    z_diff_y = y_tilde - y
    z_diff_norm = np.sqrt(np.sum(z_diff_x ** 2) + np.sum(z_diff_y ** 2))

    return F_diff_norm / z_diff_norm


class SaddlePointAlgorithm(ABC):
    """Base class for saddle point algorithms"""

    def __init__(self, name: str, track_iterates: str = 'both'):
        """Initialize algorithm

        Respects each algorithm's track_iterates setting:
        - 'last': Only plot last iterate
        - 'average': Only plot average iterate
        - 'both': Plot both iterates

        Args:
            name: Algorithm name
            track_iterates: Which iterates to track ('last', 'average', or 'both')
        """
        self.name = name
        self.history = {
            'time': [],
            # Last iterate tracking
            'iterate_x': [],
            'iterate_y': [],
            'obj_value': [],
            'sp_gap': [],
            'nat_res': [],
            'dist_to_opt': [],
            'lb_diff': [],
            # Average iterate tracking
            'avg_x': [],
            'avg_y': [],
            'avg_obj_value': [],
            'avg_sp_gap': [],
            'avg_nat_res': [],
            'avg_dist_to_opt': [],
            'avg_lb_diff': [],
        }
        self.cached_gx = None
        self.cached_gy = None
        self.step_size = None

        # Whether to track last or average iterates
        self.track_last, self.track_average, self.track_w_average = False, False, False
        if track_iterates in ['last', 'both']:
            self.track_last = True
        if track_iterates in ['average', 'both', 'weighted']:
            self.track_average = True
        if track_iterates == 'weighted':
            self.track_w_average = True

    def initialize(self, problem: SaddlePointProblem, x0: np.ndarray, y0: np.ndarray) -> None:
        """Algorithm-specific initialization"""
        # TODO: make sure things that need to be reset for a new run are coded in this function
        pass

    @abstractmethod
    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one iteration step and return"""
        pass

    def optimize(self, problem: SaddlePointProblem,
                 x0: np.ndarray, y0: np.ndarray,
                 max_iter: int = 1000, tol: float = 1e-6,
                 verbose: bool = False) -> Dict:
        """Run saddle point algorithm

        Args:
            problem: Saddle point problem to solve
            x0, y0: Initial iterates
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress
        """

        # Select stopping metric
        convergence_metric = select_convergence_metric(problem, self)

        # Retrieve ground truth optimal solution if available
        if hasattr(problem, 'get_optimal_solution'):
            problem.get_optimal_solution()

        # Initialize
        self.initialize(problem, x0, y0)
        self.reset_history()
        x, y = x0.copy(), y0.copy()

        # Initialize average iterates
        if self.track_average:
            sum_x = np.zeros_like(x)
            sum_y = np.zeros_like(y)
            sum_weight = 0

        for i in range(max_iter):

            # Perform step
            t0 = time.perf_counter()
            x_new, y_new = self.step(x, y, problem, i)
            t1 = time.perf_counter()

            x, y = x_new, y_new

            for metric, compute_metric in problem.metrics.items():
                # Compute metrics available for this instance
                value = compute_metric(x, y)
                # Store history
                self.history[metric].append(value)

            # Store history
            obj_val = problem.objective(x, y)
            self.history['obj_value'].append(obj_val)
            self.history['iterate_x'].append(x.copy())
            self.history['iterate_y'].append(y.copy())
            self.history['time'].append(t1 - t0)  # incremental

            # Update and track AVERAGE iterate
            # TODO: efficient implementation of average computation
            if self.track_average:
                weight = 1.
                if self.track_w_average:
                    weight = self.step_size
                sum_x += x * weight
                sum_y += y * weight
                sum_weight += weight
                avg_x = sum_x / sum_weight
                avg_y = sum_y / sum_weight

                for metric, compute_metric in problem.metrics.items():
                    # Compute metrics for average iterate
                    value = compute_metric(avg_x, avg_y)
                    # Store history
                    self.history[f'avg_{metric}'].append(value)

                avg_obj_val = problem.objective(avg_x, avg_y)
                self.history['avg_obj_value'].append(avg_obj_val)
                self.history['avg_x'].append(avg_x.copy())
                self.history['avg_y'].append(avg_y.copy())

            # Check convergence
            if self.history[convergence_metric][-1] < tol:
                if verbose:
                    print(f"{self.name} converged at iteration {i} in terms of {convergence_metric} up to {tol}")
                break

            if verbose and (i % 100 == 0 or i < 10):
                print(f"Iter {i}: {convergence_metric} = {self.history[convergence_metric][-1]:.6f}, "
                      f"stepsize = {self.step_size: .5f}")

        result = {
            # 'final_x': x,
            # 'final_y': y,
            # 'final_obj': self.history['obj_values'][-1],
            'iterations': len(self.history[convergence_metric]),
            'converged': self.history[convergence_metric][-1] < tol,
            'convergence_metric': convergence_metric
        }

        for key, values in self.history.items():
            if len(values) > 0:
                result[key] = values[-1]

        result['time'] = np.sum(self.history['time'])

        return result

    def reset_history(self):
        """Clear history for new run"""
        self.history = {k: [] for k in self.history.keys()}


class SPMirrorDescent(SaddlePointAlgorithm):
    """Simultaneous gradient descent-ascent"""

    def __init__(self, lipschitz: float, diameter: float, max_iter: int):
        super().__init__(f"SPMD", track_iterates='average')
        self.step_size = 1 / lipschitz * diameter / np.sqrt(max_iter)

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        gx, gy = problem.gradient(x, y)
        self.cached_gx, self.cached_gy = gx, gy
        x_new = x - self.step_size * gx
        y_new = y + self.step_size * gy  # Gradient ascent for y

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        return x_new_proj, y_new_proj


class Extragradient(SaddlePointAlgorithm):
    """Extragradient method (Korpelevich, 1976)"""

    def __init__(self, lipschitz: float):
        super().__init__(f"EG", track_iterates='both')
        self.step_size = 1 / lipschitz

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        # Extrapolation step
        gx, gy = problem.gradient(x, y)
        self.cached_gx, self.cached_gy = gx, gy
        x_tilde = x - self.step_size * gx
        y_tilde = y + self.step_size * gy

        x_tilde, y_tilde = problem.project(x_tilde, y_tilde)

        # Correction step using gradient at extrapolated point
        gx_tilde, gy_tilde = problem.gradient(x_tilde, y_tilde)
        x_new = x - self.step_size * gx_tilde
        y_new = y + self.step_size * gy_tilde

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        return x_new_proj, y_new_proj


class UniversalMirrorProx(SaddlePointAlgorithm):
    """Universal Mirror Prox."""

    def __init__(self, diameter: float, G0: float):
        super().__init__(f"Universal MP", track_iterates='average')
        self.diameter = diameter
        self.G0 = G0
        self.step_size_aux = self.G0 ** 2
        self.step_size = self.diameter / self.G0

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        # Extrapolation step
        gx, gy = problem.gradient(x, y)
        self.cached_gx, self.cached_gy = gx, gy
        x_tilde = x - self.step_size * gx
        y_tilde = y + self.step_size * gy

        x_tilde, y_tilde = problem.project(x_tilde, y_tilde)

        # Correction step using gradient at extrapolated point
        gx_tilde, gy_tilde = problem.gradient(x_tilde, y_tilde)
        x_new = x - self.step_size * gx_tilde
        y_new = y + self.step_size * gy_tilde

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        # Update stepsize
        delta = np.sum((x_tilde - x) ** 2) + np.sum((y_tilde - y) ** 2) \
            + np.sum((x_new_proj - x_tilde) ** 2) + np.sum((y_new_proj - y_tilde) ** 2)
        delta = delta / 5 / self.step_size ** 2
        self.step_size_aux += delta
        self.step_size = self.diameter / np.sqrt(self.step_size_aux)

        return x_new_proj, y_new_proj


class AdaptiveMirrorProx(SaddlePointAlgorithm):
    """Extragradient method with adaptive stepsize."""

    def __init__(self, step_size: float, theta: float = 0.9):
        super().__init__(f"Adaptive MP", track_iterates='average')
        self.step_size = step_size
        self.theta = theta

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        # Extrapolation step
        gx, gy = problem.gradient(x, y)
        self.cached_gx, self.cached_gy = gx, gy
        x_tilde = x - self.step_size * gx
        y_tilde = y + self.step_size * gy

        x_tilde, y_tilde = problem.project(x_tilde, y_tilde)

        # Correction step using gradient at extrapolated point
        gx_tilde, gy_tilde = problem.gradient(x_tilde, y_tilde)
        x_new = x - self.step_size * gx_tilde
        y_new = y + self.step_size * gy_tilde

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        # Estimate local Lipschitz constant
        L_local = compute_local_lip(x, y, x_tilde, y_tilde,
                                    gx, gy, gx_tilde, gy_tilde)

        # Update step size: α_{t+1} = min(α_t, c / L_t)
        step_size_candidate = self.theta / L_local if L_local > 0 else np.inf
        self.step_size = min(self.step_size, step_size_candidate)

        return x_new_proj, y_new_proj


class AdaProx(SaddlePointAlgorithm):
    """Extragradient method with adaptive stepsize 2."""

    def __init__(self, step_size: float):
        super().__init__(f"AdaProx", track_iterates='average')
        self.step_size = step_size
        self.step_size_aux = 1

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        # Extrapolation step
        gx, gy = problem.gradient(x, y)
        self.cached_gx, self.cached_gy = gx, gy
        x_tilde = x - self.step_size * gx
        y_tilde = y + self.step_size * gy

        x_tilde, y_tilde = problem.project(x_tilde, y_tilde)

        # Correction step using gradient at extrapolated point
        gx_tilde, gy_tilde = problem.gradient(x_tilde, y_tilde)
        x_new = x - self.step_size * gx_tilde
        y_new = y + self.step_size * gy_tilde

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        # Update stepsize
        self.step_size_aux += np.sum((gx_tilde - gx) ** 2) + np.sum((gy_tilde - gy) ** 2)
        self.step_size = 1 / np.sqrt(self.step_size_aux)

        return x_new_proj, y_new_proj


class AGRAAL(SaddlePointAlgorithm):
    """aGRAAL (adaptive Golden Ratio Algorithm), Malitsky, 2018"""

    def __init__(self, step_size: float, phi: float, lmd_bar: float, ):
        super().__init__(f"aGRAAL", track_iterates='weighted')
        self.xbar, self.ybar = None, None
        self.x_prev, self.y_prev = None, None
        self.step_size = min(step_size, lmd_bar)
        self.phi = min(phi, (np.sqrt(5) + 1) / 2)
        self.lmd_bar = lmd_bar
        self.theta = 1.
        self.rho = 1. / self.phi + 1. / self.phi ** 2

    def initialize(self, problem: SaddlePointProblem, x1: np.ndarray, y1: np.ndarray):
        # Initialize z bar and compute its gradient
        self.xbar, self.ybar = x1, y1
        gx1, gy1 = problem.gradient(x1, y1)

        # Add a small perturbation to generate x0, y0
        eps = 1e-3
        x0, y0 = problem.project(x1 + eps * np.random.uniform(-1, 1),
                                 y1 + eps * np.random.uniform(-1, 1))
        gx0, gy0 = problem.gradient(x0, y0)

        # Compute local Lipschitz constant for initial stepsize
        L_local = compute_local_lip(x1, y1, x0, y0,
                                    gx1, gy1, gx0, gy0)
        self.step_size = 1 / L_local

        # Initialize z0
        self.x_prev, self.y_prev = x0, y0
        self.cached_gx, self.cached_gy = gx0, gy0

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        # Compute gradient
        gx, gy = problem.gradient(x, y)

        # Update stepsize
        L_local = compute_local_lip(x, y, self.x_prev, self.y_prev,
                                    gx, gy, self.cached_gx, self.cached_gy)
        step_size_new = min(
            self.rho * self.step_size,
            self.phi * self.theta / (4 * self.step_size * L_local ** 2),
            self.lmd_bar
        )
        self.cached_gx, self.cached_gy = gx, gy

        # Update x_prev, y_prev
        self.x_prev, self.y_prev = x, y

        # Update auxiliary iterate
        self.xbar = ((self.phi - 1) * x + self.xbar) / self.phi
        self.ybar = ((self.phi - 1) * y + self.ybar) / self.phi

        # Update iterate
        x_new = self.xbar - self.step_size * gx
        y_new = self.ybar + self.step_size * gy

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        # Update theta
        self.theta = step_size_new / self.step_size * self.phi
        self.step_size = step_size_new

        return x_new_proj, y_new_proj


class AdaPEG(SaddlePointAlgorithm):
    """AdaPEG algorithm for unbounded domains, Ene and Nyugen, 2021"""

    def __init__(self, step_size: float, eta: float = 1.):
        super().__init__(f"AdaPEG", track_iterates='average')
        self.gamma_prev = 0
        self.gamma = 1 / step_size
        self.eta = eta
        self.step_size_aux = (self.eta * self.gamma) ** 2
        self.xbar, self.ybar = None, None
        self.x0, self.y0 = None, None
        self.cached_xbar, self.cached_ybar = None, None

    def initialize(self, problem: SaddlePointProblem, x0: np.ndarray, y0: np.ndarray):
        self.xbar, self.ybar = x0, y0
        self.x0, self.y0 = x0, y0
        self.cached_gx, self.cached_gy = problem.gradient(self.x0, self.y0)

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """AdaPEG Update: A weighted average of the previous iterate and the anchor (x0),
        followed by an adaptive gradient-style step using the operator at w"""
        # Compute momentum used for both steps
        x_base = self.xbar + (1 - self.gamma_prev / self.gamma) * (self.x0 - self.xbar)
        y_base = self.ybar + (1 - self.gamma_prev / self.gamma) * (self.y0 - self.ybar)

        # Update iterate
        gx, gy = self.cached_gx, self.cached_gy
        x_new = x_base - 1 / self.gamma * gx
        y_new = y_base + 1 / self.gamma * gy

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        # Update auxiliary iterate
        gx_new, gy_new = problem.gradient(x_new_proj, y_new_proj)
        self.xbar = x_base - 1 / self.gamma * gx_new
        self.ybar = y_base + 1 / self.gamma * gy_new

        # Update stepsize
        self.gamma_prev = self.gamma
        self.step_size_aux += (gx_new - gx) @ (gx_new - gx) + (gy_new - gy) @ (gy_new - gy)
        self.gamma = np.sqrt(self.step_size_aux) / self.eta
        self.step_size = 1 / self.gamma
        self.cached_gx, self.cached_gy = gx_new, gy_new

        return x_new_proj, y_new_proj


class AdaptExtragradient(SaddlePointAlgorithm):
    """Extragradient method with adaptive stepsize."""

    def __init__(self, step_size: float, theta: float = 0.9):
        super().__init__(f"Adapt EG", track_iterates='last')
        self.step_size = step_size
        self.step_size_aux = 1 / self.step_size ** 2

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        # Extrapolation step
        gx, gy = problem.gradient(x, y)
        self.cached_gx, self.cached_gy = gx, gy
        x_tilde = x - self.step_size * gx
        y_tilde = y + self.step_size * gy

        x_tilde, y_tilde = problem.project(x_tilde, y_tilde)

        # Correction step using gradient at extrapolated point
        gx_tilde, gy_tilde = problem.gradient(x_tilde, y_tilde)
        x_new = x - self.step_size * gx_tilde
        y_new = y + self.step_size * gy_tilde

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        # Update step size
        self.step_size_aux += (iteration + 1) * ((x_new_proj - x_tilde) @ (x_new_proj - x_tilde) +
                                                 (y_new_proj - y_tilde) @ (y_new_proj - y_tilde))
        self.step_size = 1 / np.sqrt(self.step_size_aux)

        return x_new_proj, y_new_proj


class PfNeEg(SaddlePointAlgorithm):
    """Extragradient method with adaptive stepsize."""

    def __init__(self, step_size: float = 0.5, theta: float = 0.9):
        super().__init__(f"PF-NE-EG", track_iterates='last')
        self.step_size = step_size
        self.theta = theta
        self.prev_x_tilde = None
        self.prev_y_tilde = None
        self.prev_gx_tilde = None
        self.prev_gy_tilde = None

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:

        # Extrapolation step
        gx, gy = problem.gradient(x, y)
        self.cached_gx, self.cached_gy = gx, gy

        # Update stepsize
        if self.prev_x_tilde is not None:
            L_local = compute_local_lip(x, y, self.prev_x_tilde, self.prev_y_tilde,
                                        gx, gy, self.prev_gx_tilde, self.prev_gy_tilde)
            self.step_size = min(self.step_size, self.theta / L_local)

        # Extrapolation step
        x_tilde = x - self.step_size * gx
        y_tilde = y + self.step_size * gy

        x_tilde, y_tilde = problem.project(x_tilde, y_tilde)

        # Correction step using gradient at extrapolated point
        gx_tilde, gy_tilde = problem.gradient(x_tilde, y_tilde)
        x_new = x - self.step_size * gx_tilde
        y_new = y + self.step_size * gy_tilde

        x_new_proj, y_new_proj = problem.project(x_new, y_new)

        # Update stepsize
        L_local = compute_local_lip(x, y, x_tilde, y_tilde,
                                    gx, gy, gx_tilde, gy_tilde)
        mult = 1 + 1 / np.log(iteration + 2)
        self.step_size = min(self.step_size * mult, self.theta / L_local)

        # Store current state for next iteration
        self.prev_x_tilde = x_tilde.copy()
        self.prev_y_tilde = y_tilde.copy()
        self.prev_gx_tilde = gx_tilde.copy()
        self.prev_gy_tilde = gy_tilde.copy()

        return x_new_proj, y_new_proj


class PfNeEgBacktracking(SaddlePointAlgorithm):
    """Extragradient method with backtracking linesearch stepsize."""

    def __init__(self, step_size: float = 0.5, mult: float = 0.9, theta: float = 0.9):
        super().__init__(f"PF-NE-EG bt", track_iterates='last')
        self.step_size = step_size
        self.mult = mult
        self.theta = theta
        self.max_backtracks = 20  # max_backtracks

    def step(self, x: np.ndarray, y: np.ndarray,
             problem: SaddlePointProblem, iteration: int) -> Tuple[np.ndarray, np.ndarray]:

        step_size = self.step_size / self.mult

        for backtrack_iter in range(self.max_backtracks):

            # Extrapolation step with current alpha
            if iteration > 0:
                gx, gy = self.cached_gx, self.cached_gy
            else:
                gx, gy = problem.gradient(x, y)

            # Extrapolation step
            x_tilde = x - self.step_size * gx
            y_tilde = y + self.step_size * gy

            x_tilde, y_tilde = problem.project(x_tilde, y_tilde)

            # Correction step using gradient at extrapolated point
            gx_tilde, gy_tilde = problem.gradient(x_tilde, y_tilde)
            x_new = x - self.step_size * gx_tilde
            y_new = y + self.step_size * gy_tilde

            x_new_proj, y_new_proj = problem.project(x_new, y_new)

            # Operator applied to new iterate for stepsize update
            gx_new, gy_new = problem.gradient(x_new_proj, y_new_proj)

            # Check backtracking condition using local Lipschitz constants
            L0 = compute_local_lip(x, y, x_tilde, y_tilde,
                                   gx, gy, gx_tilde, gy_tilde)
            L1 = compute_local_lip(x_new_proj, y_new_proj, x_tilde, y_tilde,
                                   gx_new, gy_new, gx_tilde, gy_tilde)
            if (step_size * L0 <= self.theta) and (step_size * L1 <= 1.0):
                break

            # Backtrack: reduce step size
            step_size *= self.mult

        # Update stepsize
        self.step_size = step_size

        # Store current state for next iteration
        self.cached_gx = gx_new.copy()
        self.cached_gy = gy_new.copy()

        return x_new_proj, y_new_proj

