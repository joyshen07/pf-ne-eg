from experiments import ExperimentConfig, ExperimentRunner
from problems import *


if __name__ == "__main__":

    # Setup
    config = ExperimentConfig(max_iter=10000, tol=1e-6, num_trials=1, verbose=True, save_path='output')
    runner = ExperimentRunner(config)

    # Define problems
    matrix_game = MatrixGameProblem(dim_x=100, dim_y=100, sparsity=1., seed=config.seed)
    problems = [matrix_game]
    problem_names = [f"MatrixGame-{matrix_game.dim_x}x{matrix_game.dim_y}-sparsity{matrix_game.sparsity}"]
    # Compute proper step size: α < 1/||M||
    M_norm = np.linalg.norm(matrix_game.M, ord=2)
    diameter = np.sqrt(2)

    # lasso = LASSO(dim_x=500, dim_y=5000, sparsity=.1)
    # problems = [lasso]
    # problem_names = [f'LASSO-{lasso.dim_x}x{lasso.dim_y}-sparsity{lasso.sparsity}']

    # fairness = GroupFairnessClassification(n_groups=20, n_samples_per_group=200, n_features=50)
    # problems = [fairness]
    # problem_names = [f'GroupFairness-({fairness.n_groups}x{fairness.n_samples_per_group})'
    #                  f'x{fairness.n_features}']

    # linx_d = LinxDoubleScaling(d=124, s=90)
    # problems = [linx_d]
    # problem_names = [f'LinxDoubleScaling-{linx_d.d}-{linx_d.s}']

    # gamma_star = GammaStar(d=124, s=60)
    # problems = [gamma_star]
    # problem_names = [f'GammaStar-{gamma_star.d}-{gamma_star.s}']

    runner.load_results('+'.join(problem_names) + f'+iter{config.max_iter}')

    runner.time_table()

    # Visualize results
    for prob_name in problem_names:
        runner.plot_convergence(prob_name, metric_to_plot='sp_gap')