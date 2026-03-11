from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *


if __name__ == "__main__":
    # Setup
    config = ExperimentConfig(max_iter=10000, tol=1e-6, num_trials=1, verbose=True)
    runner = ExperimentRunner(config)

    # # Define problems
    # matrix_game = MatrixGameProblem(dim_x=1000, dim_y=1000, sparsity=.1, seed=config.seed)
    # problems = [matrix_game]
    # problem_names = [f"MatrixGame-{matrix_game.dim_x}x{matrix_game.dim_y}-sparsity{matrix_game.sparsity}"]
    # # Compute proper step size: α < 1/||M||
    # M_norm = np.linalg.norm(matrix_game.M, ord=2)
    # diameter = np.sqrt(2)

    # lasso = LASSO(dim_x=250, dim_y=1000, sparsity=.5)
    # problems = [lasso]
    # problem_names = [f'LASSO-{lasso.dim_x}x{lasso.dim_y}-sparsity{lasso.sparsity}']

    fairness = GroupFairnessClassification(n_groups=20, n_samples_per_group=100, n_features=100)
    problems = [fairness]
    problem_names = [f'GroupFairness-({fairness.n_groups}x{fairness.n_samples_per_group})'
                     f'x{fairness.n_features}']

    # linx_d = LinxDoubleScaling(d=124, s=60)
    # problems = [linx_d]
    # problem_names = [f'LinxDoubleScaling-{linx_d.d}-{linx_d.s}']

    # gamma_star = GammaStar(d=124, s=60)
    # problems = [gamma_star]
    # problem_names = [f'GammaStar-{gamma_star.d}-{gamma_star.s}']

    step_size_guess = .8
    M_norm = 50

    # Define algorithms
    algorithms = [
        # SPMirrorDescent(lipschitz=M_norm, diameter=diameter, max_iter=config.max_iter),
        Extragradient(lipschitz=M_norm),
        AdaptExtragradient(step_size=step_size_guess),
        PfNeEg(step_size=step_size_guess),
        PfNeEgBacktracking(step_size=step_size_guess),
        # UniversalMirrorProx(diameter=diameter, G0=M_norm),
        AdaptiveMirrorProx(step_size=step_size_guess),
        AdaProx(step_size=step_size_guess),
        AGRAAL(step_size=step_size_guess, phi=2., lmd_bar=step_size_guess),
        AdaPEG(step_size=step_size_guess),
        # OptimisticGradient(step_size=0.01),
    ]

    # Run experiments
    results = runner.run_experiment(algorithms, problems, problem_names)

    # Visualize results
    for prob_name in problem_names:
        runner.plot_convergence(prob_name)

    # Save results
    runner.save_results('+'.join(problem_names) + f'+iter{config.max_iter}')
