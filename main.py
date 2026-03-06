from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *


if __name__ == "__main__":
    # Setup
    config = ExperimentConfig(max_iter=1000, tol=1e-6, num_trials=1, verbose=True)
    runner = ExperimentRunner(config)

    # Define problems
    # matrix_game = MatrixGameProblem(dim_x=500, dim_y=500, sparsity=0.2, seed=config.seed)
    # problems = [matrix_game]
    # problem_names = [f"MatrixGame-{matrix_game.dim_x}x{matrix_game.dim_y}-sparsity{matrix_game.sparsity}"]

    # # Compute proper step size: α < 1/||M||
    # M_norm = np.linalg.norm(matrix_game.M, ord=2)
    # step_size_safe = 0.9 / M_norm  # Use 90% of theoretical limit
    # print(f"\nMatrix M operator norm: {M_norm:.4f}")
    # print(f"Theoretical step size limit: {1/M_norm:.4f}")
    # print(f"Using step size: {step_size_safe:.4f}")

    # linx_d = LinxDoubleScaling(d=124, s=60)
    # problems = [linx_d]
    # problem_names = [f'LinxDoubleScaling-{linx_d.d}-{linx_d.s}']

    gamma_star = GammaStar(d=124, s=60)
    problems = [gamma_star]
    problem_names = [f'GammaStar-{gamma_star.d}-{gamma_star.s}']

    # step_size_safe = 0.1
    M_norm = 50
    diameter = np.sqrt(2)
    step_size_guess = 0.4

    # Define algorithms
    algorithms = [
        SimultaneousGradientDescent(lipschitz=M_norm, diameter=diameter, max_iter=config.max_iter),
        # AlternatingGradientDescent(step_size=0.01),
        Extragradient(lipschitz=M_norm),
        AdaptiveErgodicExtragradient(step_size=step_size_guess),
        AdaProx(step_size=step_size_guess),
        UniversalMirrorProx(diameter=diameter, G0=M_norm),
        AdaptiveExtragradient(step_size=step_size_guess),
        BacktrackingExtragradient(step_size=step_size_guess),
        # OptimisticGradient(step_size=0.01),
        # ExtraGradientWithMomentum(step_size=0.01, momentum=0.9)
    ]

    # Run experiments
    results = runner.run_experiment(algorithms, problems, problem_names)

    # Visualize results
    for prob_name in problem_names:
        runner.plot_convergence(prob_name)

    # Save results
    runner.save_results('+'.join(problem_names) + f'+iter{config.max_iter}')
