from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *


def run(experiment_type: str, run_or_load: str):

    # Setup
    if experiment_type == 'plot':
        config = ExperimentConfig(max_iter=10000, tol=1e-6, verbose=False)
        show_fig, save_fig = True, True
    elif experiment_type == 'time':
        config = ExperimentConfig(max_iter=1000000, tol=1e-5, max_time=60, verbose=False)
        show_fig, save_fig = False, False
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    runner = ExperimentRunner(config)

    for dim, sparsity, step_size_guess, show_legend, filename_suffix in [
        (100, 1., 0.5, False, ''),
        (100, 1., .02, True, '-smallstep'),
        (500, .2, 0.5, False, ''),
        (1000, .1, 0.5, False, '')
    ]:

        # Define problems
        matrix_game = MatrixGameProblem(dim_x=dim, dim_y=dim, sparsity=sparsity, seed=config.seed)
        problems = [matrix_game]
        problem_names = [f"MatrixGame-{matrix_game.dim_x}x{matrix_game.dim_y}-sparsity{matrix_game.sparsity}"
                         f"{filename_suffix}"]
        # Compute proper step size: α < 1/||M||
        M_norm = np.linalg.norm(matrix_game.M, ord=2)
        diameter = np.sqrt(2)

        # Define algorithms
        algorithms = [
            Extragradient(lipschitz=M_norm),
            UniversalMirrorProx(diameter=diameter, G0=M_norm),
            AdaptiveMirrorProx(step_size=step_size_guess),
            AdaProx(step_size=step_size_guess),
            AGRAAL(step_size=step_size_guess, phi=2., lmd_bar=step_size_guess),
            AdaPEG(step_size=step_size_guess),
            AdaptExtragradient(step_size=step_size_guess),
            PfNeEg(step_size=step_size_guess),
            PfNeEgBacktracking(step_size=step_size_guess),
            PfNeEgAdaBacktracking(step_size=step_size_guess),
        ]

        if experiment_type == 'time':
            algorithms = algorithms[1:]

        if run_or_load == 'run':

            # Run experiments
            runner.run_experiment(algorithms, problems, problem_names)

            # Save results
            runner.save_results('+'.join(problem_names) + f'+iter{config.max_iter}')

        else:

            config.save_path = 'output_server'

            runner.load_results('+'.join(problem_names) + f'+iter{config.max_iter}')

            if experiment_type == 'time':
                runner.time_table()

        # Visualize results, regardless of run or load, if experiment is to show plot
        if experiment_type == 'plot':
            for prob_name in problem_names:
                runner.plot_convergence(prob_name, metric_to_plot='sp_gap', show_legend=show_legend,
                                        show_fig=show_fig, save_fig=save_fig)
