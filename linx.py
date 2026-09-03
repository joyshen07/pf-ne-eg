from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *


def run(experiment_type: str, run_or_load: str):

    step_size_guess = 0.1
    M_norm = 1 / step_size_guess / .5

    # Setup
    if experiment_type == 'plot':
        s_range = [60, 90]
        config = ExperimentConfig(max_iter=10000, tol=1e-6, verbose=False)
        show_fig, save_fig = True, True

        algorithms = [
            Extragradient(lipschitz=M_norm, track_iterates='last'),
            AdaptiveMirrorProx(step_size=step_size_guess),
            AdaProx(step_size=step_size_guess),
            AGRAAL(step_size=step_size_guess, phi=2., lmd_bar=step_size_guess),
            AdaPEG(step_size=step_size_guess),
            AdaptExtragradient(step_size=step_size_guess),
            PfNeEg(step_size=step_size_guess),
            PfNeEgBacktracking(step_size=step_size_guess),
            PfNeEgAdaBacktracking(step_size=step_size_guess)
        ]

    elif experiment_type == 'time':
        s_range = [s for s in range(30, 101, 10)]
        config = ExperimentConfig(max_iter=200000, tol=5*1e-4, max_time=600, verbose=True)
        show_fig, save_fig = False, False

        algorithms = [
            Extragradient(lipschitz=M_norm, track_iterates='last'),
            AdaptExtragradient(step_size=step_size_guess),
            PfNeEg(step_size=step_size_guess),
            PfNeEgBacktracking(step_size=step_size_guess),
            PfNeEgAdaBacktracking(step_size=step_size_guess)
        ]

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    runner = ExperimentRunner(config)

    # Define problems
    problems = []
    problem_names = []
    d = 124
    for s in s_range:
        problems.append(LinxDoubleScaling(d=d, s=s))
        problem_names.append(f'LinxDoubleScaling-{d}-{s}')

    if run_or_load == 'run':

        # Run experiments
        runner.run_experiment(algorithms, problems, problem_names)

        # Save results
        runner.save_results(f'LinxDoubleScaling-{d}-{s_range[0]}to{s_range[-1]}' + f'-tol{config.tol}')

    else:

        config.save_path = 'output_server'

        runner.load_results(f'LinxDoubleScaling-{d}-{s_range[0]}to{s_range[-1]}' + f'-tol{config.tol}')

    # Visualize results
    if experiment_type == 'plot':
        # Visualize results
        for prob_name in problem_names:
            runner.plot_convergence(prob_name, show_legend=('60' in prob_name),
                                    show_fig=show_fig, save_fig=save_fig)
