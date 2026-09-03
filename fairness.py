from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *


def run(experiment_type: str, run_or_load: str):

    step_size_guess = .01
    M_norm = 1 / step_size_guess * 2

    # Setup
    if experiment_type == 'plot':
        config = ExperimentConfig(max_iter=10000, tol=1e-6, verbose=False)
        show_fig, save_fig = True, True

        algorithms = [
            AGRAAL(step_size=step_size_guess, phi=2., lmd_bar=step_size_guess),
            PfNeEg(step_size=step_size_guess),
            PfNeEgBacktracking(step_size=step_size_guess),
            PfNeEgAdaBacktracking(step_size=step_size_guess),
        ]

    elif experiment_type == 'time':
        config = ExperimentConfig(max_iter=100000, tol=1e-6, max_time=60, verbose=False)
        show_fig, save_fig = False, False

        algorithms = [
            PfNeEg(step_size=step_size_guess),
            PfNeEgBacktracking(step_size=step_size_guess),
            PfNeEgAdaBacktracking(step_size=step_size_guess),
        ]

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    runner = ExperimentRunner(config)

    for ng, ns, nf, show_legend in [
        (20, 200, 50, True),
        (10, 200, 100, False),
    ]:

        # Define problems
        fairness = GroupFairnessClassification(n_groups=ng, n_samples_per_group=ns, n_features=nf)
        problems = [fairness]
        problem_names = [f'GroupFairness-({fairness.n_groups}x{fairness.n_samples_per_group})'
                         f'x{fairness.n_features}']

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
                runner.plot_convergence(prob_name, metric_to_plot='nat_res', show_legend=show_legend,
                                        show_fig=show_fig, save_fig=save_fig)
