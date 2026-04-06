from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *

import os


if __name__ == "__main__":

    step_size_guess = .01
    M_norm = 1 / step_size_guess * 2

    for ng, ns, nf, show_legend in [
        (20, 1000, 50, True),
        (10, 1000, 100, False),
    ]:

        # Define problems
        fairness = GroupFairnessClassification(n_groups=ng, n_samples_per_group=ns, n_features=nf)
        problems = [fairness]
        problem_names = [f'GroupFairness-({fairness.n_groups}x{fairness.n_samples_per_group})'
                         f'x{fairness.n_features}']

        if 'HOSTNAME' in os.environ:

            # Setup
            config = ExperimentConfig(max_iter=100000, tol=1e-6, max_time=60, num_trials=1, verbose=False)
            runner = ExperimentRunner(config)

            # Define algorithms
            algorithms = [
                PfNeEg(step_size=step_size_guess),
                PfNeEgBacktracking(step_size=step_size_guess),
                PfNeEgAdaBacktracking(step_size=step_size_guess),
            ]

            # Run experiments
            results = runner.run_experiment(algorithms, problems, problem_names)

            # Save results
            runner.save_results('+'.join(problem_names) + f'+iter{config.max_iter}')

        else:

            # Setup
            config = ExperimentConfig(max_iter=10000, tol=1e-6, max_time=600, num_trials=1, verbose=True)
            runner = ExperimentRunner(config)

            # Define algorithms
            algorithms = [
                AGRAAL(step_size=step_size_guess, phi=2., lmd_bar=step_size_guess),
                PfNeEg(step_size=step_size_guess),
                PfNeEgBacktracking(step_size=step_size_guess),
                PfNeEgAdaBacktracking(step_size=step_size_guess),
            ]

            # Run experiments
            results = runner.run_experiment(algorithms, problems, problem_names)

            # Visualize results
            for prob_name in problem_names:
                runner.plot_convergence(prob_name, metric_to_plot='nat_res', show_legend=show_legend)
