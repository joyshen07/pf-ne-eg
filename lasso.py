from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *

import os


if __name__ == "__main__":

    step_size_guess = .1
    M_norm = 1 / step_size_guess / .5

    for dim_x, dim_y, sparsity, show_legend in [
        (250, 1000, 0.5, True),
        (500, 5000, 0.1, False)
    ]:

        # Define problems
        lasso = LASSO(dim_x=dim_x, dim_y=dim_y, sparsity=sparsity)
        problems = [lasso]
        problem_names = [f'LASSO-{lasso.dim_x}x{lasso.A_dim_y}-sparsity{lasso.sparsity}']

        if 'HOSTNAME' in os.environ:

            # Setup
            config = ExperimentConfig(max_iter=100000, tol=1e-6, max_time=60, num_trials=1, verbose=False)
            runner = ExperimentRunner(config)

            # Define algorithms
            algorithms = [
                Extragradient(lipschitz=M_norm, track_iterates='last'),
                AdaptExtragradient(step_size=step_size_guess),
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
            config = ExperimentConfig(max_iter=10000, tol=1e-6, max_time=600, num_trials=1, verbose=False)
            runner = ExperimentRunner(config)

            # Define algorithms
            algorithms = [
                Extragradient(lipschitz=M_norm),
                AdaptiveMirrorProx(step_size=step_size_guess),
                AdaProx(step_size=step_size_guess),
                AGRAAL(step_size=step_size_guess, phi=2., lmd_bar=step_size_guess),
                AdaPEG(step_size=step_size_guess),
                AdaptExtragradient(step_size=step_size_guess),
                PfNeEg(step_size=step_size_guess),
                PfNeEgBacktracking(step_size=step_size_guess),
                PfNeEgAdaBacktracking(step_size=step_size_guess),
            ]

            # Run experiments
            results = runner.run_experiment(algorithms, problems, problem_names)

            # Visualize results
            for prob_name in problem_names:
                runner.plot_convergence(prob_name, show_legend=show_legend)

            # Save results
            runner.save_results('+'.join(problem_names) + f'+iter{config.max_iter}')
