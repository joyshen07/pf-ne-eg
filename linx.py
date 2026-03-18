from experiments import ExperimentConfig, ExperimentRunner
from algorithms import *
from problems import *


if __name__ == "__main__":
    # Setup
    config = ExperimentConfig(max_iter=6000, tol=1e-1, num_trials=1, verbose=True)
    runner = ExperimentRunner(config)

    # Define problems
    problems = []
    problem_names = []
    d = 124
    s_range = [s for s in range(20, 101, 10)]
    for s in s_range:
        problems.append(LinxDoubleScaling(d=d, s=s))
        problem_names.append(f'LinxDoubleScaling-{d}-{s}')

    step_size_guess = 0.1
    M_norm = 1 / step_size_guess / .5

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
    ]

    # Run experiments
    results = runner.run_experiment(algorithms, problems, problem_names)

    # # Visualize results
    # for prob_name in problem_names:
    #     runner.plot_convergence(prob_name)

    # Save results
    runner.save_results(f'LinxDoubleScaling-{d}-{s_range[0]}to{s_range[-1]}' + f'-tol{config.tol}')
