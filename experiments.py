from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
import os

from problems import SaddlePointProblem
from algorithms import SaddlePointAlgorithm


plt.rcParams.update({
    'font.family': 'Arial',
    'figure.figsize': (5, 4)
})


# distinguish server (no interactive plot), Mac (default GUI), and PC (specify GUI)
is_on_server = 'HOSTNAME' in os.environ
save_fig = not is_on_server
if save_fig:
    import matplotlib
    if 'zsh' not in os.environ['SHELL']:
        matplotlib.use("TkAgg")


# TODO: wrap metric names, functions etc into a class
METRIC_NAME_MAPPING = {
    'sp_gap': 'Saddle-point gap',
    'dist_to_opt': 'Distance to proxy optimal solution',
    'lb_diff': r'lower bound $-$ best lower bound',
    'nat_res': r'natural residual'
}
ALGO_COLOR_MAPPING = {
    'EG': 'tab:blue',
    'PF-NE-EG': 'tab:orange',
    'PF-NE-EG Bt': 'tab:red',
    'PF-NE-EG AdaBt': 'tab:brown',
    'Adapt EG': 'tab:green',
    'Universal MP': 'tab:gray',
    'Adaptive MP': 'tab:pink',
    'AdaProx': 'tab:olive',
    'aGRAAL': 'tab:purple',
    'AdaPEG': 'tab:cyan',
}

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    max_iter: int = 1000
    tol: float = 1e-6
    max_time: int = 600
    num_trials: int = 1
    seed: int = 42
    verbose: bool = False
    save_path: str = 'output'


class ExperimentRunner:
    """Run and compare multiple algorithms on multiple problems"""

    def __init__(self, config: ExperimentConfig = ExperimentConfig()):
        self.config = config
        self.results = {}

    def run_experiment(self, algorithms: List[SaddlePointAlgorithm],
                       problems: List[SaddlePointProblem],
                       problem_names: List[str]) -> Dict:
        """Run all algorithms on all problems"""

        for prob_name, problem in zip(problem_names, problems):
            print(f"\n{'=' * 60}")
            print(f"Problem: {prob_name} (dim_x={problem.dim_x}, dim_y={problem.dim_y})")
            print(f"{'=' * 60}")

            self.results[prob_name] = {}

            for algo_template in algorithms:
                algo = copy.deepcopy(algo_template)

                print(f"\nRunning {algo.name}...")
                trial_results = []

                for trial in range(self.config.num_trials):
                    x0, y0 = problem.initial_point()

                    result = algo.optimize(
                        problem, x0, y0,
                        max_iter=self.config.max_iter,
                        tol=self.config.tol,
                        max_time=self.config.max_time,
                        verbose=self.config.verbose
                    )

                    trial_results.append({
                        'result': result,
                        'history': algo.history.copy(),
                        # Store algorithm settings for plotting
                        'track_last': algo.track_last,
                        'track_average': algo.track_average
                    })

                self.results[prob_name][algo.name] = trial_results

                # Print summary
                print('\n')
                for metric in problem.metrics:
                    values = [t['result'][metric] for t in trial_results]
                    print(f"  {metric} = {np.mean(values):.6e} ± {np.std(values):.6e}, ")
                    if algo.track_average:
                        metric = 'avg_' + metric
                        values = [t['result'][metric] for t in trial_results]
                        print(f"  {metric} = {np.mean(values):.6e} ± {np.std(values):.6e}, ")

                times = [t['result']['time'] for t in trial_results]
                print(f"  Time = {np.mean(times):.4f} ± {np.std(times):.4f} sec")
                converged = [t['result']['converged'] for t in trial_results]
                print(f"  Converged: {sum(converged)}/{len(converged)} trials")

        return self.results

    def plot_convergence(self, problem_name: str = None, metric_to_plot: str = 'auto', use_log: bool = True,
                         show_legend: bool = True):
        """Plot convergence curves for a specific problem

        Args:
            problem_name: Name of problem to plot
            metric_to_plot: Name of metric to plot
            use_log: Use log scale for plots
            show_legend: Show legend or not
        """
        if problem_name is None:
            # take any problem from results
            problem_name = next(iter(self.results))
        elif problem_name not in self.results:
            print(f"No results for problem: {problem_name}")
            return

        # Single plot for saddle point gap
        fig, ax = plt.subplots(1, 1)

        for algo_name_label, trials in self.results[problem_name].items():

            # Get plot preference from first trial's algorithm
            # (all trials use same algorithm instance settings)
            track_last = trials[0]['track_last']
            track_average = trials[0]['track_average']

            if metric_to_plot == 'auto':
                metric_to_plot = trials[0]['result']['convergence_metric']
                if metric_to_plot.startswith('avg_'):
                    metric_to_plot = metric_to_plot[4:]

            variants = []
            if track_last:
                variants.append(("last", metric_to_plot, '-', 0.7))
            if track_average:
                variants.append(("avg", f"avg_{metric_to_plot}", '--', 0.7))

            for last_avg, metric, linestyle, alpha in variants:

                max_len = max(len(t['history'][metric]) for t in trials)
                if max_len == 0:
                    continue

                # Auto-downsample if too many iterations
                # if max_len > 1000:
                #     downsample = max_len // 500  # Keep ~500 points
                # else:
                #     downsample = 1
                downsample = 1

                convergence_curves = np.full((len(trials), max_len), np.nan)
                for i, trial in enumerate(trials):
                    n = len(trial['history'][metric])
                    convergence_curves[i, :n] = trial['history'][metric]
                mean_convergence = np.nanmean(convergence_curves, axis=0)

                # Downsample for efficiency
                iters = np.arange(len(mean_convergence))[::downsample]
                mean_convergence = mean_convergence[::downsample]

                # Plot saddle point gap
                plot = ax.loglog if use_log else ax.plot
                last_avg = 'avg' if metric.startswith('avg_') else 'last'
                plot(iters, mean_convergence, label=f"{algo_name_label} ({last_avg})",
                     linewidth=2, alpha=alpha, linestyle=linestyle, color=ALGO_COLOR_MAPPING[algo_name_label])

        # Configure plot
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(f'{METRIC_NAME_MAPPING[metric_to_plot]}', fontsize=12)
        ax.set_title(f'{problem_name}', fontsize=14)
        if show_legend:
            legend = ax.legend(fontsize=10)
            # legend.set_zorder(1)
            legend.get_frame().set_alpha(0.5)
        ax.set_ylim(bottom=1e-6)  #, top=40)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        # Save or show
        if not is_on_server:
            fig.show()
        if save_fig:
            filepath = f'{self.config.save_path}/{problem_name}.pdf'
            fig.savefig(filepath, bbox_inches='tight')
            print(f"Figure saved to: {filepath}")
            plt.close(fig)  # Close to free memory

    def time_table(self, prob_name: str = None):

        if prob_name is None:
            # take any problem from results
            prob_name = next(iter(self.results))

        # Extract mean times for each algorithm
        algo_times = {}
        for algo_name, trials in self.results[prob_name].items():
            # take the time entry
            time_val = np.mean(trials[0]['result']['time'])
            algo_times[algo_name] = time_val

        # Generate LaTeX table
        latex_lines = []
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\begin{tabular}{lc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"\textbf{Algorithm} & \textbf{Time (s)} \\")
        latex_lines.append(r"\midrule")

        for algo_name, time_val in algo_times.items():
            latex_lines.append(f"{algo_name} & {time_val:.2f} \\\\")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(
            r"\caption{Solution time (seconds) comparison for solving ...}")
        latex_lines.append(r"\label{tab:-time}")
        latex_lines.append(r"\end{table}")

        # Output LaTeX table as a string
        latex_table = "\n".join(latex_lines)
        print(latex_table)

    def save_results(self, filepath: str):
        """Save results using pickle"""

        pkl_results = {}
        for prob_name, prob_results in self.results.items():
            pkl_results[prob_name] = {}
            for algo_name, trials in prob_results.items():
                pkl_results[prob_name][algo_name] = {'time': [t['history']['time'] for t in trials]}

                # Save the convergence metrics that are available
                for metric in METRIC_NAME_MAPPING:
                    first_trial_history = trials[0]['history']

                    variants = []
                    if trials[0]['track_last']:
                        variants.append(metric)
                    if trials[0]['track_average']:
                        variants.append(f"avg_{metric}")

                    for metric_name in variants:
                        if metric_name in first_trial_history and len(first_trial_history[metric_name]) > 0:
                            pkl_results[prob_name][algo_name][metric_name] = [t['history'][metric_name] for t in trials]

        filepath = f'{self.config.save_path}/{filepath}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(pkl_results, f)

        print(f"\nResults saved to {filepath}")

    def load_results(self, filepath: str):

        filepath = f'{self.config.save_path}/{filepath}.pkl'
        with open(filepath, 'rb') as f:
            loaded_results = pickle.load(f)

        reconstructed_results = {}
        for prob_name, prob_data in loaded_results.items():
            reconstructed_results[prob_name] = {}
            for algo_name, algo_data in prob_data.items():
                n_trials = len(algo_data['time'])
                trials = []
                for i in range(n_trials):
                    trial_history = {'time': algo_data['time'][i]}
                    for key, values in algo_data.items():
                        if key != 'time':
                            trial_history[key] = values[i]
                    avg_metrics = [s for s in algo_data.keys() if s.startswith('avg_')]
                    last_metrics = [s for s in algo_data.keys() if not s.startswith('avg_')]
                    # Add 'result': last value of each metric list in history
                    trial_result = {k: v[-1] for k, v in trial_history.items()}
                    trial_result['time'] = np.sum(trial_history['time'])
                    trial = {
                        'result': trial_result,
                        'history': trial_history,
                        'track_last': len(last_metrics) > 1,
                        'track_average': len(avg_metrics) > 0,  # True if avg_ metric exists
                    }
                    trials.append(trial)
                reconstructed_results[prob_name][algo_name] = trials

        # Assign back to self.results
        self.results = reconstructed_results
