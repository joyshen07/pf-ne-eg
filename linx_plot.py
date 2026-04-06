import numpy as np
import pickle
import matplotlib.pyplot as plt

from experiments import ALGO_COLOR_MAPPING


plt.rcParams.update({
    'font.family': 'Arial',
    'figure.figsize': (5, 4)
})


if __name__ == '__main__':

    filepath = "output_server/LinxDoubleScaling-124-20to100-tol0.0005.pkl"
    with open(filepath, "rb") as f:
        results = pickle.load(f)

    for tol in [float(filepath.split('-')[-1][3:-4])]:

        hit_records = {}

        for prob_name, prob_results in results.items():

            d = int(prob_name.split('-')[-2])
            s = int(prob_name.split('-')[-1])

            if s == 20:
                continue

            for algo_name, trials in prob_results.items():

                for metric_name in ['nat_res', 'avg_nat_res']:

                    if metric_name not in trials:
                        continue
                    if 'avg' in metric_name:
                        algo_name_label = algo_name + ' (avg)'
                    else:
                        algo_name_label = algo_name + ' (last)'

                    if algo_name_label not in hit_records:
                        hit_records[algo_name_label] = {'size': [], 'iter': [], 'time': []}

                    values = trials[metric_name][0]
                    times = trials['time'][0]

                    hits = np.where(np.array(values) <= tol)[0]

                    if len(hits) > 0:
                        hit_iter = hits[0]
                        hit_time = np.sum(times[:hit_iter])

                        hit_records[algo_name_label]['iter'].append(hit_iter)
                        hit_records[algo_name_label]['time'].append(hit_time)
                        hit_records[algo_name_label]['size'].append(s)

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        for algo_name_label, hit_record in hit_records.items():

            # use a different marker to distinguish PF-NE-EG with backtracking
            marker = 'x' if 'Bt' in algo_name_label else 'o'
            alpha = 0.7 if 'Bt' in algo_name_label else 0.9
            if 'AdaBt' in algo_name_label:
                marker = 'P'
                alpha = 0.7
            if algo_name_label == 'PF-NE-EG (last)':
                marker = 's'

            sizes = hit_record['size']
            iters = hit_record['iter']
            times = hit_record['time']

            if 'avg' in algo_name_label:
                linestyle = '--'
            else:
                linestyle = '-'

            algo_name = algo_name_label.split(' (')[0]

            ax1.plot(sizes, iters, marker=marker, label=algo_name_label,
                     linestyle=linestyle, alpha=alpha, color=ALGO_COLOR_MAPPING[algo_name])
            ax2.plot(sizes, times, marker=marker, label=algo_name_label,
                     linestyle=linestyle, alpha=alpha, color=ALGO_COLOR_MAPPING[algo_name])

        ax1.set_xlabel(r"Subset size $s$", fontsize=12)
        ax1.set_ylabel(r"Iterations to reach $\epsilon$", fontsize=12)
        ax1.set_title(rf"$\epsilon = {tol}$", fontsize=12)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()

        ax2.set_xlabel(r"Subset size $s$", fontsize=12)
        ax2.set_ylabel(r"Time to reach $\epsilon$", fontsize=12)
        ax2.set_title(rf"$\epsilon = {tol}$", fontsize=12)
        ax2.grid(True, alpha=0.3)
        # only add legend once
        ax2.legend() if tol == 1e-1 else None
        # ax2.legend()
        fig2.tight_layout()

        # if not is_on_server:
        #     plt.show()
        # else:
        filepath = f'output/LinxDoubleScaling-124-tol{tol}-iter.pdf'
        fig1.savefig(filepath, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
        filepath = f'output/LinxDoubleScaling-124-tol{tol}-time.pdf'
        fig2.savefig(filepath, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
