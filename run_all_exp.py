import matrix_game, lasso, fairness, linx

for experiment in [linx]:  # [matrix_game, lasso, fairness, linx]:
    print(f"Running {experiment.__name__}...")
    experiment.run(experiment_type='plot', run_or_load='run')
