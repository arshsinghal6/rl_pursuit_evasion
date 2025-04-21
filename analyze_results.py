import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 14,           # base font size for text
    'axes.titlesize': 16,      # title
    'axes.labelsize': 14,      # x/y labels
    'xtick.labelsize': 12,     # tick labels
    'ytick.labelsize': 12,
    'legend.fontsize': 12,     # legend text
    'figure.titlesize': 18     # figure suptitle if you use one
})

# 1) Load all results CSVs under the `results/` directory
results_dir = "results"
files = glob.glob(os.path.join(results_dir, "results_*.csv"))
if not files:
    raise FileNotFoundError(f"No result files found in '{results_dir}'")


# 2) Concatenate into one DataFrame
dfs = []
for f in files:
    strategy = os.path.basename(f).replace("results_", "").replace(".csv", "")
    df = pd.read_csv(f)
    df['strategy'] = strategy
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

# 3) Compute summary statistics
summary = all_df.groupby(['strategy', 'evader_speed']).apply(lambda g: pd.Series({
    'avg_caught': g['num_caught'].mean(),
    'pct_1':      (g['num_caught'] >= 1).mean() * 100,
    'pct_2':      (g['num_caught'] >= 2).mean() * 100,
    'pct_3':      (g['num_caught'] == 3).mean() * 100,
    'avg_chase_time':   g.loc[g['num_caught'] == 3, 'chase_time'].mean(),
    'avg_getaway_time': g.loc[g['num_caught'] < 3, 'getaway_time'].mean()
})).reset_index()

# 4) Display summary table
print("Summary statistics by strategy and evader speed:")
print(summary.to_string(index=False))

# 5) For each evader speed, bar chart with all strategies
unique_speeds = sorted(summary['evader_speed'].unique())
for speed in unique_speeds:
    df_sp = summary[summary['evader_speed'] == speed]
    pivot = df_sp.set_index('strategy')[['pct_1', 'pct_2', 'pct_3']]
    ax = pivot.plot(
        kind='bar',
        figsize=(8, 5),
        title=f'Capture Rates at Evader Speed {speed}',
        ylabel='Percentage (%)'
    )
    ax.set_xlabel('Pursuer Strategy')
    plt.xticks(rotation=45)
    plt.legend(title='Num Pursuers ≥')
    plt.tight_layout()
    plt.show()
