import pandas as pd
import numpy as np
import statsmodels.api as sm
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# --- Load factor returns ---
def load_factor_returns(file_path):
    df = pd.read_excel(file_path, parse_dates=['Date'])
    df = df.iloc[:, [0, 1]]  # Keep only first two columns: Date and Return
    df.columns = ['Date', 'Return']
    df = df.sort_values('Date').reset_index(drop=True)
    return df



# --- Trend Scanning Labeling Function ---
def trend_scanning_labels(df, window_range=(1,12), threshold=0):
    labels = []
    t_stats = []
    best_windows = []

    returns = df['Return'].values
    n = len(returns)

    for i in range(n):
        best_t_stat = 0
        best_label = 0
        best_window = 0

        for w in range(window_range[0], window_range[1] + 1):
            if i + w >= n:
                break

            y = np.cumsum(returns[i:i + w])  # Cumulative return
            x = np.arange(len(y))
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            t = model.tvalues[1]

            if abs(t) > abs(best_t_stat):
                best_t_stat = t
                best_window = w
                if t > threshold:
                    best_label = 1
                elif t < -threshold:
                    best_label = -1
                else:
                    best_label = 0

        labels.append(best_label)
        t_stats.append(best_t_stat)
        best_windows.append(best_window)
    
    df['Cumulative_Return'] = df['Return'].cumsum()
    df['Label'] = labels
    df['t_stat'] = t_stats
    df['Best_window'] = best_windows
    
    return df

# --- Load factor returns and apply trend scanning function ---
momentum = load_factor_returns('factor_returns_momentum.xlsx')
value = load_factor_returns('factor_returns_value.xlsx')
quality = load_factor_returns('factor_returns_quality.xlsx')
benchmark = load_factor_returns('benchmark_returns.xlsx')

momentum_labeled = trend_scanning_labels(momentum)
value_labeled = trend_scanning_labels(value)
quality_labeled = trend_scanning_labels(quality)


# --- save label, t-stats and best window to new worksheets ---
momentum_labeled.to_csv('factor_returns_momentum_labeled.csv', index=False)
value_labeled.to_csv('factor_returns_value_labeled.csv', index=False)
quality_labeled.to_csv('factor_returns_quality_labeled.csv', index=False)

# plot trend scan results against cumulative price moves

def plot_trend_labels(df, title='Trend Labels on Cumulative Return'):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cumulative returns
    ax.plot(df['Date'], df['Cumulative_Return'], label='Cumulative Return', color='black')

    # Plot trend label markers
    ax.scatter(df['Date'][df['Label'] == 1], df['Cumulative_Return'][df['Label'] == 1],
               label='Uptrend (+1)', color='green', marker='^', s=50)

    ax.scatter(df['Date'][df['Label'] == -1], df['Cumulative_Return'][df['Label'] == -1],
               label='Downtrend (-1)', color='red', marker='v', s=50)

    ax.scatter(df['Date'][df['Label'] == 0], df['Cumulative_Return'][df['Label'] == 0],
               label='No Trend (0)', color='blue', marker='o', s=30, alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

mom_plot=plot_trend_labels(momentum_labeled,title='Momentum (1-12m) - Trend Labels on Cumulative Return')
val_plot=plot_trend_labels(value_labeled,title='Value (1-12m) - Trend Labels on Cumulative Return')
qul_plot=plot_trend_labels(quality_labeled,title='Quality (1-12m) - Trend Labels on Cumulative Return')

