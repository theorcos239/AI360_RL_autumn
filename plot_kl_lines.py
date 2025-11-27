import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Конфигурация
LOG_FILE = 'training_logs.csv'
OUTPUT_FILE = 'results_lines_only.png'

# Цветовая схема (как в основном скрипте)
colors = {
    'clip': 'gray', 
    'kl_default': 'orange', 
    'kl_instant': 'blue', 
    'kl_smooth': 'green',
    'kl_batch_adaptive': 'red',
    'kl_all_adaptive': 'purple',
    'kl_smooth_adaptive': 'magenta',
    'kl_barrier': 'brown'
}

def smooth(y, box_pts=10):
    if len(y) == 0: return np.array([])
    # Используем pandas для корректного сглаживания
    return pd.Series(y).rolling(window=box_pts, min_periods=1, center=True).mean().values

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found. Run training first.")
        return

    print(f"Loading data from {LOG_FILE}...")
    logs_df = pd.read_csv(LOG_FILE)
    
    print("Processing data...")
    # Создаем копию для рисования
    plot_df = logs_df.copy()
    
    # Применяем сглаживание к каждой траектории отдельно (для каждого алгоритма и сида)
    # Это важно сделать ДО усреднения/медианы
    plot_df['reward_smooth'] = plot_df.groupby(['algorithm', 'seed'])['reward'].transform(lambda x: smooth(x.values, 10))

    # Настройка стиля
    sns.set_theme(style="whitegrid", context="talk") 
    
    # Создаем 2 графика: Медиана и Минимум
    fig, axes = plt.subplots(2, 1, figsize=(24, 25)) # Увеличим высоту в 2 раза

    print("Plotting Median...")
    sns.lineplot(
        data=plot_df, 
        x='steps', 
        y='reward_smooth', 
        hue='algorithm', 
        palette=colors, 
        estimator='median', 
        errorbar=None, 
        linewidth=3,
        ax=axes[0]
    )
    axes[0].set_title("Training Reward (Median Performance)", fontsize=24, pad=20)
    axes[0].set_xlabel("Steps", fontsize=20)
    axes[0].set_ylabel("Reward", fontsize=20)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=16)

    print("Plotting Minimum...")
    # estimator='min' в seaborn нет, нужно использовать lambda
    sns.lineplot(
        data=plot_df, 
        x='steps', 
        y='reward_smooth', 
        hue='algorithm', 
        palette=colors, 
        estimator=lambda x: np.min(x), 
        errorbar=None,
        linewidth=3,
        ax=axes[1]
    )
    axes[1].set_title("Training Reward (Worst Case / Minimum Performance)", fontsize=24, pad=20)
    axes[1].set_xlabel("Steps", fontsize=20)
    axes[1].set_ylabel("Reward", fontsize=20)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=16)

    plt.tight_layout()
    plt.savefig('results_min_max.png', dpi=300)
    print(f"Done! Plot saved to results_min_max.png")
    # plt.show()

if __name__ == "__main__":
    main()

