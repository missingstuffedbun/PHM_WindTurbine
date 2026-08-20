import os

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_true, y_pred, target_names, save_dir):
    """绘制每个目标变量的真实值与预测值对比。"""
    os.makedirs(save_dir, exist_ok=True)
    n = len(target_names)

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, name in enumerate(target_names):
        ax = axes[i]
        ax.plot(y_true[:, i], label="True", alpha=0.7)
        ax.plot(y_pred[:, i], label="Pred", alpha=0.7)
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_comparison.png"), dpi=150)
    plt.close()


def plot_scatter(y_true, y_pred, target_names, save_dir):
    """绘制每个目标变量的真实值 vs 预测值散点图。"""
    os.makedirs(save_dir, exist_ok=True)
    n = len(target_names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, name in enumerate(target_names):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=5)
        lim = [min(y_true[:, i].min(), y_pred[:, i].min()),
               max(y_true[:, i].max(), y_pred[:, i].max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(name)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter.png"), dpi=150)
    plt.close()


def plot_error_distribution(y_true, y_pred, target_names, save_dir):
    """绘制每个目标变量的预测误差分布。"""
    os.makedirs(save_dir, exist_ok=True)
    n = len(target_names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, name in enumerate(target_names):
        error = y_pred[:, i] - y_true[:, i]
        axes[i].hist(error, bins=50, edgecolor="black")
        axes[i].set_title(name)
        axes[i].set_xlabel("Error")
        axes[i].set_ylabel("Count")
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=150)
    plt.close()
