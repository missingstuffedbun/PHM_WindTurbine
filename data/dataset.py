import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindTurbineDataset(Dataset):
    """风机结构状态恢复数据集。

    输入：时间窗口内的可观测信号（部分被 mask）
    输出：窗口最后一个时刻的目标结构响应
    """

    def __init__(self, data_path, window_size, stride, target_signals, scenario=None):
        self.df = pd.read_csv(data_path)
        self.window_size = window_size
        self.stride = stride
        self.target_signals = target_signals
        self.scenario = scenario or {}

        self.feature_cols = [c for c in self.df.columns if c != "Time"]
        self.target_indices = [self.feature_cols.index(t) for t in target_signals]

        self.data = self.df[self.feature_cols].values.astype(np.float32)
        self.windows = self._build_windows()

    def _build_windows(self):
        n = len(self.data)
        windows = []
        for i in range(0, n - self.window_size + 1, self.stride):
            windows.append((i, i + self.window_size))
        return windows

    def _apply_scenario(self, x):
        """根据场景 mask 输入中的部分信号。

        注意：目标信号（target_signals）是模型需要预测的输出，
        即使在传感器失效场景下也不应作为输入暴露给模型，因此
        它们始终被 mask。随机稀疏时，目标信号不计入可观测通道，
        避免任务不可学习。
        """
        x_masked = x.copy()
        n_features = x_masked.shape[1]

        # 目标信号索引集合：这些信号永远不作为输入
        protected_idx = set(self.target_indices)

        # 指定传感器失效
        blocked = self.scenario.get("blocked_signals", [])
        for sig in blocked:
            if sig in self.feature_cols:
                idx = self.feature_cols.index(sig)
                x_masked[:, idx] = 0.0
                protected_idx.add(idx)

        # 随机比例稀疏：只在非目标、非 blocked 的输入特征中选择
        ratio = self.scenario.get("observable_ratio", 1.0)
        if ratio < 1.0:
            candidate_idx = [i for i in range(n_features) if i not in protected_idx]
            n_keep = max(1, int(len(candidate_idx) * ratio))
            keep_idx = np.random.choice(candidate_idx, n_keep, replace=False)
            mask = np.zeros(n_features, dtype=bool)
            mask[keep_idx] = True
            # 未保留的候选通道置零；目标/blocked 通道保持已 mask 状态
            x_masked[:, ~mask] = 0.0

        return x_masked

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        x_full = self.data[start:end]
        x_masked = self._apply_scenario(x_full)
        y = x_full[-1, self.target_indices]
        return torch.from_numpy(x_masked), torch.from_numpy(y)


def split_dataset(dataset, train_ratio, val_ratio):
    """按时间顺序划分训练/验证/测试集。"""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_set = torch.utils.data.Subset(dataset, range(0, n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_set = torch.utils.data.Subset(dataset, range(n_train + n_val, n))

    return train_set, val_set, test_set
