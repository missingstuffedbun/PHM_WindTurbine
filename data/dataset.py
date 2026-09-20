import os
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

# 稀疏屏蔽层级：按通道 / 按时间点 / 按时间段整段置空
SPARSE_LEVELS = ("channel", "timestep", "segment")


def load_processed_meta(data_path):
    """读取 processed 数据同目录下的 meta.yaml（由 preprocessing/prepare_data.py 生成）。

    其中记录了该版本的目标信号 / 输入信号 / 来源文件等，训练侧据此确定标签列，
    避免在 config.yaml 中重复声明。文件不存在时返回空字典。
    """
    meta_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), "meta.yaml")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class WindTurbineDataset(Dataset):
    """风机结构状态恢复数据集。

    输入：时间窗口内的可观测信号（部分被 mask），**只含输入通道，不含目标通道**
    输出：窗口最后一个时刻的目标结构响应

    目标信号（TMBNS / TMBEW / TMBTOR）只作为标签 y 出现，永远不进入输入 x，
    从结构上杜绝目标泄漏；输入维度由 `input_dim` 属性程序化给出（= len(input_cols)）。

    稀疏屏蔽层级由 scenario["level"] 指定（三选一，默认 channel）：
    - channel：按通道随机屏蔽。从候选通道中随机保留 observable_ratio 比例的通道，
      被屏蔽通道在整条时间轴上不可见 —— 对应“测点未安装 / 长期失效”。
    - timestep：按时间点屏蔽。候选通道都在，但随机保留 observable_ratio 比例的
      时间步，被屏蔽时间步上所有候选通道同时缺失 —— 对应“采样丢包 / 间歇采集”。
    - segment：按时间段整段置空。候选通道在一段连续时间内全部缺失，缺失长度
      (1 - observable_ratio) × window_size，起点随机 —— 对应“通信中断 / 停机”；
      可选 n_segments: K 将窗口均分为 K 段后随机置空其中连续的若干段。

    注意：对于随机稀疏场景（observable_ratio < 1），每个窗口的可见通道 / 时间点子集
    在数据集创建时固定，不会随 __getitem__ 调用而改变。这保证了同一窗口在
    不同 epoch 中看到相同的输入条件，使稀疏场景的实验结果可解释、可复现。
    """

    def __init__(self, data_path, window_size, stride, target_signals, scenario=None,
                 seed=42, row_range=None, missing_mode="raw_zero",
                 missing_indicator=False):
        """构建数据集。

        row_range：(start, end) 行索引区间，仅在该区间内滑窗。用于"先按时间切三段
        连续区间、再各自滑窗"的划分方式（见 build_split_datasets）。

        missing_mode：
        - "raw_zero"：缺失位置填入“原始域 0”对应的标准化值 (0 - mean) / scale，
          等价于断线测点在原始域读到 0 —— 测点未安装 / 通信中断的物理解释；
          对温度、转速、风速等测点这是一个远离均值的哨兵值，不会与真实读数混淆。
        - "norm_zero"：直接在标准化域置 0 —— 等价于填该通道**均值**，模型无法区分
          “传感器缺失”与“读数恰好等于均值”，仅用于消融对照。

        missing_indicator：为 true 时在每个输入通道后附加一条指示通道
        （1 = 该时刻该通道可见，0 = 缺失），模型输入维度 ×2（34 → 68）。
        这样“缺失”在任何数值编码下都严格可区分，是工程上更标准的做法。
        """
        self.df = pd.read_csv(data_path)
        self.data_path = data_path
        self.window_size = window_size
        self.stride = stride
        self.target_signals = target_signals
        self.scenario = scenario or {}
        self.seed = seed
        self.row_range = row_range

        self.level = str(self.scenario.get("level") or "channel").strip().lower()
        if self.level not in SPARSE_LEVELS:
            raise ValueError(
                f"未知稀疏层级 level={self.level!r}，可选值为 {SPARSE_LEVELS}。"
            )

        self.feature_cols = [c for c in self.df.columns if c != "Time"]
        self.target_indices = [self.feature_cols.index(t) for t in target_signals]

        target_set = set(target_signals)
        self.input_cols = [c for c in self.feature_cols if c not in target_set]
        self.input_indices = [self.feature_cols.index(c) for c in self.input_cols]
        self.input_dim = len(self.input_cols)
        self.output_dim = len(target_signals)

        self.missing_mode = str(missing_mode or "raw_zero").strip().lower()
        if self.missing_mode not in ("raw_zero", "norm_zero"):
            raise ValueError(
                f"未知缺失编码 missing_mode={self.missing_mode!r}，"
                "可选值为 ('raw_zero', 'norm_zero')。"
            )
        self.missing_indicator = bool(missing_indicator)
        self.missing_code = self._build_missing_code()
        # 模型实际输入维度：附加 missing-indicator 通道后 ×2
        self.model_input_dim = self.input_dim * (2 if self.missing_indicator else 1)

        self.data = self.df[self.feature_cols].values.astype(np.float32)
        self.windows = self._build_windows()

        # 预生成每个窗口的固定 mask
        self.window_masks = self._build_window_masks()

    def _build_missing_code(self):
        """每个输入通道在缺失时应填入的**标准化域**数值。

        数据已整体标准化，因此“原始域 0”对应的标准化值为 (0 - mean) / scale。
        把该值写入缺失位置，语义等价于在原始域把断线测点置 0。
        """
        if self.missing_mode == "norm_zero":
            return np.zeros(self.input_dim, dtype=np.float32)

        scaler_path = os.path.join(os.path.dirname(self.data_path), "scaler.npz")
        if not os.path.exists(scaler_path):
            warnings.warn(
                f"未找到 {scaler_path}，无法计算原始域 0 的编码值，"
                "raw_zero 退化为 norm_zero（置零 = 填均值）。",
                stacklevel=2,
            )
            return np.zeros(self.input_dim, dtype=np.float32)

        z = np.load(scaler_path, allow_pickle=True)
        columns = list(z["columns"])
        mean, scale = np.asarray(z["mean"]), np.asarray(z["scale"])

        code = np.zeros(self.input_dim, dtype=np.float32)
        ambiguous = []
        for i, col in enumerate(self.input_cols):
            if col not in columns:
                continue
            j = columns.index(col)
            code[i] = (0.0 - float(mean[j])) / float(scale[j]) if float(scale[j]) != 0 else 0.0
            if abs(code[i]) < 0.1:
                # 该通道原始均值本就接近 0（如某些加速度信号），哨兵值与“填均值”几乎重合
                ambiguous.append(col)

        if ambiguous and not self.missing_indicator:
            warnings.warn(
                f"通道 {ambiguous} 的原始均值接近 0，raw_zero 的哨兵值与 norm_zero 几乎等价；"
                "建议开启 missing_indicator 使缺失严格可区分。",
                stacklevel=2,
            )
        return code

    def _build_windows(self):
        n = len(self.data)
        start, end = self.row_range if self.row_range is not None else (0, n)
        start, end = max(0, int(start)), min(n, int(end))
        if end - start < self.window_size:
            raise ValueError(
                f"行区间 [{start}, {end}) 只有 {end - start} 行，"
                f"不足一个 window_size={self.window_size}。"
            )
        windows = []
        for i in range(start, end - self.window_size + 1, self.stride):
            windows.append((i, i + self.window_size))
        return windows

    def _build_window_masks(self):
        """为每个窗口预生成固定的观测 mask。

        返回 (channel_mask, time_mask) 元组列表（维度均为**输入通道**）：
        - channel_mask：(input_dim,) bool，True = 该通道可见（level=channel 时随机采样）；
        - time_mask：(window_size,) bool 或 None，True = 该时间步可见
          （level=timestep / segment 时随机生成，否则 None）。

        blocked_signals 中的通道任何层级下都不可用；目标信号不在输入中，无需屏蔽。
        """
        n_features = self.input_dim
        ratio = float(self.scenario.get("observable_ratio", 1.0) or 1.0)
        blocked = self.scenario.get("blocked_signals", [])
        target_set = set(self.target_signals)

        # 指定传感器失效 → 强制置零
        protected_idx = set()
        for sig in blocked:
            if sig in target_set:
                warnings.warn(
                    f"blocked_signals 中的 {sig} 是目标信号（本就不作为输入），已忽略；"
                    "该场景在数据层面等价于不屏蔽任何通道。",
                    stacklevel=2,
                )
                continue
            if sig in self.input_cols:
                protected_idx.add(self.input_cols.index(sig))

        # 候选可观测通道
        candidate_idx = np.array([i for i in range(n_features) if i not in protected_idx])

        # 基准通道 mask：候选通道可见，保护通道（目标 + 失效传感器）不可见
        base_channel_mask = np.zeros(n_features, dtype=bool)
        base_channel_mask[candidate_idx] = True

        if ratio >= 1.0 or len(candidate_idx) == 0:
            # 无随机稀疏：所有非保护通道可用
            return [(base_channel_mask, None)] * len(self.windows)

        n_keep = max(1, int(round(len(candidate_idx) * ratio)))
        n_keep_t = max(1, int(round(self.window_size * ratio)))
        n_segments = int(self.scenario.get("n_segments", 0) or 0)

        # 用独立随机状态，按窗口索引确定性采样
        rng = np.random.default_rng(self.seed)
        window_masks = []
        for _ in range(len(self.windows)):
            if self.level == "channel":
                keep_idx = rng.choice(candidate_idx, size=n_keep, replace=False)
                mask = np.zeros(n_features, dtype=bool)
                mask[keep_idx] = True
                window_masks.append((mask, None))
                continue

            time_mask = np.zeros(self.window_size, dtype=bool)
            if self.level == "timestep":
                # 随机保留时间点：被屏蔽的时间步上所有候选通道同时缺失
                keep_t = rng.choice(self.window_size, size=n_keep_t, replace=False)
                time_mask[keep_t] = True
            else:
                # segment：连续时间段整段置空
                time_mask[:] = True
                n_hide = self.window_size - n_keep_t
                if n_hide > 0:
                    if n_segments > 1 and n_hide >= n_segments:
                        # 窗口均分为 n_segments 段，随机置空其中连续的 m 段
                        m = min(n_segments, max(1, int(round((1.0 - ratio) * n_segments))))
                        start_seg = int(rng.integers(0, n_segments - m + 1))
                        lo = int(round(start_seg * self.window_size / n_segments))
                        hi = int(round((start_seg + m) * self.window_size / n_segments))
                        time_mask[lo:hi] = False
                    else:
                        start = int(rng.integers(0, self.window_size - n_hide + 1))
                        time_mask[start:start + n_hide] = False
            window_masks.append((base_channel_mask, time_mask))

        return window_masks

    def _apply_scenario(self, x, mask):
        """按预生成 mask 屏蔽输入，并可选附加 missing-indicator 通道。

        缺失位置写入 missing_code（raw_zero 模式下为原始域 0 的编码值，不是均值）。
        """
        channel_mask, time_mask = mask
        x_masked = x.copy()
        x_masked[:, ~channel_mask] = self.missing_code[~channel_mask]

        # 观测指示：与 mask 完全一致的 0/1 矩阵
        observed = np.broadcast_to(channel_mask, x.shape).astype(np.float32).copy()

        if time_mask is not None:
            x_masked[~time_mask, :] = self.missing_code
            observed[~time_mask, :] = 0.0

        if self.missing_indicator:
            x_masked = np.concatenate([x_masked, observed], axis=-1)
        return x_masked

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        x_full = self.data[start:end]
        # 只取输入通道作为模型输入；目标通道仅用于标签
        x_in = x_full[:, self.input_indices]
        x_masked = self._apply_scenario(x_in, self.window_masks[idx])
        y = x_full[-1, self.target_indices]
        return torch.from_numpy(x_masked), torch.from_numpy(y)


def split_row_bounds(n_rows, train_ratio, val_ratio, gap):
    """按时间顺序把数据切成三段**连续**区间，相邻区间之间丢弃 gap 行作为隔离带。

    gap 至少取一个 window_size，保证任意两个 split 的窗口（含其标签时刻）不会
    取自重叠的原始时间段 —— 在 stride < window_size（窗口高度重叠）的情况下，
    若先滑窗再切分，训练集末尾窗口会与验证/测试集开头窗口共享 90% 的原始数据，
    导致指标虚高。返回值可直接作为 WindTurbineDataset 的 row_range。
    """
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return {
        "train": (0, train_end),
        "val": (min(n_rows, train_end + gap), val_end),
        "test": (min(n_rows, val_end + gap), n_rows),
    }


def build_split_datasets(data_path, window_size, stride, target_signals,
                         train_ratio=0.7, val_ratio=0.15, gap=None,
                         scenario=None, seed=42, missing_mode="raw_zero",
                         missing_indicator=False):
    """先按时间切三段连续区间，再在各区间内独立滑窗（无窗口级随机打乱）。

    返回 (train_set, val_set, test_set, bounds)，bounds 为各段行区间，便于记录与检查。
    """
    gap = window_size if gap is None else max(int(gap), window_size)
    n_rows = len(pd.read_csv(data_path, usecols=[0]))
    bounds = split_row_bounds(n_rows, train_ratio, val_ratio, gap)

    splits = {}
    for name in ("train", "val", "test"):
        start, end = bounds[name]
        if end - start < window_size:
            raise ValueError(
                f"{name} 段只有 {end - start} 行，不足一个 window_size={window_size}；"
                "请调整 train_ratio / val_ratio 或减小隔离带 gap。"
            )
        splits[name] = WindTurbineDataset(
            data_path=data_path,
            window_size=window_size,
            stride=stride,
            target_signals=target_signals,
            scenario=scenario,
            seed=seed,
            row_range=(start, end),
            missing_mode=missing_mode,
            missing_indicator=missing_indicator,
        )
    return splits["train"], splits["val"], splits["test"], bounds
