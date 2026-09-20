"""推理侧输入构造：原始工程量纲 -> 模型输入张量。

必须严格复刻训练时的编码链路，否则服务输出与离线指标不可比：

1. **标准化** z = (x − mean) / scale，参数来自 processed.csv 同目录的
   `scaler.npz`（与训练数据完全同一套 mean / scale）；
2. **缺失编码** 缺失通道填入 `missing_code`：
   - `raw_zero`（默认）= “原始域 0”对应的标准化值 (0 − mean) / scale，
     语义等价于断线测点在原始域读到 0；
   - `norm_zero` = 标准化域置 0（等价于填通道均值），仅用于复现旧行为；
3. **观测指示** `missing_indicator=true` 时在每个输入通道后追加一条 0/1 通道
   （1 = 可见），拼接顺序与 `data/dataset.py: _apply_scenario` 一致
   —— 先 34 条数值通道，再 34 条指示通道。

通道顺序取自 processed.csv 的表头（去掉 Time 与目标信号），与训练时
`WindTurbineDataset.input_cols` 完全一致，不依赖调用方传入的顺序。
"""

import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from models.physics import load_scaler

MISSING_MODES = ("raw_zero", "norm_zero")


class ServingPreprocessor:
    """把调用方给的原始采样点编码成模型可直接推理的窗口张量。"""

    def __init__(self, processed_file, target_signals=None, window_size=100,
                 missing_mode="raw_zero", missing_indicator=True,
                 blocked_signals: Sequence[str] = ()):
        self.processed_file = os.path.abspath(str(processed_file))
        self.columns = self._read_columns()
        self.target_signals = list(target_signals or self._read_targets())
        self.input_cols = [c for c in self.columns if c not in set(self.target_signals)]
        if not self.input_cols:
            raise ValueError(f"{self.processed_file} 中没有可用的输入通道。")

        self.window_size = int(window_size)
        self.missing_mode = str(missing_mode or "raw_zero").strip().lower()
        if self.missing_mode not in MISSING_MODES:
            raise ValueError(
                f"未知缺失编码 missing_mode={self.missing_mode!r}，可选值为 {MISSING_MODES}。"
            )
        self.missing_indicator = bool(missing_indicator)
        # 场景强制失效的通道：任何请求下都按缺失处理，与训练时 blocked_signals 一致
        self.blocked = set(blocked_signals or ())

        self.scaler = load_scaler(self.processed_file) or {}
        if not self.scaler:
            raise FileNotFoundError(
                f"未找到 {os.path.dirname(self.processed_file)}/scaler.npz，"
                "无法完成标准化 / 反标准化，请先运行 preprocessing/prepare_data.py。"
            )
        self.missing_code = self._build_missing_code()
        self.model_input_dim = len(self.input_cols) * (2 if self.missing_indicator else 1)

    # ------------------------------------------------------------------ 初始化

    def _read_columns(self) -> List[str]:
        """读 processed.csv 表头（不加载数据体），得到通道顺序。"""
        cols = list(pd.read_csv(self.processed_file, nrows=0).columns)
        cols = [c for c in cols if c != "Time"]
        if not cols:
            raise ValueError(f"{self.processed_file} 表头为空，无法推断通道顺序。")
        return cols

    def _read_targets(self) -> List[str]:
        meta_path = os.path.join(os.path.dirname(self.processed_file), "meta.yaml")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"未找到 {meta_path}，且未显式传入 target_signals，无法确定目标信号。"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        targets = meta.get("target_signals")
        if not targets:
            raise ValueError(f"{meta_path} 中缺少 target_signals。")
        return list(targets)

    def _build_missing_code(self) -> np.ndarray:
        """每个输入通道在缺失时应填入的标准化域数值。"""
        code = np.zeros(len(self.input_cols), dtype=np.float32)
        if self.missing_mode == "norm_zero":
            return code
        for i, col in enumerate(self.input_cols):
            pair = self.scaler.get(col)
            if pair is None or pair[1] == 0:
                continue
            mean, scale = pair
            code[i] = (0.0 - mean) / scale
        return code

    # ------------------------------------------------------------------ 编码

    def observed_mask(self, sample: Dict) -> np.ndarray:
        """判断一个采样点上各输入通道是否可见。

        不可见的三种情况：场景 blocked_signals 强制失效；请求中未提供该通道；
        提供了但值为 None / NaN / inf（视为断线）。
        """
        mask = np.zeros(len(self.input_cols), dtype=bool)
        for i, col in enumerate(self.input_cols):
            if col in self.blocked:
                continue
            v = sample.get(col)
            mask[i] = isinstance(v, (int, float)) and not isinstance(v, bool) \
                and math.isfinite(float(v))
        return mask

    def encode(self, sample: Dict):
        """把单个采样点编码成一行模型输入，返回 (数值行, 观测指示行)。"""
        z = np.empty(len(self.input_cols), dtype=np.float32)
        obs = self.observed_mask(sample)
        for i, col in enumerate(self.input_cols):
            pair = self.scaler.get(col)
            # scaler 里没有该通道（异常配置）时按缺失处理，避免 KeyError 打断服务
            if not obs[i] or pair is None:
                obs[i] = False
                z[i] = self.missing_code[i]
                continue
            mean, scale = pair
            z[i] = (float(sample[col]) - mean) / scale
        return z, obs.astype(np.float32)

    def encode_window(self, samples: Sequence[Dict]) -> np.ndarray:
        """把一组采样点编码成 (1, window_size, model_input_dim) 的批量张量。"""
        if len(samples) < self.window_size:
            raise ValueError(
                f"样本数不足：需要 window_size={self.window_size} 个连续采样点，"
                f"实际收到 {len(samples)} 个。"
            )
        rows = []
        for sample in list(samples)[-self.window_size:]:
            z, obs = self.encode(sample)
            rows.append(np.concatenate([z, obs]) if self.missing_indicator else z)
        return np.stack(rows, axis=0)[None, ...].astype(np.float32)

    # ------------------------------------------------------------------ 解码

    def inverse_targets(self, z_pred: Sequence[float]) -> Dict[str, Optional[float]]:
        """把标准化域的预测值还原到原始工程量纲。"""
        out = {}
        for i, name in enumerate(self.target_signals):
            pair = self.scaler.get(name)
            out[name] = float(z_pred[i]) * pair[1] + pair[0] if pair else None
        return out

    # ------------------------------------------------------------------ 描述

    @property
    def input_channels(self) -> List[str]:
        return list(self.input_cols)

    def describe(self) -> Dict:
        """给 /schema 用的通道与编码说明。"""
        return {
            "input_channels": self.input_channels,
            "n_input_channels": len(self.input_cols),
            "model_input_dim": self.model_input_dim,
            "target_signals": list(self.target_signals),
            "window_size": self.window_size,
            "missing_mode": self.missing_mode,
            "missing_indicator": self.missing_indicator,
            "blocked_signals": sorted(self.blocked),
            "missing_code": {c: float(v) for c, v in zip(self.input_cols, self.missing_code)},
            "units": "输入为原始工程量纲（与 data/raw 一致），服务内部完成标准化",
        }
