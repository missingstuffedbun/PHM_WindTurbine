"""物理约束（Physics Constraints）定义。

本文件中的约束来自风机结构动力学关系，属于**物理约束**，不是数值正则：
物理约束把已知物理规律以软残差形式注入训练，而对参数范数、输出平滑性的
惩罚才是正则项（本项目由 `weight_decay` 承担）。

理论依据：塔底弯矩与推力 / 倾覆力矩的平衡式 M_base ≈ T·h、其简化假设、
量纲一致性推导与误差范围（稳态 ±20%，极端工况 30%–40%），见 `physics.md`。

────────────────────────────────────────────────────────────────────────
本次重构的依据（诊断脚本 `diagnose_ns_ew.py` / `check_frame.py`，100 Hz 版运行样本）
────────────────────────────────────────────────────────────────────────

1) TMBNS / TMBEW 是塔基弯矩在**地理固定坐标系**下的两个正交分量，必须作为
   一个二维矢量整体处理，**不能逐通道独立施加物理约束**。真实机制作用在
   机舱坐标系上：

     M_fa（fore-aft，顺风向）: 气动推力 T·h 主导
     M_ss（side-side，侧向） : 横风湍流 / 塔架摆振主导，与推力基本无关

   二者由偏航角 YP 旋转得到。旋转角扫描（δ 步长 15°）实测：

     δ≈0–15°（fore-aft）: σ=63,156  R²=0.84  corr(RST2)=−0.87
     δ≈90–105°（侧向）  : σ=18,358  R²=0.22  corr(RST2)=+0.31

   而逐通道 R² 仅 TMBNS=0.43、TMBEW=0.50 —— 两个分量都"半吊子"，因为地理轴
   与物理轴的夹角随 YP 时变（实测 YP 圆周合成矢量长度仅 0.567），统计量不稳定。
   ⇒ **推力类约束只施加在 M_fa 上；M_ss 不施加推力约束。**

2) 矢量运算必须在**原始（去标准化）域**进行。σ_TMBNS/σ_TMBEW ≈ 0.64，
   在标准化域做 √(ẑ_NS²+ẑ_EW²) 与物理域 |M| 的相关系数只有 **0.854**。
   同理，ω² / V² 必须去标准化后再平方：ẑ² ≠ (z²)^（见 `physics.md` §4）。

3) 移除了原来的"batch 内相邻差分"平滑项。`main.py` 用 `DataLoader(shuffle=True)`，
   batch 内相邻样本在**时间上并不相邻**，该约束实际惩罚的是批内输出方差
   （把预测往批均值压），而非时序跳变。它既非物理约束，实现也是错的。
   另注：`processed.csv` 实际由 9 个不相邻时间段拼接（最大断点 245 天），
   即使按时间顺序取 batch，跨段差分仍无意义。

4) 移除了 `k_bending` / `k_torsion` 两个可学习比例系数。一致性残差对每个 batch
   拟合的是**任意斜率 + 截距**，比例系数会被拟合斜率完全吸收，因此它们是不可
   辨识的自由参数——梯度下降可以通过把它们推向 0 来静默关闭约束。
   未知标定系数的处理已由"逐 batch 拟合仿射变换"承担（`physics.md` §4）。
"""

import os

import numpy as np
import torch
import torch.nn as nn

_WARNED = set()

# 判定"信号无信息"的方差阈值（去标准化后，物理量纲下的平方均值）
_VAR_EPS = 1e-12


def _warn_once(key, msg):
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[physics] 警告：{msg}")


def load_scaler(processed_file):
    """读取 processed.csv 同目录的 scaler.npz，返回 {列名: (mean, scale)}。

    物理约束需要在**原始域**做矢量旋转和平方运算，因此必须拿到标准化参数。
    找不到时返回 None（弯矩的方向性约束会被跳过并告警）。
    """
    path = os.path.join(os.path.dirname(str(processed_file)), "scaler.npz")
    if not os.path.exists(path):
        _warn_once("scaler_missing", f"未找到 {path}，无法去标准化，"
                                     "弯矩的 fore-aft / side-side 约束将被跳过。")
        return None
    z = np.load(path, allow_pickle=True)
    cols = list(z["columns"])
    mean = np.asarray(z["mean"], dtype=np.float64)
    scale = np.asarray(z["scale"], dtype=np.float64)
    return {c: (float(mean[i]), float(scale[i])) for i, c in enumerate(cols)}


class PhysicsConstraints(nn.Module):
    """物理约束模块。

    本模块不再持有可学习参数（理由见文件头第 4 条）：一致性残差对每个 batch
    拟合任意斜率 + 截距，未知的物理标定系数（ρ、A、h、C_T 等）被自动吸收。
    模块仅负责携带去标准化所需的 scaler。
    """

    def __init__(self, scaler=None):
        super().__init__()
        self.scaler = dict(scaler or {})

    def set_scaler(self, scaler):
        self.scaler = dict(scaler or {})

    def forward(self, pred, inputs, feature_names, target_names):
        return physics_loss(pred, inputs, feature_names, target_names, self)


def _physical_consistency(pred_target, driver):
    """软物理一致性残差：pred_target 应与 driver 线性相关。

    实现为对两者各自去均值后计算相关系数 ρ，残差取 **1 − ρ²**，取值 [0, 1]：

    - **量纲无关**：可与标准化域的 data_loss 用同一量级的权重相加；
    - **仿射不变**：对 pred / driver 的任意仿射变换结果不变，因此未知标定
      系数被自动吸收（`physics.md` §4）；
    - **只约束关联形状，不约束幅值**，避免把错误的幅值先验强加给模型。

    返回 None 表示该约束在本 batch 无信息（驱动被 mask / 预测退化为常数），
    调用方应跳过，避免对目标施加错误先验。
    """
    if driver is None:
        return None
    if not torch.isfinite(pred_target).all() or not torch.isfinite(driver).all():
        return None

    d = driver - driver.mean()
    d_var = (d ** 2).mean()
    if d_var < _VAR_EPS:
        # 驱动信号几乎恒定（被置零 mask），视为无信息，跳过
        return None

    p = pred_target - pred_target.mean()
    p_var = (p ** 2).mean()
    if p_var < _VAR_EPS:
        # 预测退化为常数，谈不上相关性
        return None

    rho = (d * p).mean() / torch.sqrt(d_var * p_var + 1e-12)
    return torch.clamp(1.0 - rho ** 2, min=0.0)


def _to_physical(z, scaler, name):
    """把标准化值还原到原始域。缺少 scaler 时返回 None。"""
    pair = scaler.get(name)
    if pair is None:
        return None
    mean, scale = pair
    return z * scale + mean


def _window_mean_physical(inputs, feature_names, scaler, name, square=False):
    """取某输入通道在窗口内的均值，去标准化（可选平方）后返回。

    要点：**平方运算必须在去标准化之后**做，否则 ẑ² ≠ (z²)^（`physics.md` §4）。
    通道被 mask（去标准化后仍近似恒定）时返回 None，交调用方回退到下一个代理。
    """
    if name not in feature_names:
        return None
    pair = scaler.get(name)
    if pair is None:
        return None
    mean, scale = pair
    v = inputs[:, :, feature_names.index(name)].mean(dim=1) * scale + mean
    if square:
        v = v ** 2
    if not torch.isfinite(v).all():
        return None
    if ((v - v.mean()) ** 2).mean() < _VAR_EPS:
        return None
    return v


def _yaw_angle(inputs, feature_names, scaler):
    """取预测时刻的偏航方位角 ψ（弧度），用于把地理坐标系旋转到机舱坐标系。

    取窗口最后一个时间步（与预测目标时刻对齐）。YP 缺失、缺少 scaler、
    或在本 batch 内恒定（被 mask）时返回 None —— 此时无法确定方位，
    调用方应跳过方向性约束而不是猜一个角度。
    """
    if "YP" not in feature_names:
        _warn_once("yp_missing", "输入特征中没有 YP（偏航方位角），"
                                 "无法把 TMBNS/TMBEW 旋转到机舱坐标系，跳过弯矩约束。")
        return None
    pair = scaler.get("YP")
    if pair is None:
        return None
    mean, scale = pair
    yp = inputs[:, -1, feature_names.index("YP")] * scale + mean
    if ((yp - yp.mean()) ** 2).mean() < _VAR_EPS:
        # YP 被 mask，方位不可知
        return None
    return torch.deg2rad(yp % 360.0)


def _thrust_driver(inputs, feature_names, scaler):
    """气动推力 T 的可观测代理，按可靠性依次回退。

    - RST2（轴扭矩）：Q ∝ V² ∝ T（额定风速以下），与弯矩相关性最强；
    - ω²：由 λ = ωR/V 恒定得到 T ∝ V² ∝ ω²；
    - V²（WSN / WS30 / WindEst）：式 (2) 的直接形式。

    三者都在**去标准化后**平方。若全部不可用（被 mask），返回 None。
    """
    d = _window_mean_physical(inputs, feature_names, scaler, "RST2")
    if d is not None:
        return d
    for name in ("XTurbSpeed1", "TurbSpeed2"):
        d = _window_mean_physical(inputs, feature_names, scaler, name, square=True)
        if d is not None:
            return d
    for name in ("WSN", "WS30", "WindEst"):
        d = _window_mean_physical(inputs, feature_names, scaler, name, square=True)
        if d is not None:
            return d
    return None


def _torsion_driver(inputs, feature_names, scaler):
    """气动扭矩代理 ω²（与 `physics.md` 约束 3 同源）。"""
    for name in ("XTurbSpeed1", "TurbSpeed2"):
        d = _window_mean_physical(inputs, feature_names, scaler, name, square=True)
        if d is not None:
            return d
    return None


def physics_loss(pred, inputs, feature_names, target_names, phys_module=None):
    """基于风机结构动力学的多物理约束损失。

    约束包括：
    1. **M_fa（顺风向塔基弯矩）↔ 气动推力代理**（式 (1)(2)：M_base ≈ T·h）。
       只在 fore-aft 分量上施加——实测该方向 R²=0.84、corr(RST2)=−0.87。
    2. **M_ss（侧向）不施加推力约束**：实测 R² 仅 0.22、corr(RST2)=+0.31，
       它由横风湍流与塔架摆振主导，套推力先验是错误先验。
    3. **TMBTOR ↔ ω²**（气动扭矩 ∝ ω²，与式 (3) 同源）。

    被 mask 的驱动信号会被 `_window_mean_physical` / `_physical_consistency`
    自动跳过，因此约束是场景感知的：传感器失效越严重，可用的约束越少。

    各项的物理来源与平衡式推导见 `physics.md`。
    """
    if phys_module is None:
        phys_module = PhysicsConstraints().to(pred.device)

    # 初始化为 tensor，避免所有约束被跳过时返回 Python float
    loss = torch.tensor(0.0, device=pred.device)

    scaler = getattr(phys_module, "scaler", None) or {}

    tmbns_idx = target_names.index("TMBNS")
    tmbew_idx = target_names.index("TMBEW")
    tmbtor_idx = target_names.index("TMBTOR")

    # 1. fore-aft 弯矩 ↔ 推力：需要先把 (TMBNS, TMBEW) 还原到原始域再旋转
    psi = _yaw_angle(inputs, feature_names, scaler)
    if psi is None:
        _warn_once("no_rotation", "无法确定偏航方位角，跳过 fore-aft 弯矩约束"
                                  "（不会退化为错误的逐通道约束）。")
    else:
        ns = _to_physical(pred[:, tmbns_idx], scaler, "TMBNS")
        ew = _to_physical(pred[:, tmbew_idx], scaler, "TMBEW")
        if ns is None or ew is None:
            _warn_once("no_target_scaler", "scaler 中缺少 TMBNS/TMBEW，跳过弯矩约束。")
        else:
            m_fa = ns * torch.cos(psi) + ew * torch.sin(psi)
            res = _physical_consistency(m_fa, _thrust_driver(inputs, feature_names, scaler))
            if res is not None:
                loss = loss + res

    # 2. side-side 分量：不施加推力类约束（依据见文件头第 1 条实测数据）

    # 3. 塔底扭矩与转子转速平方成正比（气动扭矩 ∝ ω²）
    res = _physical_consistency(
        pred[:, tmbtor_idx], _torsion_driver(inputs, feature_names, scaler)
    )
    if res is not None:
        loss = loss + res

    return loss
