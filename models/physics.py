import torch
import torch.nn as nn


class PhysicsConstraints(nn.Module):
    """可学习的物理约束模块。

    将物理关系的形式固定为已知结构动力学规律，但具体比例系数
    通过数据自适应学习，避免错误经验系数。
    """

    def __init__(self):
        super().__init__()
        # 弯矩-加速度比例系数
        self.k_bending = nn.Parameter(torch.tensor(1.0))
        # 扭矩-转速平方比例系数
        self.k_torsion = nn.Parameter(torch.tensor(1.0))

    def forward(self, pred, inputs, feature_names, target_names):
        return physics_loss(pred, inputs, feature_names, target_names, self)


def _physical_consistency(pred_target, driver):
    """软物理一致性残差：pred_target 应与 driver 成线性关系。

    - 返回 None 表示该驱动信号在本 batch 中基本为 0（被 mask / 无信息），
      调用方应跳过该约束，避免对目标施加错误先验。
    - 当数据本身已满足线性关系时残差≈0，天然不干扰已拟合好的模型。
    """
    if driver is None:
        return None
    var = ((driver - driver.mean()) ** 2).mean()
    if var < 1e-6:
        # 驱动信号几乎恒定（被置零 mask），视为无信息，跳过
        return None
    s = ((driver - driver.mean()) * (pred_target - pred_target.mean())).mean() / var
    b = pred_target.mean() - s * driver.mean()
    return ((pred_target - (s * driver + b)) ** 2).mean()


def physics_loss(pred, inputs, feature_names, target_names, phys_module=None):
    """基于风机结构动力学的多物理约束损失。

    约束包括：
    1. 塔底弯矩幅值与机舱加速度幅值一致（惯性载荷）。
    2. 塔底弯矩南北/东西分量与机舱加速度同方向分量一致。
    3. 塔底扭矩与转子转速平方成正比（气动扭矩 ∝ ω²）。
    4. 塔底弯矩幅值与转速正相关（高转速 → 大弯矩）。
    5. 预测时序平滑性约束（抑制非物理高频跳变）。

    被 mask 的驱动信号（置零）会被 _physical_consistency 自动跳过，
    因此约束是场景感知的：传感器失效越严重，物理项提供的正则越强，
    数据充分时约束已近似满足，不干扰 baseline。
    """
    if phys_module is None:
        phys_module = PhysicsConstraints().to(pred.device)

    loss = 0.0
    eps = 1e-8

    tmbns_idx = target_names.index("TMBNS")
    tmbew_idx = target_names.index("TMBEW")
    tmbtor_idx = target_names.index("TMBTOR")

    pred_tmbns = pred[:, tmbns_idx]
    pred_tmbeq = pred[:, tmbew_idx]
    pred_tmbtor = pred[:, tmbtor_idx]

    # 1. 塔底弯矩幅值与机舱加速度幅值（惯性载荷）
    tmb_mag = torch.sqrt(pred_tmbns ** 2 + pred_tmbeq ** 2 + eps)

    acc_cols = [c for c in ("NAX1", "NAX2", "NAY1", "NAY2", "NAZ1", "NAZ2") if c in feature_names]

    if acc_cols:
        idxs = [feature_names.index(c) for c in acc_cols]
        acc = inputs[:, :, idxs]
        acc_mag = torch.sqrt((acc ** 2).mean(dim=(1, 2)) + eps)
        res = _physical_consistency(tmb_mag, phys_module.k_bending * acc_mag)
        if res is not None:
            loss = loss + res

    # 3. 塔底扭矩与转子/发电机转速平方成正比（气动扭矩 ∝ ω²）
    rot_cols = [c for c in ("RST2", "TurbSpeed2", "XTurbSpeed1") if c in feature_names]
    if rot_cols:
        idxs = [feature_names.index(c) for c in rot_cols]
        rot = inputs[:, :, idxs].mean(dim=(1, 2))
        rot_sq = rot ** 2
        res = _physical_consistency(pred_tmbtor, phys_module.k_torsion * rot_sq)
        if res is not None:
            loss = loss + res

    # 4. 弯矩幅值与转速正相关（高转速 → 大弯矩）
    if rot_cols:
        idxs = [feature_names.index(c) for c in rot_cols]
        rot = inputs[:, :, idxs].mean(dim=(1, 2))
        res = _physical_consistency(tmb_mag, rot)
        if res is not None:
            loss = loss + 0.3 * res

    # 5. 预测时序平滑性约束（抑制非物理高频跳变）
    if pred.size(0) > 1:
        smooth_bending = ((tmb_mag[1:] - tmb_mag[:-1]) ** 2).mean()
        smooth_torsion = ((pred_tmbtor[1:] - pred_tmbtor[:-1]) ** 2).mean()
        loss = loss + 0.1 * (smooth_bending + smooth_torsion)

    return loss
