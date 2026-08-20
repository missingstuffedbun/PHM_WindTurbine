import torch
import torch.nn as nn


def physics_loss(pred, inputs, feature_names, target_names):
    """基于物理关系的约束损失。

    当前约束：
    1. 塔底弯矩应大致同向，避免预测符号严重冲突；
    2. 结构响应随时间变化应平滑（相邻样本差异不宜过大）。
    """
    loss = 0.0

    # 约束1：TMBNS 与 TMBEW 符号一致性（鼓励同号）
    tmbns_idx = target_names.index("TMBNS")
    tmbew_idx = target_names.index("TMBEW")
    sign_conflict = torch.relu(-pred[:, tmbns_idx] * pred[:, tmbew_idx])
    loss += sign_conflict.mean()

    # 约束2：预测值平滑性（相邻时间步差异小）
    if pred.size(0) > 1:
        smooth = torch.mean((pred[1:] - pred[:-1]) ** 2)
        loss += smooth

    return loss
