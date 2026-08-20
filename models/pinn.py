import torch.nn as nn

from models.base import build_model
from models.physics import physics_loss


class PINNWrapper(nn.Module):
    """物理约束包装器：将任意 backbone 模型转换为 PINN。"""

    def __init__(self, config):
        super().__init__()
        self.backbone = build_model(config)
        self.target_names = config["preprocessing"]["target_signals"]
        self.feature_names = None  # 在训练时从 dataset 传入

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self, x, y_pred, y_true):
        data_loss = nn.MSELoss()(y_pred, y_true)
        phys_loss = physics_loss(y_pred, x, self.feature_names, self.target_names)
        return data_loss, phys_loss
