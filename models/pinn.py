import torch.nn as nn

from models.base import build_model
from models.physics import physics_loss, PhysicsConstraints


class PINNWrapper(nn.Module):
    """物理约束包装器：将任意 backbone 模型转换为 PINN。"""

    def __init__(self, config, scaler=None):
        super().__init__()
        self.backbone = build_model(config)
        # 目标信号：优先 processing 段落（main.py 已注入），回退到 data 段落
        self.target_names = ((config.get("preprocessing") or {}).get("target_signals")
                             or (config.get("data") or {}).get("target_signals"))
        self.feature_names = []  # 在训练时从 dataset 传入
        # scaler：物理约束需在原始域做矢量旋转与平方，由 main.py 从 processed
        # 目录的 scaler.npz 加载后注入；为 None 时弯矩约束自动跳过。
        self.phys_module = PhysicsConstraints(scaler=scaler)

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self, x, y_pred, y_true):
        data_loss = nn.MSELoss()(y_pred, y_true)
        phys_loss = physics_loss(
            y_pred, x, self.feature_names, self.target_names, self.phys_module
        )
        return data_loss, phys_loss
