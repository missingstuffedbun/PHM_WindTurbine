import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM 基线模型。"""

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """GRU 基线模型。"""

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TCNModel(nn.Module):
    """TCN（时序卷积网络）基线模型。

    使用因果卷积 + 空洞卷积扩大感受野，适合时序建模。
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size,
                padding=padding, dilation=dilation
            )
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.conv_net = nn.ModuleList(layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, window, input_dim)
        x = self.input_proj(x)  # (batch, window, hidden_dim)
        x = x.transpose(1, 2)   # (batch, hidden_dim, window)

        for i in range(0, len(self.conv_net), 3):
            conv = self.conv_net[i]
            relu = self.conv_net[i + 1]
            dropout = self.conv_net[i + 2]
            out = conv(x)
            # 因果：去掉末尾填充，保持长度一致
            out = out[:, :, :x.size(2)]
            out = relu(out)
            out = dropout(out)
            x = x + out  # 残差连接

        x = x.transpose(1, 2)  # (batch, window, hidden_dim)
        return self.fc(x[:, -1, :])


class TransformerModel(nn.Module):
    """Transformer 基线模型。"""

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


def build_model(config):
    name = config["model"]["backbone"]
    input_dim = config["model"]["input_dim"]
    output_dim = config["model"]["output_dim"]
    if input_dim is None:
        raise ValueError(
            "model.input_dim 为 null 但未被自动推导：应由 main.py 用数据集的 "
            "input_dim（不含目标通道的输入通道数）填充。"
        )
    if output_dim is None:
        raise ValueError(
            "model.output_dim 为 null 但未被自动推导：应由 main.py 用目标信号数填充。"
        )
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]

    if name == "lstm":
        return LSTMModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    elif name == "gru":
        return GRUModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    elif name == "tcn":
        return TCNModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    elif name == "transformer":
        return TransformerModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown backbone: {name}. Supported: lstm, gru, tcn, transformer")
