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


class GNNModel(nn.Module):
    """简单时序 GNN 基线模型（将每个特征视为图节点）。"""

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.node_embed = nn.Linear(1, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * input_dim, output_dim)

    def forward(self, x):
        # x: (batch, window, features)
        # 取最后一个时间步，每个特征作为一个节点
        x_last = x[:, -1, :].unsqueeze(-1)  # (batch, features, 1)
        h = self.node_embed(x_last)  # (batch, features, hidden)

        for layer in self.gnn_layers:
            # 简单图卷积：每个节点聚合所有邻居均值
            h_neigh = h.mean(dim=1, keepdim=True).expand(-1, h.size(1), -1)
            h = torch.relu(layer(h + h_neigh))
            h = self.dropout(h)

        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class MLPModel(nn.Module):
    """MLP 基线模型（取窗口最后一个时间步）。"""

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_last = x[:, -1, :]
        return self.net(x_last)


def build_model(config):
    name = config["model"]["backbone"]
    input_dim = config["model"]["input_dim"]
    output_dim = config["model"]["output_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]

    if name == "lstm":
        return LSTMModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    elif name == "transformer":
        return TransformerModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    elif name == "gnn":
        return GNNModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    elif name == "mlp":
        return MLPModel(input_dim, output_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown backbone: {name}")
