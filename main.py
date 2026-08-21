import argparse
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import WindTurbineDataset, split_dataset
from models.base import build_model
from models.pinn import PINNWrapper
from utils.metrics import compute_metrics
from utils.visualize import plot_predictions, plot_scatter, plot_error_distribution


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_output_dir(base_dir, scenario_name, backbone, use_pinn):
    """创建以时间戳+场景+backbone 命名的输出文件夹。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"{timestamp}_{scenario_name}_{backbone}"
    if use_pinn:
        folder_name += "_pinn"
    out_dir = os.path.join(base_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def train_epoch(model, loader, optimizer, config, is_pinn=False):
    model.train()
    total_loss = 0.0
    data_weight = config["loss"]["data_weight"]
    physics_weight = config["loss"]["physics_weight"]

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if is_pinn:
            y_pred = model(x)
            data_loss, phys_loss = model.compute_loss(x, y_pred, y)
            loss = data_weight * data_loss + physics_weight * phys_loss
        else:
            y_pred = model(x)
            loss = torch.nn.MSELoss()(y_pred, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, config, target_names, is_pinn=False):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    data_weight = config["loss"]["data_weight"]
    physics_weight = config["loss"]["physics_weight"]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_pinn:
                y_pred = model(x)
                data_loss, phys_loss = model.compute_loss(x, y_pred, y)
                loss = data_weight * data_loss + physics_weight * phys_loss
            else:
                y_pred = model(x)
                loss = torch.nn.MSELoss()(y_pred, y)

            total_loss += loss.item()
            all_true.append(y.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    metrics = compute_metrics(all_true, all_pred, target_names)
    metrics["loss"] = total_loss / len(loader)
    return metrics, all_true, all_pred


def main(config_path, scenario_name):
    global device

    config = load_config(config_path)
    set_seed(config["seed"])

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型类型
    backbone = config["model"]["backbone"]
    is_pinn = config["model"].get("use_pinn", False)

    # 创建本次运行的输出目录
    out_dir = create_output_dir(config["paths"]["results_dir"], scenario_name, backbone, is_pinn)
    print(f"Output directory: {out_dir}")

    # 保存本次运行的配置文件
    shutil.copy(config_path, os.path.join(out_dir, "config.yaml"))

    # 加载场景
    scenario = config["scenarios"].get(scenario_name, {})
    print(f"Scenario: {scenario_name} - {scenario.get('description', '')}")

    # 构建数据集
    data_path = os.path.join(config["paths"]["processed_dir"], "processed.csv")
    target_signals = config["preprocessing"]["target_signals"]
    dataset = WindTurbineDataset(
        data_path=data_path,
        window_size=config["preprocessing"]["window_size"],
        stride=config["preprocessing"]["stride"],
        target_signals=target_signals,
        scenario=scenario,
    )

    train_set, val_set, test_set = split_dataset(
        dataset,
        config["preprocessing"]["train_ratio"],
        config["preprocessing"]["val_ratio"],
    )

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["training"]["batch_size"])

    # 构建模型：相同 backbone，可选是否加入物理约束
    if is_pinn:
        model = PINNWrapper(config).to(device)
        model.feature_names = dataset.feature_cols
        print(f"Model: {backbone} + PINN")
    else:
        model = build_model(config).to(device)
        print(f"Model: {backbone} (baseline)")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, config, is_pinn)
        val_metrics, _, _ = evaluate(model, val_loader, config, target_signals, is_pinn)
        val_loss = val_metrics["loss"]

        print(f"Epoch {epoch+1}/{config['training']['epochs']} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    # 测试
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    test_metrics, y_true, y_pred = evaluate(model, test_loader, config, target_signals, is_pinn)

    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # 保存结果
    np.savez(
        os.path.join(out_dir, "results.npz"),
        y_true=y_true,
        y_pred=y_pred,
    )

    with open(os.path.join(out_dir, "metrics.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(test_metrics, f, allow_unicode=True, sort_keys=False)

    # 可视化
    plot_predictions(y_true, y_pred, target_signals, out_dir)
    plot_scatter(y_true, y_pred, target_signals, out_dir)
    plot_error_distribution(y_true, y_pred, target_signals, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scenario", default="s0_full")
    args = parser.parse_args()
    main(args.config, args.scenario)
