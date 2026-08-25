import argparse
import csv
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


def load_scenarios(config, config_path):
    """加载场景策略。

    优先从 config 文件同级目录下的 config/scenarios.yaml 加载，
    回退到 config.yaml 中内嵌的 scenarios 字典（兼容旧格式）。
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    scenarios_file = os.path.join(base_dir, "config", "scenarios.yaml")

    if os.path.exists(scenarios_file):
        with open(scenarios_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    scenarios = config.get("scenarios", {})
    if isinstance(scenarios, dict):
        return scenarios
    raise ValueError(
        "config.yaml 中的 scenarios 应为场景名称列表，"
        "场景策略请定义在 config/scenarios.yaml 中。"
    )


def to_list(value):
    """将配置中的单个值或列表统一为列表。"""
    if isinstance(value, list):
        return value
    return [value]


def create_experiment_root(base_dir, experiment_name=None):
    """创建一次实验运行的总输出目录。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"{timestamp}"
    if experiment_name:
        folder_name += f"_{experiment_name}"
    root_dir = os.path.join(base_dir, folder_name)
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def create_run_dir(root_dir, scenario_name, backbone, use_pinn):
    """创建单次（scenario, backbone, use_pinn）实验的子目录。"""
    folder_name = f"{scenario_name}_{backbone}"
    if use_pinn:
        folder_name += "_pinn"
    out_dir = os.path.join(root_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def effective_physics_weight(data_loss, phys_loss, config):
    """自适应物理约束权重。

    物理项只在“数据不足 / 数据损失较大”时才应主导，数据已拟合良好时
    应自动退场，避免干扰已经收敛的 baseline。

    公式： w_eff = physics_weight * data_loss / (phys_loss + eps)
    含义：让物理损失项在数据损失尺度上与之可比；数据损失越小，w_eff 越小。
    上限钳制避免 phys_loss≈0 时权重爆炸。
    """
    physics_weight = float(config["loss"]["physics_weight"])
    if not config["loss"].get("adaptive_physics", True):
        return torch.tensor(physics_weight)
    eps = 1e-8
    ratio = data_loss.detach() / (phys_loss.detach() + eps)
    w_eff = (physics_weight * ratio).clamp(max=1.0)
    return w_eff


def train_epoch(model, loader, optimizer, config, is_pinn=False, device=None):
    model.train()
    total_loss = 0.0
    data_weight = config["loss"]["data_weight"]

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if is_pinn:
            y_pred = model(x)
            data_loss, phys_loss = model.compute_loss(x, y_pred, y)
            w_eff = effective_physics_weight(data_loss, phys_loss, config)
            loss = data_weight * data_loss + w_eff * phys_loss
        else:
            y_pred = model(x)
            loss = torch.nn.MSELoss()(y_pred, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, config, target_names, is_pinn=False, device=None):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    data_weight = config["loss"]["data_weight"]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_pinn:
                y_pred = model(x)
                data_loss, phys_loss = model.compute_loss(x, y_pred, y)
                w_eff = effective_physics_weight(data_loss, phys_loss, config)
                loss = data_weight * data_loss + w_eff * phys_loss
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


def run_single_experiment(config, scenarios_dict, scenario_name, backbone, use_pinn,
                          device, root_dir, config_path):
    """运行单个（scenario, backbone, use_pinn）组合的实验。"""
    set_seed(config["seed"])

    out_dir = create_run_dir(root_dir, scenario_name, backbone, use_pinn)
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name} | Backbone: {backbone} | PINN: {use_pinn}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}")

    # 加载场景
    scenario = scenarios_dict.get(scenario_name, {})
    print(f"Scenario description: {scenario.get('description', '')}")

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

    # 构建模型
    run_config = config.copy()
    run_config["model"] = config["model"].copy()
    run_config["model"]["backbone"] = backbone
    run_config["model"]["use_pinn"] = use_pinn

    if use_pinn:
        model = PINNWrapper(run_config).to(device)
        model.feature_names = dataset.feature_cols
        print(f"Model: {backbone} + PINN")
    else:
        model = build_model(run_config).to(device)
        print(f"Model: {backbone} (baseline)")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, config, use_pinn, device)
        val_metrics, _, _ = evaluate(model, val_loader, config, target_signals, use_pinn, device)
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
    test_metrics, y_true, y_pred = evaluate(model, test_loader, config, target_signals, use_pinn, device)

    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # 保存结果
    np.savez(
        os.path.join(out_dir, "results.npz"),
        y_true=y_true,
        y_pred=y_pred,
    )

    serializable_metrics = {
        k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v)
        for k, v in test_metrics.items()
    }

    with open(os.path.join(out_dir, "metrics.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(serializable_metrics, f, allow_unicode=True, sort_keys=False)

    # 可视化
    plot_predictions(y_true, y_pred, target_signals, out_dir)
    plot_scatter(y_true, y_pred, target_signals, out_dir)
    plot_error_distribution(y_true, y_pred, target_signals, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")

    return out_dir, serializable_metrics


def write_csv_summary(summary_path, records, fieldnames):
    """将实验记录写入 CSV 汇总文件。"""
    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


def main(config_path, scenario_names, experiment_name=None):
    config = load_config(config_path)
    scenarios_dict = load_scenarios(config, config_path)
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 展开模型配置组合
    backbones = to_list(config["model"]["backbone"])
    use_pinns = to_list(config["model"]["use_pinn"])

    # 创建一次实验的总输出目录
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    root_dir = create_experiment_root(config["paths"]["results_dir"], experiment_name)
    print(f"\nExperiment root directory: {root_dir}")

    # 保存本次实验的配置文件
    shutil.copy(config_path, os.path.join(root_dir, "config.yaml"))
    scenarios_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "config", "scenarios.yaml")
    if os.path.exists(scenarios_path):
        shutil.copy(scenarios_path, os.path.join(root_dir, "scenarios.yaml"))

    summary_path = os.path.join(root_dir, "metrics_summary.csv")

    records = []
    fieldnames = [
        "timestamp", "scenario", "backbone", "use_pinn",
        "overall_rmse", "overall_mae", "overall_mape", "overall_r2",
        "TMBNS_rmse", "TMBEW_rmse", "TMBTOR_rmse",
        "output_dir",
    ]

    for scenario_name in scenario_names:
        if scenario_name not in scenarios_dict:
            raise ValueError(
                f"未知场景: {scenario_name}。请在 config/scenarios.yaml 中定义。"
            )
        for backbone in backbones:
            for use_pinn in use_pinns:
                out_dir, metrics = run_single_experiment(
                    config, scenarios_dict, scenario_name, backbone, use_pinn,
                    device, root_dir, config_path
                )
                record = {
                    "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                    "scenario": scenario_name,
                    "backbone": backbone,
                    "use_pinn": use_pinn,
                    "overall_rmse": metrics.get("overall_rmse", ""),
                    "overall_mae": metrics.get("overall_mae", ""),
                    "overall_mape": metrics.get("overall_mape", ""),
                    "overall_r2": metrics.get("overall_r2", ""),
                    "TMBNS_rmse": metrics.get("TMBNS_rmse", ""),
                    "TMBEW_rmse": metrics.get("TMBEW_rmse", ""),
                    "TMBTOR_rmse": metrics.get("TMBTOR_rmse", ""),
                    "output_dir": out_dir,
                }
                records.append(record)

    write_csv_summary(summary_path, records, fieldnames)
    print(f"\n{'='*60}")
    print(f"All experiments completed. Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=None,
        help="一个或多个场景名称，例如：--scenario s0_full s1_medium",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="实验名称，用于生成总输出目录，例如：--name exp_v1",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    scenarios_dict = load_scenarios(config, args.config)

    if args.scenario is None:
        # 默认运行 config.yaml 中列出的所有场景
        scenario_names = config.get("scenarios", [])
        if isinstance(scenario_names, dict):
            scenario_names = list(scenario_names.keys())
    else:
        scenario_names = args.scenario

    if not scenario_names:
        raise ValueError("未指定任何场景。请在 config.yaml 或命令行中指定。")

    main(args.config, scenario_names, experiment_name=args.name)
