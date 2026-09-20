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

from data.dataset import build_split_datasets, load_processed_meta
from models.base import build_model
from models.physics import load_scaler
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


def resolve_processed_file(config, config_path):
    """确定本次训练直接读取的 processed 数据文件。

    `data.processed_file` 指向数据处理阶段（process.yaml）产出的文件，
    相对路径按 config.yaml 所在目录解析；兼容旧格式 `paths.processed_dir`。
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    processed_file = (config.get("data") or {}).get("processed_file")

    if not processed_file:
        legacy_dir = (config.get("paths") or {}).get("processed_dir")
        if legacy_dir:
            processed_file = os.path.join(legacy_dir, "processed.csv")
        else:
            raise ValueError(
                "config.yaml 缺少 data.processed_file。请先运行 "
                "`python preprocessing/prepare_data.py`（配置见 process.yaml），"
                "再把产出的 processed.csv 路径填入 data.processed_file。"
            )

    if not os.path.isabs(processed_file):
        processed_file = os.path.join(base_dir, processed_file)
    return processed_file


def resolve_target_signals(config, processed_file):
    """确定目标信号：优先 config.data.target_signals，否则读 processed 目录的 meta.yaml。"""
    targets = (config.get("data") or {}).get("target_signals")
    if not targets:
        targets = load_processed_meta(processed_file).get("target_signals")
    if not targets:
        raise ValueError(
            f"无法确定目标信号：{processed_file} 同目录缺少 meta.yaml，"
            "且 config.yaml 未设置 data.target_signals。"
        )
    return list(targets)


def to_list(value):
    """将配置中的单个值或列表统一为列表。"""
    if isinstance(value, list):
        return value
    return [value]


def check_or_set_dim(model_cfg, key, actual):
    """模型维度以“数据推导值”为准：配置为 null 时自动填充，硬编码不一致则报错。

    输入维度 = 输入通道数（目标信号不进输入），输出维度 = 目标信号数。
    这样可避免把目标通道数进 input_dim 造成的维度错位 / 目标泄漏。
    """
    cfg_value = model_cfg.get(key)
    if cfg_value is not None and int(cfg_value) != int(actual):
        raise ValueError(
            f"配置 model.{key}={cfg_value} 与数据推导值 {actual} 不一致。"
            f"请将 model.{key} 设为 null 由数据自动推导，或修正为 {actual}。"
        )
    model_cfg[key] = int(actual)


def create_experiment_root(base_dir, experiment_name=None):
    """创建一次实验运行的总输出目录。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"{timestamp}"
    if experiment_name:
        folder_name += f"_{experiment_name}"
    root_dir = os.path.join(base_dir, folder_name)
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def create_run_dir(root_dir, scenario_name, backbone, use_pinn, seed=None, multi_seed=False):
    """创建单次（scenario, backbone, use_pinn[, seed]）实验的子目录。

    多种子重复实验时附加 `_seed{seed}` 后缀，避免不同 seed 的结果互相覆盖。
    """
    folder_name = f"{scenario_name}_{backbone}"
    if use_pinn:
        folder_name += "_pinn"
    if multi_seed and seed is not None:
        folder_name += f"_seed{seed}"
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
                          seed, device, root_dir, config_path, multi_seed=False):
    """运行单个（scenario, backbone, use_pinn, seed）组合的实验。"""
    set_seed(seed)

    out_dir = create_run_dir(root_dir, scenario_name, backbone, use_pinn,
                             seed=seed, multi_seed=multi_seed)
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name} | Backbone: {backbone} | "
          f"PINN: {use_pinn} | Seed: {seed}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}")

    # 加载场景
    scenario = scenarios_dict.get(scenario_name, {})
    print(f"Scenario description: {scenario.get('description', '')}")

    # 构建数据集：先按时间切三段连续区间（相邻区间之间留隔离带），再各自滑窗
    # 数据已由 process.yaml 阶段处理好，这里直接读取对应文件
    data_path = config["data"]["processed_file"]
    target_signals = config["data"]["target_signals"]
    train_set, val_set, test_set, split_bounds = build_split_datasets(
        data_path=data_path,
        window_size=config["preprocessing"]["window_size"],
        stride=config["preprocessing"]["stride"],
        target_signals=target_signals,
        train_ratio=config["preprocessing"]["train_ratio"],
        val_ratio=config["preprocessing"]["val_ratio"],
        gap=config["preprocessing"].get("split_gap"),
        scenario=scenario,
        seed=seed,
        missing_mode=config["preprocessing"].get("missing_mode", "raw_zero"),
        missing_indicator=config["preprocessing"].get("missing_indicator", False),
    )
    print(f"Split rows: {split_bounds} | windows: "
          f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print(f"Missing encoding: mode={train_set.missing_mode}, "
          f"indicator={train_set.missing_indicator}, "
          f"model_input_dim={train_set.model_input_dim}")

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["training"]["batch_size"])

    # 构建模型
    run_config = config.copy()
    run_config["model"] = config["model"].copy()
    run_config["model"]["backbone"] = backbone
    run_config["model"]["use_pinn"] = use_pinn
    # PINN 需要知道目标信号名，这里把已解析的结果注入（不污染原配置）
    run_config["preprocessing"] = {**config["preprocessing"], "target_signals": target_signals}

    # 维度由数据推导（目标信号不进输入；开启 missing_indicator 时输入维度 ×2）
    check_or_set_dim(run_config["model"], "input_dim", train_set.model_input_dim)
    check_or_set_dim(run_config["model"], "output_dim", train_set.output_dim)

    if use_pinn:
        # 物理约束需在原始域做矢量旋转（TMBNS/TMBEW -> fore-aft/side-side）与
        # 平方运算（ω²、V²），因此注入 processed 目录的标准化参数。
        scaler = load_scaler(config["data"]["processed_file"])
        model = PINNWrapper(run_config, scaler=scaler).to(device)
        model.feature_names = train_set.input_cols
        print(f"Model: {backbone} + PINN (input_dim={train_set.model_input_dim})")
    else:
        model = build_model(run_config).to(device)
        print(f"Model: {backbone} (baseline, input_dim={train_set.model_input_dim})")

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


def main(config_path, scenario_names, experiment_name=None, seeds=None):
    config = load_config(config_path)
    scenarios_dict = load_scenarios(config, config_path)
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据来源：直接读取数据处理阶段（process.yaml）产出的文件
    config.setdefault("data", {})
    config["data"]["processed_file"] = resolve_processed_file(config, config_path)
    config["data"]["target_signals"] = resolve_target_signals(
        config, config["data"]["processed_file"]
    )
    print(f"Processed data: {config['data']['processed_file']}")
    print(f"Target signals: {config['data']['target_signals']}")

    # 展开模型配置组合（与 backbone / use_pinn 一样，seed 也可以是列表）
    backbones = to_list(config["model"]["backbone"])
    use_pinns = to_list(config["model"]["use_pinn"])
    # 命令行 --seed 优先，否则取 config.yaml 中的 seed（单个值或列表）
    seeds = [int(s) for s in to_list(seeds if seeds else config["seed"])]
    multi_seed = len(seeds) > 1

    # 创建一次实验的总输出目录
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    root_dir = create_experiment_root(config["paths"]["results_dir"], experiment_name)
    print(f"\nExperiment root directory: {root_dir}")

    # 保存本次实验的配置副本（含已解析的数据路径与目标信号，便于复现与分析）
    with open(os.path.join(root_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    scenarios_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "config", "scenarios.yaml")
    if os.path.exists(scenarios_path):
        shutil.copy(scenarios_path, os.path.join(root_dir, "scenarios.yaml"))

    summary_path = os.path.join(root_dir, "metrics_summary.csv")

    fieldnames = [
        "timestamp", "scenario", "backbone", "use_pinn", "seed",
        "overall_rmse", "overall_mae", "overall_r2",
        "TMBNS_rmse", "TMBNS_r2",
        "TMBEW_rmse", "TMBEW_r2",
        "TMBTOR_rmse", "TMBTOR_r2",
        "output_dir",
    ]

    n_total = len(scenario_names) * len(backbones) * len(use_pinns) * len(seeds)
    print(f"Total runs: {n_total} = {len(scenario_names)} scenarios "
          f"× {len(backbones)} backbones × {len(use_pinns)} pinn settings "
          f"× {len(seeds)} seeds ({seeds})")

    for seed in seeds:
        for scenario_name in scenario_names:
            if scenario_name not in scenarios_dict:
                raise ValueError(
                    f"未知场景: {scenario_name}。请在 config/scenarios.yaml 中定义。"
                )
            for backbone in backbones:
                for use_pinn in use_pinns:
                    out_dir, metrics = run_single_experiment(
                        config, scenarios_dict, scenario_name, backbone, use_pinn,
                        seed, device, root_dir, config_path, multi_seed=multi_seed
                    )
                    record = {
                        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
                        "scenario": scenario_name,
                        "backbone": backbone,
                        "use_pinn": use_pinn,
                        "seed": seed,
                        "overall_rmse": metrics.get("overall_rmse", ""),
                        "overall_mae": metrics.get("overall_mae", ""),
                        "overall_r2": metrics.get("overall_r2", ""),
                        "TMBNS_rmse": metrics.get("TMBNS_rmse", ""),
                        "TMBNS_r2": metrics.get("TMBNS_r2", ""),
                        "TMBEW_rmse": metrics.get("TMBEW_rmse", ""),
                        "TMBEW_r2": metrics.get("TMBEW_r2", ""),
                        "TMBTOR_rmse": metrics.get("TMBTOR_rmse", ""),
                        "TMBTOR_r2": metrics.get("TMBTOR_r2", ""),
                        "output_dir": out_dir,
                    }
                    write_csv_summary(summary_path, [record], fieldnames)

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
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=None,
        help="随机种子（可多个），覆盖 config.yaml 中的 seed；"
             "多个 seed 表示重复实验，例如：--seed 42 2026 916",
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

    main(args.config, scenario_names, experiment_name=args.name, seeds=args.seed)
