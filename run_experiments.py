import argparse
import copy
import subprocess
import sys

import yaml


SCENARIOS = ["s0_full", "s1_medium", "s2_severe", "s3_tower_failure", "s4_nacelle_failure", "s5_rotor_failure"]
BACKBONES = ["lstm", "transformer", "gnn", "mlp"]


def run_single(config_path, scenario, backbone, use_pinn):
    """运行单个实验配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["backbone"] = backbone
    config["model"]["use_pinn"] = use_pinn

    # 生成临时配置文件
    exp_name = f"{scenario}_{backbone}{'_pinn' if use_pinn else ''}"
    tmp_config_path = f"tmp_config_{exp_name}.yaml"
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"{'='*60}")

    subprocess.run([
        sys.executable, "main.py",
        "--config", tmp_config_path,
        "--scenario", scenario,
    ])


def main(config_path, scenarios, backbones, compare_pinn):
    for scenario in scenarios:
        for backbone in backbones:
            # baseline
            run_single(config_path, scenario, backbone, use_pinn=False)

            # pinn 对比
            if compare_pinn:
                run_single(config_path, scenario, backbone, use_pinn=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--backbones", nargs="+", default=BACKBONES)
    parser.add_argument("--compare_pinn", action="store_true", help="同时运行 baseline 和 PINN 对比")
    args = parser.parse_args()

    main(args.config, args.scenarios, args.backbones, args.compare_pinn)
