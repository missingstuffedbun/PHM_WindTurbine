"""数据处理阶段：raw -> processed。

读取 `process.yaml`，对其中声明的每个版本（20 Hz / 100 Hz）分别执行
「选点 -> 按 metadata 过滤不可靠测点 -> 清洗 -> 标准化」，产出：

    <out_dir>/processed.csv   # 标准化后的数据（含 Time / 输入通道 / 目标通道）
    <out_dir>/scaler.npz      # 标准化参数（mean / scale / columns），供数据侧还原原始域
    <out_dir>/meta.yaml       # 该版本的元信息（来源文件、行数、目标信号等）

训练侧不再做任何数据处理，只在 config.yaml 的 `data.processed_file` 里指向产出的
processed.csv 即可（目标信号会从同目录的 meta.yaml 读取）。
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

DEFAULT_PROCESS = "process.yaml"


def load_process_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path, base_dir):
    """相对路径按 base_dir 解析，绝对路径原样返回。"""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_metadata(meta_path):
    """读取 metadata，返回信号名到可靠性的映射。"""
    meta = pd.read_csv(meta_path, encoding="latin1")
    meta.columns = meta.columns.str.strip()
    mapping = dict(zip(meta["Internal Signal Name"], meta["Reliable Measurement"]))
    return mapping


def select_signals(df, reliability_map, input_signals, target_signals):
    """按配置选择输入/目标信号，并过滤 metadata 中标记为不可靠的测点。"""
    selected = ["Time"] + list(input_signals) + list(target_signals)
    selected = [s for s in selected if s in df.columns]

    # 过滤 metadata 中标记为不可靠的信号
    reliable = [s for s in selected if reliability_map.get(s, True) is True]

    # 去重但保持顺序
    seen, ordered = set(), []
    for s in reliable:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    available_targets = [s for s in target_signals if s in ordered]
    missing_targets = [s for s in target_signals if s not in ordered]
    if missing_targets:
        raise ValueError(
            f"目标信号 {missing_targets} 不在可用列中，请检查 target_signals / metadata。"
        )
    return df[ordered].copy(), ordered


def clean_data(df, clean_cfg):
    """基础清洗：删除全空行、处理 inf、前后向填充缺失值。"""
    clean_cfg = clean_cfg or {}
    if clean_cfg.get("drop_all_na", True):
        df = df.dropna(how="all")
    if clean_cfg.get("remove_inf", True):
        df = df.replace([np.inf, -np.inf], np.nan)
    if clean_cfg.get("ffill", True):
        df = df.ffill()
    if clean_cfg.get("bfill", True):
        df = df.bfill()
    return df


def normalize_and_save(df, out_dir, meta):
    """对数值列做标准化，保存数据、scaler 参数与元信息。"""
    feature_cols = [c for c in df.columns if c != "Time"]

    values = df[feature_cols].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    df_out = df.copy()
    df_out[feature_cols] = scaled

    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(os.path.join(out_dir, "processed.csv"), index=False)
    np.savez(
        os.path.join(out_dir, "scaler.npz"),
        mean=scaler.mean_,
        scale=scaler.scale_,
        columns=np.array(feature_cols),
    )
    with open(os.path.join(out_dir, "meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    return scaler


def process_dataset(process_cfg, dataset_cfg, base_dir):
    """处理单个版本（一个采样率），返回其 meta 字典。"""
    name = dataset_cfg["name"]
    raw_dir = resolve_path(process_cfg["raw_dir"], base_dir)
    meta_file = process_cfg.get("metadata_file", "Bjorko_Sensors_Specs_Metadata.csv")
    meta_path = resolve_path(os.path.join(raw_dir, meta_file), base_dir)
    data_path = resolve_path(os.path.join(raw_dir, dataset_cfg["input_file"]), base_dir)
    out_dir = resolve_path(dataset_cfg["out_dir"], base_dir)

    print(f"\n{'='*60}")
    print(f"[{name}] {dataset_cfg.get('description', '')}")
    print(f"Source : {data_path}")
    print(f"Output : {out_dir}")
    print(f"{'='*60}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到原始数据文件: {data_path}")

    reliability_map = load_metadata(meta_path)

    df = pd.read_csv(data_path)
    raw_rows = len(df)
    df, selected_signals = select_signals(
        df, reliability_map,
        process_cfg.get("input_signals", []),
        process_cfg.get("target_signals", []),
    )
    print(f"Selected signals ({len(selected_signals)}): {selected_signals}")

    df = clean_data(df, process_cfg.get("clean"))
    print(f"Rows: raw={raw_rows} -> cleaned={len(df)}")

    meta = {
        "name": name,
        "description": dataset_cfg.get("description", ""),
        "source_file": os.path.relpath(data_path, base_dir),
        "rows": int(len(df)),
        "raw_rows": int(raw_rows),
        "columns": [c for c in df.columns if c != "Time"],
        "input_signals": [s for s in selected_signals
                          if s not in set(process_cfg.get("target_signals", [])) and s != "Time"],
        "target_signals": list(process_cfg.get("target_signals", [])),
    }
    normalize_and_save(df, out_dir, meta)
    print(f"Saved processed data to: {out_dir}")
    return meta


def main(process_path, only=None):
    process_path = os.path.abspath(process_path)
    base_dir = os.path.dirname(process_path)
    process_cfg = load_process_config(process_path)

    datasets = process_cfg.get("datasets", [])
    if not datasets:
        raise ValueError(f"{process_path} 中未定义任何 datasets 条目。")

    if only:
        wanted = set(only)
        datasets = [d for d in datasets if d["name"] in wanted]
        unknown = wanted - {d["name"] for d in datasets}
        if unknown:
            raise ValueError(f"process.yaml 中不存在的数据集: {sorted(unknown)}")

    metas = [process_dataset(process_cfg, d, base_dir) for d in datasets]

    print(f"\n{'='*60}")
    print("全部数据处理完成：")
    for m in metas:
        print(f"  - {m['name']}: {m['rows']} 行 -> {m['source_file']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按 process.yaml 处理原始数据")
    parser.add_argument("--process", default=DEFAULT_PROCESS,
                        help="数据处理配置文件路径（默认 process.yaml）")
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="只处理指定版本，例如：--dataset b1_cl4_20")
    args = parser.parse_args()
    main(args.process, only=args.dataset)
