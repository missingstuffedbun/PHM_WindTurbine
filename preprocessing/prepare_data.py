import os
import re
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_metadata(meta_path):
    """读取 metadata，返回信号名到可靠性的映射。"""
    meta = pd.read_csv(meta_path, encoding="latin1")
    meta.columns = meta.columns.str.strip()
    mapping = dict(zip(meta["Internal Signal Name"], meta["Reliable Measurement"]))
    return mapping


def find_raw_csvs(raw_dir):
    """查找 raw 目录下所有工况 CSV 文件。"""
    files = []
    for f in os.listdir(raw_dir):
        if f.endswith(".csv") and f != "Bjorko_Sensors_Specs_Metadata.csv":
            files.append(os.path.join(raw_dir, f))
    return files


def select_signals(df, reliability_map):
    """根据 sensor 配置和 metadata 选择可靠信号。"""
    # 目标输出：塔底结构响应
    targets = ["TMBNS", "TMBEW", "TMBTOR"]

    # 按 sensor 配置选择输入信号
    inputs = [
        # Hub and Blades
        "PAB1", "PAB2", "PAB3",          # Pitch angles x 3
        "B1POS", "B2POS", "B3POS",       # Blade positions
        # Shaft
        "RST2",                           # Shaft torque
        # Nacelle
        "YP",                             # Yaw position
        "NAX1", "NAX2", "NAY1", "NAY2", "NAZ1", "NAZ2",  # Accelerometers x,y,z
        # Rotor / Turbine speed
        "XTurbSpeed1", "TurbSpeed2",
        # Generator
        "GTEMP1", "GTEMP4",
        # Meteorological mast / Nacelle environment
        "WS30", "WD30", "WSN", "WDN",
        "AIRTN", "AIRHNA",
        # Control / Grid
        "GenTorqSP", "DCCREF", "DCC", "DCV",
        "PwrPercent", "OptRpm", "WindEst", "MaxPwrEst",
        "Fgrid", "Pwaste",
    ]

    selected = ["Time"] + inputs + targets
    selected = [s for s in selected if s in df.columns]

    # 过滤 metadata 中标记为不可靠的信号
    reliable = [s for s in selected if reliability_map.get(s, True) is True]

    return df[reliable].copy(), reliable


def clean_data(df):
    """基础清洗：删除全空行、用前后向填充处理缺失值。"""
    df = df.dropna(how="all")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    return df


def normalize_and_save(df, out_dir, scaler=None, fit=True):
    """对数值列做标准化，保存数据与 scaler 参数。"""
    feature_cols = [c for c in df.columns if c != "Time"]

    values = df[feature_cols].values
    if fit or scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
    else:
        scaled = scaler.transform(values)

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
    return scaler


def main(raw_dir, out_dir):
    meta_path = os.path.join(raw_dir, "Bjorko_Sensors_Specs_Metadata.csv")
    reliability_map = load_metadata(meta_path)

    csv_files = find_raw_csvs(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    # 目前只处理单个文件；多个文件可在此拼接
    data_path = csv_files[0]
    print(f"Processing: {data_path}")

    df = pd.read_csv(data_path)
    df, selected_signals = select_signals(df, reliability_map)
    print(f"Selected signals: {selected_signals}")

    df = clean_data(df)
    print(f"Cleaned data shape: {df.shape}")

    scaler = normalize_and_save(df, out_dir, fit=True)
    print(f"Saved processed data to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
