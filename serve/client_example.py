"""推理服务冒烟测试客户端。

只依赖标准库（urllib），不需要 requests / httpx。
服务地址默认从 serve.yaml 的 server.host / server.port 解析（`--url` 可覆盖）。
调用 `GET /schema` 拿到通道清单与窗口长度，再从 processed.csv 取一个真实窗口
反标准化回原始工程量纲，构造请求体真实地打一遍 `POST /predict`。

用法（服务已启动的前提下）：

    python -m serve.client_example
    python -m serve.client_example --drop NAX1 NAX2 NAY1   # 模拟机舱加速度失效
    python -m serve.client_example --ratio 0.5             # 模拟随机稀疏可见 50%
    python -m serve.client_example --stream                # 演示逐点流式累积
"""

import argparse
import json
import os
import random
import urllib.error
import urllib.request
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from models.physics import load_scaler
from serve.config import ServeConfig

DEFAULT_URL = "http://127.0.0.1:8000"


def resolve_url(config_path: str) -> str:
    """服务地址：优先 serve.yaml 的 server.host / port，没有则用默认地址。"""
    path = os.path.abspath(str(config_path))
    return ServeConfig.from_yaml(path).base_url if os.path.exists(path) else DEFAULT_URL


def http_json(url, payload=None, method=None, timeout=300):
    data, headers = None, {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body}


def load_window(processed_file: str, channels: List[str], window: int,
                scaler: Dict) -> List[Dict[str, float]]:
    """取 processed.csv 末尾 window 行，反标准化成原始工程量纲的采样点。"""
    meta_path = processed_file.replace("processed.csv", "meta.yaml")
    n_rows = 0
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            n_rows = int((yaml.safe_load(f) or {}).get("rows", 0))
    except OSError:
        pass
    if n_rows <= 0:
        n_rows = len(pd.read_csv(processed_file, usecols=[0]))
    skip = max(0, n_rows - window)
    df = pd.read_csv(processed_file, skiprows=range(1, skip + 1), nrows=window)

    samples = []
    for _, row in df.iterrows():
        sample = {}
        for ch in channels:
            pair = scaler.get(ch)
            if pair is None:
                continue
            mean, scale = pair
            sample[ch] = float(row[ch]) * scale + mean
        samples.append(sample)
    return samples


def mask(samples: List[Dict[str, float]], channels: List[str], ratio: float,
         drop: List[str], seed: int) -> List[Dict[str, float]]:
    """按 --ratio / --drop 抽掉通道，模拟稀疏监测。"""
    forced = set(drop or ())
    if ratio >= 1.0 and not forced:
        return samples
    rng = random.Random(seed)
    out = []
    for sample in samples:
        keep = set(channels) - forced
        if ratio < 1.0:
            n_keep = max(1, int(round(ratio * len(channels))))
            keep = set(rng.sample(sorted(keep), min(n_keep, len(keep))))
        out.append({k: v for k, v in sample.items() if k in keep})
    return out


def main():
    p = argparse.ArgumentParser(description="推理服务冒烟测试客户端")
    p.add_argument("--config", default="serve.yaml",
                   help="服务配置文件，用于解析服务地址（server.host / server.port）")
    p.add_argument("--url", default=None,
                   help=f"服务地址，默认取 serve.yaml，其次 {DEFAULT_URL}")
    p.add_argument("--processed-file", default=None,
                   help="默认取 /model 返回的 processed_file")
    p.add_argument("--ratio", type=float, default=1.0,
                   help="模拟随机稀疏：每个采样点保留的通道比例")
    p.add_argument("--drop", nargs="*", default=(), help="强制失效的通道")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream", action="store_true", help="逐点推送，演示滚动缓存")
    p.add_argument("--show-samples", type=int, default=0, help="打印前 N 个采样点")
    args = p.parse_args()

    url = (args.url or resolve_url(args.config)).rstrip("/")
    status, schema = http_json(f"{url}/schema")
    if status != 200:
        raise SystemExit(f"GET /schema 失败（{status}）：{schema}\n"
                         "服务没起来？用 python -m serve.server 启动。")
    channels = schema["input"]["input_channels"]
    window = schema["input"]["window_size"]
    print(f"服务在线：{url}")
    print(f"  模型输入：{len(channels)} 个通道 × 窗口 {window} "
          f"-> model_input_dim={schema['input']['model_input_dim']}")
    print(f"  缺失编码：{schema['input']['missing_mode']} / "
          f"missing_indicator={schema['input']['missing_indicator']} / "
          f"blocked={schema['input']['blocked_signals'] or '无'}")

    status, model = http_json(f"{url}/model")
    if status != 200:
        raise SystemExit(f"GET /model 失败（{status}）：{model}")
    print(f"  已加载：{model['run_name']}（{model['backbone']}"
          f"{' + PINN' if model['use_pinn'] else ''}，场景 {model['scenario']}）")

    processed_file = args.processed_file or model["data"]["processed_file"]
    scaler = load_scaler(processed_file)
    samples = mask(load_window(processed_file, channels, window, scaler),
                   channels, args.ratio, list(args.drop), args.seed)

    if args.show_samples:
        print(f"  采样点样例（前 {args.show_samples} 个）：")
        for s in samples[:args.show_samples]:
            print("   ", json.dumps(s, ensure_ascii=False))

    if args.stream:
        print(f"\n流式推送 {len(samples)} 个采样点（session_id=demo）...")
        last = None
        for i, sample in enumerate(samples, start=1):
            status, out = http_json(
                f"{url}/predict",
                {"samples": [sample], "session_id": "demo"})
            if status != 200:
                raise SystemExit(f"POST /predict 失败（{status}）：{out}")
            if not out.get("ready"):
                print(f"  第 {i:>3} 点 -> 预热中（缓存 {out['buffer_length']}"
                      f"/{out['required']}）")
                continue
            last = out
            if i == len(samples):
                print(f"  第 {i:>3} 点 -> 出结果")
        out = last
    else:
        print(f"\nPOST /predict（{len(samples)} 点一个完整窗口）...")
        status, out = http_json(f"{url}/predict", {"samples": samples})
        if status != 200:
            raise SystemExit(f"POST /predict 失败（{status}）：{out}")

    print("\n响应：")
    print(json.dumps({k: out[k] for k in ("ready", "status", "physical", "standardized",
                                          "diagnostics", "warnings") if k in out},
                     ensure_ascii=False, indent=2))
    if out.get("warnings"):
        print("\n注意：服务返回了告警，见上。")


if __name__ == "__main__":
    main()
