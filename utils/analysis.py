"""实验结果汇总与物理约束增益边界分析。

输入：一次或多次实验的输出根目录（由 `main.py` 生成，内含若干
`{scenario}_{backbone}[_pinn]/` 子目录、`metrics_summary.csv`、`config.yaml`）。

输出：结构化的分析字典（供 `analyze_results.py` 渲染 HTML 报表），核心是
**物理约束增益的边界分析**——PINN 相对同 backbone baseline 的精度改善随监测
稀疏度的变化规律、增益转正的临界稀疏度，以及发生负迁移的条件与归因。
"""

import csv
import os
import re
import statistics
from datetime import datetime

import numpy as np
import yaml

from utils.metrics import compute_metrics

DEFAULT_TARGETS = ["TMBNS", "TMBEW", "TMBTOR"]
DEFAULT_TOL = 2.0  # 判定死区：|增益| 小于该百分比视为“等效”

# 物理约束 → 驱动通道（与 models/physics.py 中的约束一一对应）
CONSTRAINT_DRIVERS = {
    "C1 机舱惯性 |M|↔加速度": ("NAX1", "NAX2", "NAY1", "NAY2", "NAZ1", "NAZ2"),
    "C3 气动扭矩 TMBTOR↔ω²": ("RST2", "TurbSpeed2", "XTurbSpeed1"),
    "C4 推力代理 |M|↔ω": ("RST2", "TurbSpeed2", "XTurbSpeed1"),
}


def load_yaml(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_run_dirs(root):
    """找出所有含 metrics.yaml 或 results.npz 的 run 目录。"""
    run_dirs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "metrics.yaml" in filenames or "results.npz" in filenames:
            run_dirs.append(dirpath)
    return sorted(run_dirs)


SEED_SUFFIX = re.compile(r"_seed\d+$")


def parse_seed(name):
    """从 run 目录名解析 seed（多种子重复实验目录带 `_seed{seed}` 后缀）。"""
    match = SEED_SUFFIX.search(name)
    return int(match.group(0)[len("_seed"):]) if match else None


def parse_run_name(name, scenario_names=()):
    """从 run 目录名解析 (scenario, backbone, use_pinn)，忽略 seed 后缀。"""
    core = SEED_SUFFIX.sub("", name)
    use_pinn = core.endswith("_pinn")
    core = core[:-5] if use_pinn else core

    for s in sorted(scenario_names, key=len, reverse=True):
        if core == s or core.startswith(s + "_"):
            return s, (core[len(s) + 1:] or "unknown"), use_pinn

    if "_" in core:
        scenario, backbone = core.rsplit("_", 1)
    else:
        scenario, backbone = core, "unknown"
    return scenario, backbone, use_pinn


def load_metrics(run_dir, target_names=None):
    """读取 run 指标：优先 metrics.yaml，回退到 results.npz 现算。"""
    metrics = load_yaml(os.path.join(run_dir, "metrics.yaml"))
    if metrics:
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    npz_path = os.path.join(run_dir, "results.npz")
    if os.path.exists(npz_path) and target_names:
        data = np.load(npz_path)
        return {k: float(v) for k, v in
                compute_metrics(data["y_true"], data["y_pred"], target_names).items()}
    return {}


def load_run_meta(root):
    """从 metrics_summary.csv 建立 run 名 → (scenario, backbone, use_pinn) 映射。"""
    path = os.path.join(root, "metrics_summary.csv")
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out_dir = (row.get("output_dir") or "").replace("\\", "/").rstrip("/")
            key = os.path.basename(out_dir)
            if key:
                mapping[key] = {
                    "scenario": row.get("scenario", ""),
                    "backbone": row.get("backbone", ""),
                    "use_pinn": str(row.get("use_pinn", "")).strip().lower() in ("true", "1", "yes"),
                }
    return mapping


def load_scenarios(roots):
    """加载场景定义：优先实验根目录下的 scenarios.yaml，回退到 config/scenarios.yaml。"""
    candidates = []
    for root in roots:
        candidates.append(os.path.join(root, "scenarios.yaml"))
        candidates.append(os.path.join(root, "config", "scenarios.yaml"))
    candidates.append(os.path.join("config", "scenarios.yaml"))

    for path in candidates:
        cfg = load_yaml(path)
        if isinstance(cfg, dict) and cfg:
            return cfg
    return {}


def driver_availability(scenario_cfg, drivers):
    """物理约束驱动通道的可用比例（0–1）。

    - blocked_signals 中的通道 → 0（确定性失效）；
    - 随机稀疏（observable_ratio < 1）→ 有效观测占比即 ratio，三种 level 的含义不同：
      channel 为每个通道被保留的概率；timestep 为每个时间步被保留的概率；
      segment 为连续缺失区间之外的剩余时间占比（连续缺失对时序相关性估计的
      破坏强于同比例的随机时间点缺失，实际危害高于该数值）；
    - 否则 → 1。
    """
    ratio = float(scenario_cfg.get("observable_ratio", 1.0) or 1.0)
    blocked = set(scenario_cfg.get("blocked_signals") or [])
    if not drivers:
        return 0.0
    avail = [0.0 if ch in blocked else (1.0 if ratio >= 1.0 else ratio) for ch in drivers]
    return sum(avail) / len(avail)


def constraint_activity(scenario_cfg):
    """返回每个物理约束在该场景下的驱动可用性。"""
    return {name: driver_availability(scenario_cfg, chans)
            for name, chans in CONSTRAINT_DRIVERS.items()}


def collect_runs(roots):
    """收集所有 run 记录，并补全场景元数据。"""
    scenarios = load_scenarios(roots)
    runs = []

    for root in roots:
        root_cfg = load_yaml(os.path.join(root, "config.yaml")) or {}
        # 目标信号：新格式在 data.target_signals（main.py 已解析写入），兼容旧的 preprocessing
        targets = ((root_cfg.get("data") or {}).get("target_signals")
                   or (root_cfg.get("preprocessing") or {}).get("target_signals")
                   or DEFAULT_TARGETS)
        meta_map = load_run_meta(root)

        for run_dir in discover_run_dirs(root):
            run_name = os.path.basename(run_dir)
            meta = meta_map.get(run_name)
            if meta:
                scenario, backbone, use_pinn = meta["scenario"], meta["backbone"], meta["use_pinn"]
            else:
                scenario, backbone, use_pinn = parse_run_name(run_name, scenarios.keys())

            metrics = load_metrics(run_dir, targets)
            if not metrics:
                continue

            runs.append({
                "run_name": run_name,
                "run_dir": run_dir,
                "root": root,
                "scenario": scenario,
                "backbone": backbone,
                "use_pinn": bool(use_pinn),
                "seed": parse_seed(run_name),
                "metrics": metrics,
                "images": [n for n in ("prediction_comparison.png", "scatter.png",
                                       "error_distribution.png")
                           if os.path.exists(os.path.join(run_dir, n))],
            })

    return runs, scenarios


def _agg(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def group_runs(runs):
    """按 (scenario, backbone, use_pinn) 聚合多次重复实验的指标。"""
    groups = {}
    for r in runs:
        key = (r["scenario"], r["backbone"], r["use_pinn"])
        groups.setdefault(key, []).append(r)

    grouped = {}
    for key, items in groups.items():
        keys = sorted({k for it in items for k in it["metrics"]})
        grouped[key] = {
            "n": len(items),
            "run_names": [it["run_name"] for it in items],
            "seeds": [it.get("seed") for it in items],
            "metrics": {k: _agg([it["metrics"][k] for it in items if k in it["metrics"]])
                        for k in keys},
        }
    return grouped


def pct_gain(base, pinn):
    """相对改善百分比（正 = PINN 更好）。"""
    if base in (None, 0) or base != base:
        return float("nan")
    return (base - pinn) / abs(base) * 100.0


def target_names_of(metrics):
    return [k[:-5] for k in metrics if k.endswith("_rmse") and k != "overall_rmse"]


def build_gain_rows(grouped, scenarios):
    """配对 baseline / PINN，计算增益。"""
    rows = []
    for (scenario, backbone, use_pinn), agg in grouped.items():
        if use_pinn:
            continue
        pinn_agg = grouped.get((scenario, backbone, True))
        if pinn_agg is None:
            continue

        base = {k: v["mean"] for k, v in agg["metrics"].items()}
        pinn = {k: v["mean"] for k, v in pinn_agg["metrics"].items()}
        if "overall_rmse" not in base or "overall_rmse" not in pinn:
            continue

        scenario_cfg = scenarios.get(scenario, {}) or {}
        gain = pct_gain(base["overall_rmse"], pinn["overall_rmse"])
        targets = {}
        for name in target_names_of(base):
            if f"{name}_rmse" in pinn:
                targets[name] = {
                    "base": base[f"{name}_rmse"],
                    "pinn": pinn[f"{name}_rmse"],
                    "gain_pct": pct_gain(base[f"{name}_rmse"], pinn[f"{name}_rmse"]),
                }

        rows.append({
            "scenario": scenario,
            "backbone": backbone,
            "ratio": float(scenario_cfg.get("observable_ratio", 1.0) or 1.0),
            "level": str(scenario_cfg.get("level") or "channel"),
            "blocked": scenario_cfg.get("blocked_signals") or [],
            "description": scenario_cfg.get("description", ""),
            "base_rmse": base["overall_rmse"],
            "pinn_rmse": pinn["overall_rmse"],
            "gain_rmse_pct": gain,
            "base_r2": base.get("overall_r2", float("nan")),
            "pinn_r2": pinn.get("overall_r2", float("nan")),
            "gain_r2": (pinn.get("overall_r2", float("nan"))
                        - base.get("overall_r2", float("nan"))),
            "base_mae": base.get("overall_mae", float("nan")),
            "pinn_mae": pinn.get("overall_mae", float("nan")),
            "n_base": agg["n"],
            "n_pinn": pinn_agg["n"],
            "targets": targets,
            "activity": constraint_activity(scenario_cfg),
        })
    return rows


def summarize_by(gain_rows, key, peer="backbone"):
    """按指定维度（scenario / backbone）汇总增益。

    peer：用于标注“最优 / 最差”的对照维度（按场景汇总时取 backbone，反之取 scenario）。
    """
    buckets = {}
    for row in gain_rows:
        buckets.setdefault(row[key], []).append(row)

    out = []
    for key_val, rows in buckets.items():
        gains = [r["gain_rmse_pct"] for r in rows]
        out.append({
            key: key_val,
            "ratio": rows[0]["ratio"],
            "mean_gain": statistics.fmean(gains),
            "min_gain": min(gains),
            "max_gain": max(gains),
            "n_pos": sum(1 for g in gains if g > 0),
            "n": len(gains),
            "best": max(rows, key=lambda r: r["gain_rmse_pct"])[peer],
            "worst": min(rows, key=lambda r: r["gain_rmse_pct"])[peer],
        })
    return sorted(out, key=lambda d: d["ratio"], reverse=True)


def _crossings(points):
    """在 (ratio, gain) 折线上找增益符号翻转点（线性插值）。

    points 按 ratio 降序（由完整到稀疏）；返回由负转正 / 由正转负的临界稀疏度。
    """
    pts = sorted(points, key=lambda p: p[0], reverse=True)
    out = []
    for (r1, g1), (r2, g2) in zip(pts, pts[1:]):
        if g1 * g2 >= 0:  # 同号，无翻转
            continue
        r_star = r1 + (0.0 - g1) * (r2 - r1) / (g2 - g1)
        out.append({
            "ratio_from": r1, "gain_from": g1,
            "ratio_to": r2, "gain_to": g2,
            "ratio_star": r_star,
            "direction": "sparse_gain" if g2 > g1 else "dense_gain",
        })
    return out


def boundary_analysis(gain_rows, tol=DEFAULT_TOL):
    """稀疏度–增益边界分析：全局 + 每个 backbone。"""
    result = {"tol": tol, "global": None, "per_backbone": [], "per_level": [],
              "series": {}, "note": ""}

    if not gain_rows:
        result["note"] = "没有可配对的 baseline / PINN 实验，无法进行边界分析。"
        return result

    levels = sorted({str(r.get("level") or "channel") for r in gain_rows})
    if len(levels) > 1:
        result["note"] = (
            result["note"]
            + " 注意：本次对比混合了多种稀疏层级（" + " / ".join(levels)
            + "），同一 observable_ratio 在不同 level 下的信息损失并不可比，"
            "全局曲线仅供参考，临界稀疏度应按 level 分组解读。"
        ).strip()

    ratios = sorted({r["ratio"] for r in gain_rows})
    if len(ratios) < 2:
        result["note"] = (
            result["note"]
            + f" 仅观测到单一稀疏度 ratio={ratios[0]}，缺少稀疏度梯度，"
            "无法给出临界稀疏度；请补充不同 observable_ratio 的场景后重跑分析。"
        ).strip()

    def series_for(rows):
        pts = []
        for ratio in sorted({r["ratio"] for r in rows}, reverse=True):
            gains = [r["gain_rmse_pct"] for r in rows if r["ratio"] == ratio]
            pts.append((ratio, statistics.fmean(gains)))
        return pts

    global_pts = series_for(gain_rows)
    result["series"]["__global__"] = global_pts
    result["global"] = {
        "points": global_pts,
        "crossings": _crossings(global_pts),
        "mean_gain": statistics.fmean([g for _r, g in global_pts]),
    }

    backbones = sorted({r["backbone"] for r in gain_rows})
    for bb in backbones:
        rows = [r for r in gain_rows if r["backbone"] == bb]
        pts = series_for(rows)
        result["series"][bb] = pts
        result["per_backbone"].append({
            "backbone": bb,
            "points": pts,
            "crossings": _crossings(pts),
            "mean_gain": statistics.fmean([g for _r, g in pts]),
        })

    # 混合多种稀疏层级时，额外给出按 level 分组的边界
    if len(levels) > 1:
        for lv in levels:
            rows = [r for r in gain_rows if str(r.get("level") or "channel") == lv]
            pts = series_for(rows)
            result["series"][f"level={lv}"] = pts
            result["per_level"].append({
                "level": lv,
                "points": pts,
                "crossings": _crossings(pts),
                "mean_gain": statistics.fmean([g for _r, g in pts]),
            })

    return result


def negative_transfer(gain_rows, tol=DEFAULT_TOL):
    """识别负迁移 / 等效样本，并给出基于证据的归因。"""
    cases = []
    for row in sorted(gain_rows, key=lambda r: r["gain_rmse_pct"]):
        if row["gain_rmse_pct"] > -tol:
            continue

        reasons = []
        ratio = row["ratio"]
        activity = row["activity"]
        min_act = min(activity.values()) if activity else 1.0
        avg_act = statistics.fmean(activity.values()) if activity else 1.0

        if ratio >= 1.0 and not row["blocked"]:
            reasons.append(
                f"监测完整（observable_ratio={ratio:g}，无传感器失效）：数据项已能提供充分监督，"
                "物理约束成为冗余先验，属于典型的‘数据充足时物理项过约束’。"
            )
        if row["base_r2"] == row["base_r2"] and row["base_r2"] > 0.95:
            reasons.append(
                f"baseline 已高度拟合（overall R²={row['base_r2']:.4f}），可改进空间小，"
                "物理残差的微小偏差也足以拉低精度。"
            )
        if avg_act < 0.95:
            reasons.append(
                "物理约束驱动通道可用性不足（"
                + "；".join(f"{k}={v:.0%}" for k, v in activity.items())
                + "）：约束退化为对含噪驱动信号的强制线性拟合，等效于错误先验。"
            )
        if min_act < 0.5:
            worst = min(activity, key=lambda k: activity[k])
            reasons.append(
                f"约束 {worst} 的驱动通道几乎不可用（{min_act:.0%}），该项应被跳过或降权。"
            )
        if row.get("level") == "segment":
            reasons.append(
                f"segment 级连续缺失（缺失长度约 {(1 - ratio) * 100:.0f}% 窗口）："
                "驱动通道在一段连续时间内整段不可用，该区间内无法估计加速度–弯矩等"
                "时序相关性，物理残差只能由剩余片段估计，方差显著增大。"
            )
        elif row.get("level") == "timestep":
            reasons.append(
                "timestep 级随机缺失：驱动信号在时间轴上不连续，差分 / 相关性类物理残差"
                "在缺失点处被跳过或退化，约束对模型的校正作用被削弱。"
            )

        tgt_gains = {k: v["gain_pct"] for k, v in row["targets"].items()}
        if tgt_gains:
            worst_t = min(tgt_gains, key=lambda k: tgt_gains[k])
            if tgt_gains[worst_t] < -tol:
                reasons.append(
                    f"分项上 {worst_t} 恶化最明显（{tgt_gains[worst_t]:+.1f}%），"
                    "说明对应的物理约束项与该目标的真实关系不匹配。"
                )
            best_t = max(tgt_gains, key=lambda k: tgt_gains[k])
            if tgt_gains[best_t] > tol and tgt_gains[worst_t] < -tol:
                reasons.append(
                    f"同一模型内 {best_t} 改善 {tgt_gains[best_t]:+.1f}% 而 {worst_t} 恶化 "
                    f"{tgt_gains[worst_t]:+.1f}%：物理约束对不同目标的作用方向不一致，"
                    "建议按目标分量分别设置约束权重。"
                )
        reasons.append(
            "机制层面：自适应权重 w_eff = clamp(physics_weight × data_loss / phys_loss, max=1.0) "
            "在物理残差远小于数据残差时会把物理项推到上限，使其与数据项同量级甚至更强，"
            "这是过约束导致负迁移的直接原因。"
        )

        cases.append({
            "scenario": row["scenario"],
            "backbone": row["backbone"],
            "ratio": ratio,
            "gain_rmse_pct": row["gain_rmse_pct"],
            "base_rmse": row["base_rmse"],
            "pinn_rmse": row["pinn_rmse"],
            "base_r2": row["base_r2"],
            "avg_activity": avg_act,
            "verdict": "负迁移" if row["gain_rmse_pct"] < -tol else "等效",
            "reasons": reasons,
        })
    return cases


def build_conclusions(gain_rows, boundary, neg_cases, tol=DEFAULT_TOL):
    """根据数据自动生成结论文本。"""
    lines = []
    if not gain_rows:
        return ["未找到可配对的 baseline / PINN 实验，无法给出结论。"]

    gains = [r["gain_rmse_pct"] for r in gain_rows]
    n_pos = sum(1 for g in gains if g > tol)
    n_neg = sum(1 for g in gains if g < -tol)
    n_neu = len(gains) - n_pos - n_neg
    mean_gain = statistics.fmean(gains)

    lines.append(
        f"共配对 {len(gain_rows)} 组 baseline / PINN 实验：正增益 {n_pos} 组、"
        f"等效 {n_neu} 组、负迁移 {n_neg} 组；平均 RMSE 改善 {mean_gain:+.2f}%"
        f"（判定死区 ±{tol:g}%）。"
    )

    if boundary.get("note"):
        lines.append(boundary["note"])

    g = boundary.get("global") or {}
    crossings = g.get("crossings") or []
    if crossings:
        for c in crossings:
            if c["direction"] == "sparse_gain":
                lines.append(
                    f"全局临界稀疏度 r* ≈ {c['ratio_star']:.2f}"
                    f"（在 ratio={c['ratio_from']:g} 增益 {c['gain_from']:+.1f}% 与 "
                    f"ratio={c['ratio_to']:g} 增益 {c['gain_to']:+.1f}% 之间线性插值）："
                    f"当 observable_ratio < {c['ratio_star']:.2f} 时，PINN 平均优于 baseline；"
                    f"稀疏度轻于该值时增益消失甚至转负。"
                )
            else:
                lines.append(
                    f"观测到反向翻转：临界稀疏度 r* ≈ {c['ratio_star']:.2f}，"
                    f"当 observable_ratio > {c['ratio_star']:.2f} 时 PINN 才占优，"
                    "说明该组实验中物理约束在稀疏侧失效（多为驱动通道被稀疏掉）。"
                )
    elif len({r["ratio"] for r in gain_rows}) >= 2:
        pts = g.get("points", [])
        if pts:
            worst_pt = min(pts, key=lambda p: p[1])
            lines.append(
                f"在已测稀疏度区间（ratio {min(p[0] for p in pts):g}–{max(p[0] for p in pts):g}）"
                f"内未出现增益符号翻转；增益最低点为 ratio={worst_pt[0]:g}"
                f"（{worst_pt[1]:+.1f}%），即本实验范围内未观测到系统性临界点，"
                "结论不宜外推到区间之外。"
            )

    for bb in boundary.get("per_backbone", []):
        for c in bb["crossings"]:
            if c["direction"] == "sparse_gain":
                lines.append(
                    f"分 backbone：{bb['backbone']} 的临界稀疏度 r* ≈ {c['ratio_star']:.2f}"
                    f"（ratio {c['ratio_from']:g}→{c['ratio_to']:g}，"
                    f"增益 {c['gain_from']:+.1f}%→{c['gain_to']:+.1f}%）。"
                )

    for lv in boundary.get("per_level", []):
        for c in lv["crossings"]:
            if c["direction"] == "sparse_gain":
                lines.append(
                    f"分稀疏层级：level={lv['level']} 的临界稀疏度 r* ≈ {c['ratio_star']:.2f}"
                    f"（ratio {c['ratio_from']:g}→{c['ratio_to']:g}，"
                    f"增益 {c['gain_from']:+.1f}%→{c['gain_to']:+.1f}%）。"
                )

    by_sc = summarize_by(gain_rows, "scenario")
    for item in by_sc:
        lines.append(
            f"场景 {item['scenario']}（ratio={item['ratio']:g}）：平均增益 "
            f"{item['mean_gain']:+.1f}%，最优 {item['best']}（{item['max_gain']:+.1f}%），"
            f"最差 {item['worst']}（{item['min_gain']:+.1f}%）。"
        )

    if neg_cases:
        names = "、".join(f"{c['scenario']}/{c['backbone']}（{c['gain_rmse_pct']:+.1f}%）"
                         for c in neg_cases)
        lines.append(f"负迁移 / 等效样本：{names}。归因见「负迁移归因」一节。")

    lines.append(
        "使用建议：稀疏度高于临界值（信息充足）时应降低 physics_weight 或关闭物理约束；"
        "驱动通道被屏蔽的场景应跳过对应约束项；物理约束权重宜按目标分量分别设置。"
    )
    return lines


def build_report(roots, tol=DEFAULT_TOL):
    """分析入口：给定结果根目录列表，返回完整报表字典。"""
    runs, scenarios = collect_runs(roots)
    grouped = group_runs(runs)
    gain_rows = build_gain_rows(grouped, scenarios)
    boundary = boundary_analysis(gain_rows, tol)
    neg_cases = negative_transfer(gain_rows, tol)

    configs = {}
    for root in roots:
        cfg = load_yaml(os.path.join(root, "config.yaml")) or {}
        if cfg:
            configs[os.path.basename(root.rstrip("\\/"))] = {
                "loss": cfg.get("loss", {}),
                "training": {k: cfg.get("training", {}).get(k)
                             for k in ("epochs", "lr", "batch_size", "early_stopping_patience")},
                "backbones": cfg.get("model", {}).get("backbone"),
            }

    scenario_info = []
    for name in sorted({r["scenario"] for r in runs}):
        cfg = scenarios.get(name, {}) or {}
        scenario_info.append({
            "name": name,
            "description": cfg.get("description", ""),
            "ratio": float(cfg.get("observable_ratio", 1.0) or 1.0),
            "level": str(cfg.get("level") or "channel"),
            "blocked": cfg.get("blocked_signals") or [],
            "activity": constraint_activity(cfg),
            "n_runs": sum(1 for r in runs if r["scenario"] == name),
        })

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "roots": [os.path.abspath(r) for r in roots],
        "tol": tol,
        "runs": runs,
        "scenarios": scenario_info,
        "configs": configs,
        "gain_rows": gain_rows,
        "by_scenario": summarize_by(gain_rows, "scenario"),
        "by_backbone": summarize_by(gain_rows, "backbone", peer="scenario"),
        "boundary": boundary,
        "negative": neg_cases,
        "conclusions": build_conclusions(gain_rows, boundary, neg_cases, tol),
    }
