"""从实验结果目录里定位“最优模型”。

排序规则（默认 aggregate="mean"）：

1. 只考虑含 `best_model.pt` 的 run（训练中断 / 未产出的直接排除）；
2. 按 (scenario, backbone, use_pinn) 聚合多种子重复实验，用**指标均值**排序。
   多种子取均值比挑单个幸运 seed 稳健；`aggregate="single"` 则直接按单次 run 排序；
3. 胜出组内再取**指标最好的那一次 run** 的权重（均值最好不代表其中某次最好，
   但部署必须落到具体的一份权重上）。

指标方向：`*_r2` 越大越好，其余（rmse / mae / mape / smape / loss）越小越好。

⚠️ 跨场景比较不是同口径：s0_full（完整监测）天然比 s2_severe（严重稀疏）精度高，
默认取全局最优等价于“选最好测点条件下的模型”。部署到真实稀疏场景时应显式
指定 `--scenario`，否则服务的输入语义（哪些通道缺失）与训练场景不一致。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from utils.analysis import collect_runs, group_runs

CHECKPOINT_NAME = "best_model.pt"
DEFAULT_METRIC = "overall_rmse"


def metric_direction(metric: str) -> int:
    """排序方向：1 = 越大越好，-1 = 越小越好。"""
    return 1 if str(metric).endswith("r2") else -1


def _to_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


@dataclass
class Candidate:
    """一个可部署的模型候选（胜出 run + 其所属分组的聚合信息）。"""

    scenario: str
    backbone: str
    use_pinn: bool
    score: float
    metric: str
    run: Dict[str, Any]
    std: float = 0.0
    n_runs: int = 1
    peers: List[Dict[str, Any]] = field(default_factory=list)
    # serve.yaml 显式指定权重时使用：覆盖默认权重路径，并标记"非选优来源"
    checkpoint_path: Optional[str] = None
    selected_by: str = "auto"

    @property
    def run_name(self) -> str:
        return self.run["run_name"]

    @property
    def run_dir(self) -> str:
        return self.run["run_dir"]

    @property
    def checkpoint(self) -> str:
        return self.checkpoint_path or os.path.join(self.run["run_dir"], CHECKPOINT_NAME)

    @property
    def seed(self):
        return self.run.get("seed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "backbone": self.backbone,
            "use_pinn": self.use_pinn,
            "seed": self.seed,
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "checkpoint": self.checkpoint,
            "metric": self.metric,
            "score": self.score,
            "std": self.std,
            "n_runs": self.n_runs,
            "metrics": self.run.get("metrics", {}),
            "mode": self.selected_by,
        }


def _best_single(runs: Sequence[Dict[str, Any]], metric: str):
    """在一组 run 中挑指标最好的那一次。"""
    direction = metric_direction(metric)
    scored = [(r, float(r["metrics"][metric])) for r in runs if metric in r.get("metrics", {})]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1] * direction)[0]


def _match_filters(run: Dict[str, Any], scenario=None, backbone=None,
                   use_pinn=None, seed=None) -> bool:
    if scenario is not None and run["scenario"] not in _to_list(scenario):
        return False
    if backbone is not None and run["backbone"] not in _to_list(backbone):
        return False
    if use_pinn is not None and bool(run["use_pinn"]) != bool(use_pinn):
        return False
    # run 的 seed 是 int，过滤条件来自命令行是 str，统一按字符串比较
    if seed is not None and str(run.get("seed")) not in _to_list(seed):
        return False
    return True


def collect_candidates(root, metric=DEFAULT_METRIC, aggregate="mean",
                       scenario=None, backbone=None, use_pinn=None, seed=None,
                       run_dir=None) -> List[Candidate]:
    """收集并排序候选模型，返回按优劣降序排列的列表。"""
    runs, _scenarios = collect_runs([root])
    runs = [r for r in runs if os.path.exists(os.path.join(r["run_dir"], CHECKPOINT_NAME))]
    if not runs:
        raise FileNotFoundError(
            f"在 {root} 下没有找到任何含 {CHECKPOINT_NAME} 的 run 目录。"
            "请先完成训练（python main.py），或显式传入 --run-dir。"
        )

    if run_dir:
        target = os.path.abspath(run_dir)
        matched = [r for r in runs if os.path.abspath(r["run_dir"]) == target]
        if not matched:
            raise FileNotFoundError(
                f"{run_dir} 下没有 {CHECKPOINT_NAME}，或该目录不是有效的 run 目录。"
            )
        run = matched[0]
        if metric not in run["metrics"]:
            raise KeyError(
                f"run {run['run_name']} 的指标中没有 {metric}，可用指标："
                f"{sorted(run['metrics'])}"
            )
        return [Candidate(
            scenario=run["scenario"], backbone=run["backbone"], use_pinn=bool(run["use_pinn"]),
            score=float(run["metrics"][metric]), metric=metric, run=run,
        )]

    runs = [r for r in runs if _match_filters(r, scenario, backbone, use_pinn, seed)]
    if not runs:
        raise ValueError("过滤条件过严，没有匹配的 run。")

    direction = metric_direction(metric)
    candidates: List[Candidate] = []

    if aggregate == "single":
        for run in runs:
            if metric not in run["metrics"]:
                continue
            candidates.append(Candidate(
                scenario=run["scenario"], backbone=run["backbone"],
                use_pinn=bool(run["use_pinn"]), score=float(run["metrics"][metric]),
                metric=metric, run=run,
            ))
    else:
        grouped = group_runs(runs)
        for (sc, bb, up), agg in grouped.items():
            stat = agg["metrics"].get(metric)
            if stat is None:
                continue
            peers = [r for r in runs
                     if (r["scenario"], r["backbone"], r["use_pinn"]) == (sc, bb, up)]
            best = _best_single(peers, metric)
            if best is None:
                continue
            candidates.append(Candidate(
                scenario=sc, backbone=bb, use_pinn=bool(up), score=float(stat["mean"]),
                metric=metric, run=best, std=float(stat["std"]), n_runs=int(agg["n"]),
                peers=peers,
            ))

    if not candidates:
        raise KeyError(f"所有 run 的指标里都没有 {metric}，请换一个 --metric。")

    candidates.sort(key=lambda c: c.score * direction, reverse=True)
    return candidates


def select_best_run(root, metric=DEFAULT_METRIC, aggregate="mean", scenario=None,
                    backbone=None, use_pinn=None, seed=None, run_dir=None) -> Candidate:
    """返回最优候选（列表第一项）。"""
    return collect_candidates(
        root, metric=metric, aggregate=aggregate, scenario=scenario,
        backbone=backbone, use_pinn=use_pinn, seed=seed, run_dir=run_dir,
    )[0]


def format_ranking(candidates: Sequence[Candidate], top: int = 10) -> str:
    """把候选排名渲染成可读表格（CLI --list 用）。"""
    if not candidates:
        return "（无候选）"
    head = f"{'#':>2}  {'scenario':<24}{'backbone':<12}{'PINN':<6}{'seed':>6}" \
           f"{candidates[0].metric:>16}  run"
    lines = [head, "-" * len(head)]
    for i, c in enumerate(candidates[:top], start=1):
        lines.append(
            f"{i:>2}  {c.scenario:<24}{c.backbone:<12}"
            f"{('PINN' if c.use_pinn else '-'):<6}{str(c.seed):>6}"
            f"{c.score:>16.6f}  {c.run_name}"
        )
    return "\n".join(lines)
