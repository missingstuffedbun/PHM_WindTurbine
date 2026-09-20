"""最优模型推理器：加载指定的 best_model.pt 并提供 predict()。

只依赖 torch / numpy / yaml，不含任何 Web 框架，可直接在离线脚本里调用。

两种加载方式：
  - `from_config(cfg)`：按 serve.yaml 显式指定的权重加载（**不做选优**，推荐）；
  - `from_results(...)`：按指标从实验结果目录里选优（实验对比 / 离线分析用）。

用法：

    from serve.config import ServeConfig
    from serve.predictor import BestModelPredictor

    p = BestModelPredictor.from_config(ServeConfig.from_yaml("serve.yaml"))
    out = p.predict(samples)      # samples: window_size 个原始采样点
    print(out["physical"])        # {"TMBNS": ..., "TMBEW": ..., "TMBTOR": ...}
"""

import copy
import os
import threading
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from models.base import build_model
from models.pinn import PINNWrapper
from serve.config import ServeConfig, parse_run_dir_name
from serve.preprocess import ServingPreprocessor
from serve.registry import DEFAULT_METRIC, Candidate, collect_candidates
from utils.analysis import load_scenarios


def _load_yaml(path) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_root_config(run_dir: str) -> Optional[str]:
    """从 run 目录向上找到实验根目录的 config.yaml（main.py 保存的副本）。"""
    path = os.path.abspath(run_dir)
    while True:
        cfg = os.path.join(path, "config.yaml")
        if os.path.exists(cfg):
            return cfg
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def _resolve_processed_file(processed_file: str, root_dir: str) -> str:
    """把实验记录里的 processed.csv 路径解析成本机存在的绝对路径。

    训练机上记录的多半是相对路径或绝对路径，换机器后可能都不在，
    因此依次尝试：原样 / 相对实验根目录 / 相对当前工作目录。
    """
    if not os.path.isabs(processed_file):
        cand = os.path.join(root_dir, processed_file)
        processed_file = cand if os.path.exists(cand) else os.path.abspath(processed_file)
    if not os.path.exists(processed_file):
        raise FileNotFoundError(
            f"processed 数据不存在: {processed_file}\n"
            "换机器部署时请在 serve.yaml 的 data.processed_file 里指定本机路径。"
        )
    return processed_file


def _build_contract(cfg: "ServeConfig") -> Dict:
    """把 serve.yaml 的 data.requirements 整理成请求校验用的契约。"""
    contract = dict(cfg.requirements or {})
    contract["on_violation"] = cfg.on_violation
    contract["tolerance"] = cfg.tolerance
    if cfg.observable_ratio is not None:
        contract["observable_ratio"] = cfg.observable_ratio
    if cfg.blocked is not None:
        contract["blocked"] = list(cfg.blocked)
    return contract


class BestModelPredictor:
    """把某个 run 的 best_model.pt 包装成一个可调用的推理器。"""

    def __init__(self, candidate: Candidate, config: Dict, root_dir: str,
                 processed_file: str, device=None, contract: Optional[Dict] = None):
        self.candidate = candidate
        self.config = config
        self.root_dir = root_dir
        self.processed_file = processed_file
        # serve.yaml 的 data.requirements：每次请求都按它校验输入分布
        self.contract = dict(contract or {})

        prep_cfg = dict(config.get("preprocessing") or {})
        targets = ((config.get("data") or {}).get("target_signals")
                   or prep_cfg.get("target_signals"))
        if not targets:
            raise ValueError("无法从实验 config.yaml 中确定 target_signals。")

        scenarios = load_scenarios([root_dir])
        scenario_cfg = scenarios.get(candidate.scenario, {}) or {}

        self.preprocessor = ServingPreprocessor(
            processed_file=processed_file,
            target_signals=list(targets),
            window_size=int(prep_cfg.get("window_size", 100)),
            missing_mode=prep_cfg.get("missing_mode", "raw_zero"),
            missing_indicator=bool(prep_cfg.get("missing_indicator", False)),
            blocked_signals=scenario_cfg.get("blocked_signals") or (),
        )
        self.scenario_cfg = scenario_cfg

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._build_model()
        self._buffers: Dict[str, deque] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 构建

    @classmethod
    def from_results(cls, results_dir="results", metric=DEFAULT_METRIC, aggregate="mean",
                     scenario=None, backbone=None, use_pinn=None, seed=None,
                     run_dir=None, processed_file=None, device=None) -> "BestModelPredictor":
        """从实验结果目录挑选最优模型并加载。"""
        root = os.path.abspath(results_dir)
        candidates = collect_candidates(
            root, metric=metric, aggregate=aggregate, scenario=scenario,
            backbone=backbone, use_pinn=use_pinn, seed=seed, run_dir=run_dir,
        )
        best = candidates[0]
        return cls.from_candidate(best, root, processed_file=processed_file, device=device)

    @classmethod
    def from_candidate(cls, candidate: Candidate, root_dir: str,
                       processed_file=None, device=None) -> "BestModelPredictor":
        """直接加载指定候选（供 --list / 指定 run 的场景复用）。"""
        config_path = find_root_config(candidate.run_dir)
        if config_path is None:
            raise FileNotFoundError(
                f"未找到 {candidate.run_dir} 所属实验根目录的 config.yaml，"
                "无法还原模型结构；请确认该 run 由 main.py 生成。"
            )
        config = _load_yaml(config_path)
        root_dir = os.path.dirname(config_path)

        if processed_file is None:
            processed_file = (config.get("data") or {}).get("processed_file")
        if not processed_file:
            raise ValueError(
                "实验 config.yaml 中缺少 data.processed_file，"
                "请用 serve.yaml 的 data.processed_file 或 --processed-file 指定。"
            )
        processed_file = _resolve_processed_file(processed_file, root_dir)
        return cls(candidate, config, root_dir, processed_file, device=device)

    @classmethod
    def from_config(cls, cfg: "ServeConfig") -> "BestModelPredictor":
        """按 serve.yaml 显式指定的权重加载，**不做任何选优**。

        只认 `model.checkpoint` 一条路径；模型结构、场景、编码口径均由
        权重所在目录 + 实验 config.yaml 还原，并与 serve.yaml 的数据契约核对。
        """
        if not cfg.checkpoint:
            raise ValueError(
                "serve.yaml 缺少 model.checkpoint：请填写 best_model.pt 的路径。"
            )
        checkpoint = os.path.abspath(str(cfg.checkpoint))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"serve.yaml 的 model.checkpoint 不存在：{checkpoint}")

        # run 目录名形如 {scenario}_{backbone}[_pinn][_seedN]，据此还原元信息
        run_dir = os.path.dirname(checkpoint)
        parsed = parse_run_dir_name(os.path.basename(run_dir))
        config_path = cfg.config_file or find_root_config(run_dir)
        config = copy.deepcopy(_load_yaml(config_path))
        root_dir = os.path.dirname(config_path) if config_path else run_dir

        mcfg = dict(config.get("model") or {})
        backbone = str(cfg.backbone or parsed.get("backbone") or mcfg.get("backbone") or "").lower()
        if not backbone:
            raise ValueError(
                "无法确定 backbone：run 目录名里没有骨架名、实验 config.yaml 也没有，"
                "请在 serve.yaml 的 model.backbone 显式指定（可选值：lstm / gru / tcn / transformer）。"
            )
        if cfg.use_pinn is not None:
            use_pinn = bool(cfg.use_pinn)
        elif parsed.get("use_pinn") is not None:
            use_pinn = bool(parsed["use_pinn"])
        else:
            use_pinn = bool(mcfg.get("use_pinn", False))
        scenario = cfg.scenario or cfg.req_scenario or parsed.get("scenario") or "-"

        # 配置显式值与 run 目录名矛盾时直接报错：否则会拿错结构去套权重
        name = os.path.basename(os.path.normpath(run_dir))
        if parsed.get("backbone") and parsed["backbone"] != backbone:
            raise ValueError(
                f"backbone 冲突：配置写的是 {backbone!r}，但 run 目录 {name!r} 表明它是 "
                f"{parsed['backbone']!r}；请改 model.backbone 或删掉该字段交由目录名推断。"
            )
        if parsed.get("use_pinn") is not None and bool(parsed["use_pinn"]) != bool(use_pinn):
            raise ValueError(
                f"use_pinn 冲突：配置写的是 {use_pinn}，但 run 目录 {name!r} 表明它是 "
                f"{bool(parsed['use_pinn'])}；请改 model.use_pinn 或删掉该字段交由目录名推断。"
            )

        processed_file = cfg.processed_file or (config.get("data") or {}).get("processed_file")
        if not processed_file:
            raise ValueError(
                "无法确定 processed.csv：实验 config.yaml 缺少 data.processed_file，"
                "请在 serve.yaml 的 data.processed_file 指定。"
            )
        processed_file = _resolve_processed_file(processed_file, root_dir)

        # ---- preprocessing：必须与训练时的实验 config.yaml 同口径 ----
        exp_prep = dict(config.get("preprocessing") or {})
        prep = cls._resolve_preprocessing(cfg, exp_prep)
        config["preprocessing"] = dict(exp_prep, **prep)

        data = dict(config.get("data") or {})
        if cfg.target_signals:
            data["target_signals"] = list(cfg.target_signals)
        config["data"] = data

        metrics = _load_yaml(os.path.join(run_dir, "metrics.yaml")) or {}
        run = {
            "run_name": os.path.basename(os.path.normpath(run_dir)),
            "run_dir": run_dir,
            "scenario": scenario,
            "backbone": backbone,
            "use_pinn": use_pinn,
            "seed": cfg.seed if cfg.seed is not None else parsed.get("seed"),
            "metrics": metrics,
        }
        candidate = Candidate(
            scenario=scenario, backbone=backbone, use_pinn=use_pinn,
            score=(float(metrics[DEFAULT_METRIC]) if DEFAULT_METRIC in metrics else None),
            metric=DEFAULT_METRIC, run=run,
            checkpoint_path=checkpoint, selected_by="explicit",
        )
        obj = cls(candidate, config, root_dir, processed_file,
                  device=cfg.device, contract=_build_contract(cfg))
        obj._check_data_contract(cfg)      # 需要 preprocessor 建好之后才能核对
        return obj

    @staticmethod
    def _resolve_preprocessing(cfg: "ServeConfig", exp_prep: Dict) -> Dict:
        """取预处理口径：serve.yaml 留空则沿用训练配置，写了就与训练配置核对。"""
        exp = {
            "window_size": int(exp_prep.get("window_size", 100)),
            "missing_mode": str(exp_prep.get("missing_mode", "raw_zero")).lower(),
            "missing_indicator": bool(exp_prep.get("missing_indicator", False)),
        }
        got = {
            "window_size": int(cfg.window_size) if cfg.window_size else exp["window_size"],
            "missing_mode": str(cfg.missing_mode).lower() if cfg.missing_mode else exp["missing_mode"],
            "missing_indicator": bool(cfg.missing_indicator)
            if cfg.missing_indicator is not None else exp["missing_indicator"],
        }
        diffs = [f"{k}：配置 {got[k]!r} != 训练 {exp[k]!r}" for k in exp if got[k] != exp[k]]
        if diffs:
            msg = ("preprocessing 与训练时的实验 config.yaml 不一致：" + "；".join(diffs)
                   + "。请改成训练口径；确需偏离（复现消融实验）请把 preprocessing.strict 设为 false。")
            if cfg.strict:
                raise ValueError(msg)
            warnings.warn(msg + "（strict=false，按 serve.yaml 的值运行，结果不可与离线指标对照）")
        return got

    def _check_data_contract(self, cfg: "ServeConfig"):
        """启动时核对 serve.yaml 的 data 契约与训练口径是否一致。"""
        problems = []
        want = cfg.input_channels
        actual = self.preprocessor.input_channels
        if want and list(want) != actual:
            problems.append(
                "data.input_channels 与 processed.csv 推导出的通道不一致"
                f"（配置 {len(list(want))} 个 / 实际 {len(actual)} 个；"
                f"仅配置有：{sorted(set(want) - set(actual))}，"
                f"仅数据有：{sorted(set(actual) - set(want))}）"
            )
        tg = cfg.target_signals
        if tg and list(tg) != list(self.preprocessor.target_signals):
            problems.append(
                f"data.target_signals={list(tg)} 与训练口径 {list(self.preprocessor.target_signals)} 不一致"
            )
        ratio, s_ratio = cfg.observable_ratio, self.scenario_cfg.get("observable_ratio")
        if ratio is not None and s_ratio is not None and abs(float(ratio) - float(s_ratio)) > 1e-6:
            problems.append(
                f"data.requirements.observable_ratio={ratio} 与场景 {self.candidate.scenario} "
                f"的 {s_ratio} 不一致（可见通道比例不同 = 输入分布不同）"
            )
        blocked, s_blocked = cfg.blocked, sorted(self.scenario_cfg.get("blocked_signals") or [])
        if blocked is not None and sorted(blocked) != s_blocked:
            problems.append(
                f"data.requirements.blocked={sorted(blocked)} 与场景 {self.candidate.scenario} "
                f"的失效通道 {s_blocked} 不一致"
            )
        req_scenario = cfg.req_scenario
        if req_scenario and self.candidate.scenario not in ("-", "") \
                and req_scenario != self.candidate.scenario:
            problems.append(
                f"data.requirements.scenario={req_scenario} 与权重所在 run 的场景 "
                f"{self.candidate.scenario} 不一致"
            )
        if problems:
            raise ValueError(
                "serve.yaml 的 data 段与训练口径不一致：\n  - " + "\n  - ".join(problems)
                + "\n请按 /schema 或实验 config.yaml 修正后重启。"
            )

    def _build_model(self) -> torch.nn.Module:
        cfg = copy.deepcopy(self.config)
        cfg.setdefault("model", {})
        cfg["model"]["backbone"] = self.candidate.backbone
        cfg["model"]["use_pinn"] = self.candidate.use_pinn
        # 维度按当前数据的通道顺序重新推导，不使用训练时的硬编码值
        cfg["model"]["input_dim"] = self.preprocessor.model_input_dim
        cfg["model"]["output_dim"] = len(self.preprocessor.target_signals)
        cfg.setdefault("preprocessing", {})
        cfg["preprocessing"]["target_signals"] = list(self.preprocessor.target_signals)
        cfg.setdefault("data", {})
        cfg["data"]["target_signals"] = list(self.preprocessor.target_signals)

        if self.candidate.use_pinn:
            model = PINNWrapper(cfg, scaler=self.preprocessor.scaler)
            model.feature_names = self.preprocessor.input_channels
        else:
            model = build_model(cfg)

        state = torch.load(self.candidate.checkpoint, map_location=self.device)
        self._load_state_dict(model, state)
        return model.to(self.device).eval()

    @staticmethod
    def _load_state_dict(model: torch.nn.Module, state: Dict):
        """加载权重，并兼容旧版遗留的物理模块参数。

        旧版 `PhysicsConstraints` 含 `k_bending` / `k_torsion` 两个可学习比例系数，
        现行版本已移除（不可辨识，见 models/physics.py 文件头第 4 条）。这两个参数
        不参与推理期的任何计算，丢弃不影响输出；其余任何键不匹配都直接报错。
        """
        own = set(model.state_dict().keys())
        dropped = sorted(k for k in state if k not in own)
        if dropped:
            legacy = [k for k in dropped if k.startswith("phys_module.")]
            if len(legacy) != len(dropped):
                raise RuntimeError(
                    f"权重与当前模型结构不匹配，多余的键：{dropped}。"
                    "请确认该 best_model.pt 与当前代码版本一致。"
                )
            warnings.warn(
                f"忽略旧权重中已移除的物理模块参数：{legacy}"
                "（不可辨识参数，推理期不参与计算，不影响输出）"
            )
        missing = sorted(k for k in own if k not in state)
        if missing:
            raise RuntimeError(
                f"权重缺少模型参数：{missing}。"
                "请确认该 best_model.pt 与当前代码版本一致。"
            )
        model.load_state_dict({k: v for k, v in state.items() if k in own})

    # ------------------------------------------------------------------ 推理

    def predict(self, samples: Sequence[Dict], session_id: Optional[str] = None,
                return_standardized: bool = True) -> Dict[str, Any]:
        """对一组采样点做一次预测。

        samples：按时间顺序排列的原始采样点，每个元素为 {通道名: 原始工程值}。
          - 不传 session_id：samples 至少要有 window_size 个（多用最后 window_size 个）；
          - 传 session_id：samples 追加进该会话的滚动缓存，凑够 window_size 才出结果。
        """
        samples = list(samples or [])
        if not samples:
            raise ValueError("samples 为空。")

        if session_id is not None:
            with self._lock:
                buf = self._buffers.setdefault(
                    session_id, deque(maxlen=self.preprocessor.window_size))
                buf.extend(samples)
                window = list(buf)
            if len(window) < self.preprocessor.window_size:
                return {
                    "ready": False,
                    "status": "warming_up",
                    "session_id": session_id,
                    "buffer_length": len(window),
                    "required": self.preprocessor.window_size,
                    "model": self._model_stub(),
                }
        else:
            if len(samples) < self.preprocessor.window_size:
                raise ValueError(
                    f"样本数不足：需要 {self.preprocessor.window_size} 个连续采样点，"
                    f"实际收到 {len(samples)} 个；"
                    "如需流式累积请传 session_id。"
                )
            window = samples[-self.preprocessor.window_size:]

        # 先按数据契约校验输入分布，再决定是拒绝还是带告警继续推理
        last_obs = self.preprocessor.observed_mask(window[-1])
        observed = [c for c, ok in zip(self.preprocessor.input_channels, last_obs) if ok]
        missing = [c for c, ok in zip(self.preprocessor.input_channels, last_obs) if not ok]
        violations, unknown = self._check_input_contract(window, observed)
        if violations and self._reject_on_violation():
            raise ValueError(violations[0])

        x = self.preprocessor.encode_window(window)
        with torch.no_grad():
            z = self.model(torch.from_numpy(x).to(self.device))[0].detach().cpu().numpy()

        standardized = {name: float(v)
                        for name, v in zip(self.preprocessor.target_signals, z)}
        physical = self.preprocessor.inverse_targets(z)

        result: Dict[str, Any] = {
            "ready": True,
            "status": "ok",
            "physical": physical,
            "model": self._model_stub(),
            "diagnostics": {
                "window_size": self.preprocessor.window_size,
                "n_samples_used": self.preprocessor.window_size,
                "n_observed": len(observed),
                "n_missing": len(missing),
                "observed_channels": observed,
                "missing_channels": missing,
            },
            "warnings": list(violations),
        }
        if unknown:
            result["diagnostics"]["unknown_channels"] = unknown
        if session_id is not None:
            result["session_id"] = session_id
            result["diagnostics"]["buffer_length"] = len(window)
        if return_standardized:
            result["standardized"] = standardized

        return result

    def _check_input_contract(self, window: Sequence[Dict], observed: List[str]):
        """按 serve.yaml 的 data.requirements 校验一次请求的输入分布。

        返回 (违规描述列表, 未知通道列表)：前者非空表示输入分布与训练不一致。
        """
        contract = self.contract or {}
        n = len(self.preprocessor.input_channels)
        known = set(self.preprocessor.input_channels)
        unknown = sorted({k for s in window for k in s.keys() if k not in known})
        msgs: List[str] = []
        if unknown:
            msgs.append(
                f"请求含 {len(unknown)} 个模型未知的通道 {unknown}（已忽略），"
                "请对照 /schema 的 input_channels 准备数据。"
            )

        ratio = contract.get("observable_ratio")
        ratio = float(ratio) if ratio is not None else \
            float(self.scenario_cfg.get("observable_ratio", 1.0) or 1.0)
        tol = float(contract.get("tolerance", 1.2) or 1.2)
        if ratio < 1.0:
            typical = ratio * n
            if len(observed) > typical * tol:
                msgs.append(
                    f"该模型训练于稀疏场景（observable_ratio={ratio:g}），当前请求可见 "
                    f"{len(observed)}/{n} 个通道，多于训练时的典型可见数量（≈{typical:.0f}，"
                    f"容差 ×{tol:g}），输入分布与训练不一致。"
                )
            elif len(observed) < typical / tol:
                msgs.append(
                    f"当前请求可见 {len(observed)}/{n} 个通道，少于该模型训练场景的典型可见数量"
                    f"（observable_ratio={ratio:g}，≈{typical:.0f}，容差 ×{tol:g}），"
                    "输入分布与训练不一致。"
                )
        return msgs, unknown

    def _reject_on_violation(self) -> bool:
        return str((self.contract or {}).get("on_violation", "warn")).lower() == "reject"

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            return self._buffers.pop(session_id, None) is not None

    def list_sessions(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._buffers.items()}

    def _model_stub(self) -> Dict[str, Any]:
        c = self.candidate
        return {
            "run_name": c.run_name,
            "scenario": c.scenario,
            "backbone": c.backbone,
            "use_pinn": c.use_pinn,
            "seed": c.seed,
        }

    # ------------------------------------------------------------------ 元信息

    def info(self) -> Dict[str, Any]:
        """服务所加载模型的完整描述（/model 接口）。"""
        c = self.candidate
        return {
            "run_name": c.run_name,
            "run_dir": c.run_dir,
            "checkpoint": c.checkpoint,
            "scenario": c.scenario,
            "scenario_config": {
                "observable_ratio": float(self.scenario_cfg.get("observable_ratio", 1.0) or 1.0),
                "level": str(self.scenario_cfg.get("level") or "channel"),
                "blocked_signals": list(self.scenario_cfg.get("blocked_signals") or []),
                "description": self.scenario_cfg.get("description", ""),
            },
            "backbone": c.backbone,
            "use_pinn": c.use_pinn,
            "seed": c.seed,
            "selection": {
                "mode": c.selected_by,
                "mode_note": "explicit = serve.yaml 显式指定的权重；auto = 按指标选优",
                "metric": c.metric,
                "score": c.score,
                "std_over_seeds": c.std,
                "n_runs_in_group": c.n_runs,
                "test_metrics": c.run.get("metrics", {}),
            },
            "data": {
                "processed_file": self.processed_file,
                "target_signals": list(self.preprocessor.target_signals),
            },
            "input": self.preprocessor.describe(),
            # 调用方必须满足的数据要求（来自 serve.yaml 的 data.requirements）
            "input_requirements": {
                "units": self.contract.get(
                    "units", "输入为原始工程量纲（与 data/raw 一致），服务内部完成标准化"),
                "input_channels": list(self.preprocessor.input_channels),
                "n_input_channels": len(self.preprocessor.input_channels),
                "window_size": self.preprocessor.window_size,
                "missing_mode": self.preprocessor.missing_mode,
                "missing_indicator": self.preprocessor.missing_indicator,
                "blocked_signals": sorted(self.preprocessor.blocked),
                "observable_ratio": float(self.scenario_cfg.get("observable_ratio", 1.0) or 1.0),
                "expected_observed_channels": round(
                    float(self.scenario_cfg.get("observable_ratio", 1.0) or 1.0)
                    * len(self.preprocessor.input_channels), 1),
                "tolerance": float(self.contract.get("tolerance", 1.2) or 1.2),
                "on_violation": self._reject_on_violation() and "reject" or "warn",
            },
            "device": str(self.device),
        }

    # ------------------------------------------------------------------ 自检

    def self_check(self) -> Dict[str, Any]:
        """端到端自检：取 processed.csv 末尾一个窗口，反标准化回原始量纲后走一遍完整链路。

        用于验证「原始值 -> 标准化 -> 模型 -> 反标准化」是否与离线数据对齐；
        它用的是时间顺序切分下测试段的末尾窗口，仅作链路体检，不作为精度指标。
        """
        window = self.preprocessor.window_size
        meta_path = os.path.join(os.path.dirname(self.processed_file), "meta.yaml")
        meta = _load_yaml(meta_path)
        n_rows = int(meta.get("rows", 0)) if meta else 0
        if n_rows <= 0:
            df = pd.read_csv(self.processed_file, usecols=[0])
            n_rows = len(df)
        skip = max(0, n_rows - window)
        df = pd.read_csv(self.processed_file, skiprows=range(1, skip + 1), nrows=window)

        samples: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            sample = {}
            for col in self.preprocessor.input_channels:
                z = float(row[col])
                mean, scale = self.preprocessor.scaler[col]
                sample[col] = z * scale + mean
            samples.append(sample)

        out = self.predict(samples)
        truth = {t: float(df.iloc[-1][t]) for t in self.preprocessor.target_signals}
        pred = out["standardized"]
        errors = {t: abs(pred[t] - truth[t]) for t in truth}
        out["self_check"] = {
            "source": f"{self.processed_file} 末尾 {window} 行（原始量纲回代）",
            "true_standardized": truth,
            "abs_error_standardized": errors,
            "note": "链路体检用：误差应远小于该 run 的 overall_rmse 量级之外的明显异常；"
                    "若误差与该 run 测试集 RMSE 同量级，说明编码链路自洽。",
        }
        return out
