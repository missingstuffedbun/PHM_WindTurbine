"""对外推理服务：以 HTTP API 的形式暴露一个已训练好的模型。

**默认按 serve.yaml 加载指定权重，不做任何自动选优**（换模型只改 `model.checkpoint`
这一条路径）。`data` 段声明输入数据要求、`preprocessing` 段锁定训练时的编码口径，
启动时都会与训练产物核对，不一致直接报错。命令行参数可临时覆盖配置。

启动：

    # 加载 serve.yaml 里指定的 best_model.pt
    python -m serve.server --config serve.yaml
    python -m serve.server                       # 默认就是 --config serve.yaml

    # 临时换权重 / 换端口（不动配置文件）
    python -m serve.server --run-dir results/xxx/s0_full_gru_pinn_seed42
    python -m serve.server --checkpoint /path/to/other.pt --port 9000

    # 打印配置摘要 / 只做一次端到端自检（不启动服务）
    python -m serve.server --list
    python -m serve.server --check

    # 没有 serve.yaml 时退回旧行为：按指标从 results/ 里选优
    python -m serve.server --results-dir results --metric overall_rmse

依赖：fastapi + uvicorn（`pip install fastapi uvicorn`）。
不装也能用：`--list` / `--check` 以及 `serve.predictor` 只依赖 torch。
"""

import argparse
import os
from typing import Dict, List, Optional

from serve.config import DEFAULT_CONFIG_PATH, ServeConfig
from serve.predictor import BestModelPredictor
from serve.registry import DEFAULT_METRIC, collect_candidates, format_ranking


def _str2bool(value):
    if value is None or isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"无法解析为布尔值: {value!r}")


def build_predictor(args) -> BestModelPredictor:
    return BestModelPredictor.from_results(
        results_dir=args.results_dir,
        metric=args.metric,
        aggregate=args.aggregate,
        scenario=args.scenario,
        backbone=args.backbone,
        use_pinn=_str2bool(args.use_pinn),
        seed=args.seed,
        run_dir=args.run_dir,
        processed_file=args.processed_file,
        device=args.device,
    )


def create_app(predictor: BestModelPredictor):
    """构造 FastAPI 应用（fastapi 在此处才导入，便于无 Web 依赖时使用 CLI 其他子命令）。"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "缺少 Web 依赖，请执行：pip install fastapi uvicorn\n"
            f"（原始错误：{exc}）"
        )

    class PredictRequest(BaseModel):
        samples: List[Dict[str, Optional[float]]]
        session_id: Optional[str] = None
        return_standardized: bool = True

    class BatchRequest(BaseModel):
        windows: List[List[Dict[str, Optional[float]]]]

    app = FastAPI(
        title="风机塔底响应恢复服务",
        description="稀疏监测条件下由最优模型恢复塔底弯矩 / 扭矩（TMBNS / TMBEW / TMBTOR）",
        version="1.0.0",
    )

    @app.get("/")
    def index():
        return {
            "service": "风机塔底响应恢复服务",
            "endpoints": {
                "GET /health": "存活与模型就绪状态",
                "GET /model": "当前服务的模型与选优信息",
                "GET /schema": "输入通道、窗口长度与缺失语义",
                "POST /predict": "单窗口预测（可带 session_id 流式累积）",
                "POST /predict/batch": "多窗口批量预测",
                "GET /sessions": "流式会话缓存状态",
                "DELETE /sessions/{session_id}": "清空某个会话缓存",
            },
            "docs": "/docs",
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": predictor.candidate.run_name,
            "device": str(predictor.device),
            "sessions": predictor.list_sessions(),
        }

    @app.get("/model")
    def model_info():
        return predictor.info()

    @app.get("/schema")
    def schema():
        info = predictor.info()
        req = info["input_requirements"]
        return {
            "input": info["input"],
            "targets": info["data"]["target_signals"],
            "scenario": info["scenario_config"],
            # 调用方必须满足的数据要求：不满足会进 warnings，或按 on_violation=reject 直接 400
            "requirements": req,
            "semantics": {
                "values": "原始工程量纲（与 data/raw 一致），服务内部用 scaler.npz 标准化",
                "missing": "通道缺省 / 值为 null / NaN / inf 均视为缺失，"
                           "按 missing_mode 填入哨兵值，并在 missing_indicator 下置 0",
                "ordering": "samples 按时间先后顺序排列，取最后 window_size 个",
                "observed": f"可见通道数应接近 {req['expected_observed_channels']} 个"
                            f"（observable_ratio={req['observable_ratio']:g}，"
                            f"容差 ×{req['tolerance']:g}），偏离会被告警为输入分布不一致",
                "output": "physical = 原始工程量纲；standardized = 标准化域（可直接与离线指标对比）",
            },
        }

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            return predictor.predict(
                req.samples, session_id=req.session_id,
                return_standardized=req.return_standardized,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/predict/batch")
    def predict_batch(req: BatchRequest):
        if not req.windows:
            raise HTTPException(status_code=400, detail="windows 为空。")
        results, errors = [], []
        for i, window in enumerate(req.windows):
            try:
                results.append(predictor.predict(window))
            except ValueError as exc:
                errors.append({"index": i, "error": str(exc)})
        if errors and not results:
            raise HTTPException(status_code=400, detail=errors)
        return {"results": results, "errors": errors, "n_windows": len(results)}

    @app.get("/sessions")
    def sessions():
        return {"sessions": predictor.list_sessions()}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        return {"session_id": session_id, "removed": predictor.reset_session(session_id)}

    return app


def resolve_config(args) -> ServeConfig:
    """以 serve.yaml 为底、命令行为顶解析配置。"""
    path = os.path.abspath(args.config)
    if os.path.exists(path):
        cfg = ServeConfig.from_yaml(path)
    elif args.config != DEFAULT_CONFIG_PATH:
        raise FileNotFoundError(f"--config 指定的配置文件不存在：{path}")
    else:
        cfg = ServeConfig()          # 没有 serve.yaml：退回按指标选优

    if args.run_dir:      # 兼容旧写法：run 目录 -> 该目录下的 best_model.pt
        cfg.model["checkpoint"] = os.path.join(args.run_dir, "best_model.pt")
    if args.checkpoint:
        cfg.model["checkpoint"] = args.checkpoint
    if args.processed_file:
        cfg.data["processed_file"] = args.processed_file
    if args.backbone:
        cfg.model["backbone"] = args.backbone[0]
    if args.use_pinn is not None:
        cfg.model["use_pinn"] = _str2bool(args.use_pinn)
    if args.scenario:
        cfg.model["scenario"] = args.scenario[0]
    if args.device:
        cfg.server["device"] = args.device
    if args.host:
        cfg.server["host"] = args.host
    if args.port:
        cfg.server["port"] = args.port
    cfg.validate()
    return cfg


def build_arg_parser():
    p = argparse.ArgumentParser(description="启动模型的对外推理服务")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help=f"服务配置文件，默认 {DEFAULT_CONFIG_PATH}"
                        "（文件里指定了权重就不做选优；文件不存在则退回按指标选优）")
    p.add_argument("--results-dir", default="results",
                   help="仅在没有 serve.yaml、退回选优时使用：实验结果根目录")
    p.add_argument("--run-dir", default=None,
                   help="覆盖 serve.yaml 的 model.run_dir（直接指定某个 run 目录）")
    p.add_argument("--checkpoint", default=None,
                   help="覆盖 serve.yaml 的 model.checkpoint（直接指定 .pt 权重文件）")
    p.add_argument("--processed-file", default=None,
                   help="覆盖 serve.yaml 的 data.processed_file（换机器部署时用）")
    p.add_argument("--device", default=None, help="推理设备，默认 cuda 可用则 cuda 否则 cpu")
    p.add_argument("--host", default=None, help="监听地址，默认取 serve.yaml")
    p.add_argument("--port", type=int, default=None, help="监听端口，默认取 serve.yaml")
    p.add_argument("--list", action="store_true",
                   help="只打印配置摘要（无 serve.yaml 时打印候选排行榜），不启动服务")
    p.add_argument("--check", action="store_true",
                   help="只做一次端到端自检（从 processed.csv 取窗口回代），不启动服务")

    # 以下仅在"没有 serve.yaml、退回按指标选优"时生效
    p.add_argument("--metric", default=DEFAULT_METRIC,
                   help=f"选优指标，默认 {DEFAULT_METRIC}（*_r2 越大越好，其余越小越好）")
    p.add_argument("--aggregate", choices=("mean", "single"), default="mean",
                   help="多种子聚合方式：mean=按组均值排序（默认），single=按单次 run 排序")
    p.add_argument("--scenario", nargs="+", default=None, help="只考虑指定场景")
    p.add_argument("--backbone", nargs="+", default=None, help="只考虑指定 backbone")
    p.add_argument("--use-pinn", default=None, metavar="{true,false}",
                   help="只考虑 PINN（true）或 baseline（false）")
    p.add_argument("--seed", nargs="+", default=None, help="只考虑指定 seed")
    p.add_argument("--top", type=int, default=10, help="--list（排行榜）显示的条数")
    return p


def main():
    args = build_arg_parser().parse_args()
    cfg = resolve_config(args)

    if args.list:
        if cfg.checkpoint:
            print(cfg.describe())
        else:
            candidates = collect_candidates(
                os.path.abspath(args.results_dir), metric=args.metric,
                aggregate=args.aggregate, scenario=args.scenario, backbone=args.backbone,
                use_pinn=_str2bool(args.use_pinn), seed=args.seed, run_dir=args.run_dir,
            )
            print(f"选优指标：{args.metric}（{'越大越好' if args.metric.endswith('r2') else '越小越好'}）"
                  f"　聚合方式：{args.aggregate}")
            print(format_ranking(candidates, top=args.top))
            if len({c.scenario for c in candidates}) > 1:
                print("\n提示：候选跨多个场景，不同场景的观测条件不同，指标并非同口径；"
                      "部署到具体场景时建议加 --scenario 限定。")
        return

    if cfg.checkpoint:
        # serve.yaml / 命令行显式指定的权重：不做任何选优
        predictor = BestModelPredictor.from_config(cfg)
        c = predictor.candidate
        score = f" | 该 run 测试集 {c.metric} = {c.score:.6f}" if c.score is not None else ""
        print(f"已加载模型（配置指定，未做选优）：{c.run_name} "
              f"({c.backbone}{' + PINN' if c.use_pinn else ''}"
              f" | 场景 {c.scenario} | seed {c.seed}{score})")
        print(f"权重：{c.checkpoint}")
        print(f"配置：{cfg.source or '（命令行）'}　设备：{predictor.device}")
    else:
        # 没有 serve.yaml：退回旧行为，按指标从 results/ 里选优
        print(f"未指定 model.run_dir，退回按指标选优："
              f"results-dir={args.results_dir} metric={args.metric}")
        predictor = build_predictor(args)
        print(f"已加载模型：{predictor.candidate.run_name} "
              f"({predictor.candidate.backbone}"
              f"{' + PINN' if predictor.candidate.use_pinn else ''}) "
              f"| 场景 {predictor.candidate.scenario} | {args.metric}={predictor.candidate.score:.6f}")
        print(f"权重：{predictor.candidate.checkpoint}　设备：{predictor.device}")

    if args.check:
        out = predictor.self_check()
        print("\n自检结果（标准化域）：")
        for name, err in out["self_check"]["abs_error_standardized"].items():
            print(f"  {name}: 预测 {out['standardized'][name]:+.4f} | "
                  f"真值 {out['self_check']['true_standardized'][name]:+.4f} | "
                  f"绝对误差 {err:.4f}")
        print(f"  该 run 测试集 overall_rmse = "
              f"{predictor.candidate.run['metrics'].get('overall_rmse', '-')}")
        return

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        raise SystemExit("缺少 uvicorn，请执行：pip install uvicorn")

    app = create_app(predictor)
    print(f"监听：http://{cfg.host}:{cfg.port}　接口文档：http://{cfg.host}:{cfg.port}/docs")
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
