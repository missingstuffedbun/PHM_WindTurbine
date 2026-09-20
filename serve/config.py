"""推理服务配置（serve.yaml）解析。

服务**不做任何自动选优**：加载哪个权重完全由 `model.checkpoint` 一条路径决定。

三段配置的职责：
  - `model.checkpoint`  权重路径（唯一必填）；
  - `data`              **输入数据契约**：通道清单、目标、单位，以及"请求必须满足的数据要求"
                        （可见比例、失效通道、容差、违规处理），启动时与训练口径核对；
  - `preprocessing`     训练时的编码链路（window_size / missing_mode / missing_indicator），
                        启动时与实验 config.yaml 核对，不一致直接报错，防止口径漂移。

参数优先级：命令行参数 > serve.yaml > 训练产物里的实验 config.yaml > 内置默认值。

用法：

    from serve.config import ServeConfig
    cfg = ServeConfig.from_yaml("serve.yaml")
    print(cfg.checkpoint, cfg.input_channels, cfg.requirements)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

BACKBONES = ("lstm", "gru", "tcn", "transformer")
MISSING_MODES = ("raw_zero", "norm_zero")
VIOLATION_ACTIONS = ("warn", "reject")
DEFAULT_CONFIG_PATH = "serve.yaml"


def str2bool(value):
    if value is None or isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"无法解析为布尔值: {value!r}")


def parse_run_dir_name(name: str) -> Dict[str, Any]:
    """从 run 目录名还原 `{scenario}_{backbone}[_pinn][_seedN]`。

    解析不出来时对应字段返回 None，由调用方回退到实验 config.yaml / 报错。
    """
    tokens = str(name).split("_")
    seed = None
    if len(tokens) > 1 and tokens[-1].startswith("seed"):
        try:
            seed = int(tokens[-1][len("seed"):])
            tokens = tokens[:-1]
        except ValueError:
            seed = None
    pinn_token = bool(tokens) and tokens[-1].lower() == "pinn"
    if pinn_token:
        tokens = tokens[:-1]
    backbone, use_pinn = None, None
    if tokens and tokens[-1].lower() in BACKBONES:
        backbone = tokens.pop().lower()
        use_pinn = pinn_token
    scenario = "_".join(tokens) or None
    return {"scenario": scenario, "backbone": backbone,
            "use_pinn": use_pinn, "seed": seed}


@dataclass
class ServeConfig:
    """serve.yaml 的四段：model / data / preprocessing / server。"""

    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    server: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    # ------------------------------------------------------------------ 读取

    @classmethod
    def from_yaml(cls, path: str) -> "ServeConfig":
        p = os.path.abspath(str(path))
        if not os.path.exists(p):
            raise FileNotFoundError(f"服务配置文件不存在：{p}")
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{p} 的顶层结构应为映射（key: value）。")
        cfg = cls(
            model=dict(raw.get("model") or {}),
            data=dict(raw.get("data") or {}),
            preprocessing=dict(raw.get("preprocessing") or {}),
            server=dict(raw.get("server") or {}),
            source=p,
        )
        cfg.validate()
        return cfg

    def validate(self):
        backbone = self.backbone
        if backbone and str(backbone).lower() not in BACKBONES:
            raise ValueError(f"model.backbone={backbone!r} 非法，可选值：{BACKBONES}")
        mode = self.missing_mode
        if mode and str(mode).lower() not in MISSING_MODES:
            raise ValueError(f"preprocessing.missing_mode={mode!r} 非法，可选值：{MISSING_MODES}")
        action = str(self.requirements.get("on_violation", "warn")).lower()
        if action not in VIOLATION_ACTIONS:
            raise ValueError(
                f"data.requirements.on_violation={action!r} 非法，可选值：{VIOLATION_ACTIONS}")
        try:
            int(self.server.get("port", 8000))
        except (TypeError, ValueError):
            raise ValueError(f"server.port={self.server.get('port')!r} 不是整数。")

    # ------------------------------------------------------------------ model

    @property
    def checkpoint(self) -> Optional[str]:
        """唯一的必填项：best_model.pt 的路径。"""
        v = self.model.get("checkpoint")
        return str(v) if v else None

    @property
    def run_dir(self) -> Optional[str]:
        """权重所在目录（run 目录名里含 scenario / backbone / seed）。"""
        v = self.model.get("run_dir")
        if v:
            return str(v)
        return os.path.dirname(self.checkpoint) if self.checkpoint else None

    @property
    def config_file(self) -> Optional[str]:
        """还原模型结构用的实验 config.yaml；None = 从权重所在目录向上自动查找。"""
        v = self.model.get("config_file")
        return str(v) if v else None

    # 以下三项通常无需填写（由 run 目录名 / 实验 config.yaml 推断），仅在推断不出来时兜底
    @property
    def backbone(self) -> Optional[str]:
        v = self.model.get("backbone")
        return str(v).lower() if v else None

    @property
    def use_pinn(self) -> Optional[bool]:
        return str2bool(self.model.get("use_pinn"))

    @property
    def scenario(self) -> Optional[str]:
        v = self.model.get("scenario")
        return str(v) if v else None

    @property
    def seed(self) -> Optional[int]:
        v = self.model.get("seed")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ data

    @property
    def processed_file(self) -> Optional[str]:
        v = self.data.get("processed_file")
        return str(v) if v else None

    @property
    def input_channels(self) -> Optional[list]:
        v = self.data.get("input_channels")
        return list(v) if v else None

    @property
    def target_signals(self) -> Optional[list]:
        v = self.data.get("target_signals")
        return list(v) if v else None

    @property
    def units(self) -> str:
        return str(self.data.get("units") or "raw")

    @property
    def requirements(self) -> Dict[str, Any]:
        """输入数据要求：可见比例、失效通道、容差与违规处理。"""
        return dict(self.data.get("requirements") or {})

    @property
    def observable_ratio(self) -> Optional[float]:
        v = self.requirements.get("observable_ratio")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def tolerance(self) -> float:
        v = self.requirements.get("tolerance", 1.2)
        try:
            return float(v) if v else 1.2
        except (TypeError, ValueError):
            return 1.2

    @property
    def blocked(self) -> Optional[list]:
        v = self.requirements.get("blocked")
        return list(v) if v is not None else None

    @property
    def on_violation(self) -> str:
        return str(self.requirements.get("on_violation", "warn")).lower()

    @property
    def req_scenario(self) -> Optional[str]:
        v = self.requirements.get("scenario")
        return str(v) if v else None

    # ------------------------------------------------------- preprocessing

    @property
    def window_size(self) -> Optional[int]:
        v = self.preprocessing.get("window_size")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def missing_mode(self) -> Optional[str]:
        v = self.preprocessing.get("missing_mode")
        return str(v).lower() if v else None

    @property
    def missing_indicator(self) -> Optional[bool]:
        return str2bool(self.preprocessing.get("missing_indicator"))

    @property
    def strict(self) -> bool:
        """preprocessing 与训练口径不一致时是否直接报错（默认 true）。"""
        return str2bool(self.preprocessing.get("strict", True)) is not False

    # ------------------------------------------------------------------ server

    @property
    def host(self) -> str:
        return str(self.server.get("host") or "127.0.0.1")

    @property
    def port(self) -> int:
        return int(self.server.get("port") or 8000)

    @property
    def device(self) -> Optional[str]:
        v = self.server.get("device")
        return str(v) if v else None

    @property
    def base_url(self) -> str:
        """客户端用的服务地址；host 为 0.0.0.0 时自动换成本机回环地址。"""
        url = self.server.get("base_url")
        if url:
            return str(url).rstrip("/")
        host = "127.0.0.1" if self.host in ("0.0.0.0", "", "*") else self.host
        return f"http://{host}:{self.port}"

    # ------------------------------------------------------------------ 展示

    def describe(self) -> str:
        ch = self.input_channels
        ch_desc = f"{len(ch)} 个通道" if ch else "（未指定，取 processed.csv 表头）"
        req = self.requirements
        lines = [
            f"配置文件：{self.source or '（未指定，使用默认值）'}",
            f"  model.checkpoint   : {self.checkpoint or '（未指定）'}",
            f"  model.config_file  : {self.config_file or '（从权重所在目录向上自动查找）'}",
            f"  data.processed_file: {self.processed_file or '（取实验 config.yaml）'}",
            f"  data.input_channels: {ch_desc}",
            f"  data.target_signals: {self.target_signals or '（取实验 config.yaml / meta.yaml）'}",
            f"  data.units         : {self.units}",
            "  data.requirements  : "
            f"scenario={req.get('scenario') or '（推断）'} "
            f"observable_ratio={self.observable_ratio if self.observable_ratio is not None else '（取场景配置）'} "
            f"tolerance={self.tolerance:g} "
            f"blocked={self.blocked if self.blocked is not None else '（取场景配置）'} "
            f"on_violation={self.on_violation}",
            f"  preprocessing      : window_size={self.window_size or '（实验配置）'} "
            f"missing_mode={self.missing_mode or '（实验配置）'} "
            f"missing_indicator={self.missing_indicator if self.missing_indicator is not None else '（实验配置）'} "
            f"strict={self.strict}",
            f"  server             : {self.host}:{self.port} "
            f"device={self.device or 'auto'} url={self.base_url}",
        ]
        return "\n".join(lines)
