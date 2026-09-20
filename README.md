# PHM_WindTurbine

基于稀疏传感与物理约束的风机塔架结构健康监测项目。

## 文件结构

```
PHM_WindTurbine/
│
├── README.md
├── physics.md             # 物理约束说明：塔底弯矩-推力/倾覆力矩平衡式、简化假设、量纲推导、误差范围
├── config.yaml            # 训练参数（数据入口、场景名称列表、模型、训练、损失）
├── process.yaml           # 数据处理参数（选点、清洗、标准化；含 20 Hz / 100 Hz 两个版本）
├── serve.yaml             # 推理服务配置：指定加载哪一份 best_model.pt（不做自动选优）
├── config/
│   └── scenarios.yaml     # 场景策略定义（observable_ratio / level / blocked_signals）
├── main.py                # 训练入口，支持单场景或多场景批量实验，接受 --config / --scenario
├── analyze_results.py     # 结果分析入口：读取已完成实验目录，生成 HTML 报表（含物理约束增益边界分析）
├── research_direction.md  # 研究方案：问题定义、场景设计、对比方法与已知局限
├── diagnose_ns_ew.py      # 诊断：NS/EW 旋转到 fore-aft / side-side 后的可预测性对照（含 20/100 Hz 跨版本）
├── check_frame.py         # 诊断：旋转约定确认（δ 扫描）+ 数据时间连续性检查
│
├── data/
│   ├── raw/               # 原始 Björkö 风机数据（B1_CL4_20.csv / B1_CL4_100.csv）
│   │                      #   Bjorko_Sensors_Specs_Metadata.csv：传感器规格与可靠标记
│   ├── processed/         # 数据处理产物，每个版本一个目录
│   │   ├── B1_CL4_20/     #   processed.csv + scaler.npz + meta.yaml（20 Hz）
│   │   └── B1_CL4_100/    #   processed.csv + scaler.npz + meta.yaml（100 Hz）
│   └── dataset.py         # WindTurbineDataset（加窗 + 场景 mask）/ build_split_datasets
│
├── preprocessing/
│   └── prepare_data.py    # 按 process.yaml 批量清洗、归一化、落盘（raw -> processed）
│
├── models/
│   ├── base.py            # 数据驱动 backbone：LSTMModel / GRUModel / TCNModel / TransformerModel / build_model
│   ├── pinn.py            # PINNWrapper：包装任意 backbone，compute_loss 返回 data_loss + physics_loss
│   └── physics.py         # 物理约束残差定义
│
├── utils/
│   ├── metrics.py         # RMSE / MAE / R² / SMAPE，分目标变量评价
│   ├── visualize.py       # 预测对比图、散点图、误差分布图
│   └── analysis.py        # 实验结果汇总、物理约束增益与边界分析
│
├── serve/                 # 推理服务：把一份训练好的 best_model.pt 包装成 HTTP API
│   ├── config.py          # serve.yaml 解析（权重由配置显式指定，不做选优）
│   ├── registry.py        # 选优：按指标排序 + 定位 best_model.pt（无 serve.yaml 时回退用）
│   ├── preprocess.py      # 推理侧输入构造（原始工程量纲 -> 模型输入张量）
│   ├── predictor.py       # 加载权重并提供 predict()（纯 torch，无 Web 依赖）
│   ├── server.py          # FastAPI 封装 + 启动入口
│   └── client_example.py  # 冒烟测试客户端（仅用标准库）
│
└── results/               # 每次运行一个独立文件夹：{时间戳}_{场景}_{backbone}[_pinn]/
    └── <run_name>/
        ├── config.yaml    # 本次运行的完整配置副本
        ├── best_model.pt  # 验证集最优模型权重
        ├── results.npz    # y_true / y_pred
        ├── metrics.yaml   # 总体 + 分目标指标
        └── *.png          # 3 张可视化图
```

## 代码内容

### `preprocessing/prepare_data.py` + `process.yaml`
数据处理阶段与训练完全解耦，配置集中在 `process.yaml`：

- `raw_dir` / `metadata_file`：原始数据目录与传感器规格元数据；
- `input_signals` / `target_signals`：输入通道候选与塔底响应目标（实际可用通道再按 metadata 的 `Reliable Measurement` 过滤）；
- `clean`：删除全空行、`±inf → NaN`、前后向填充；
- `datasets`：需要产出的版本列表。默认同时产出 **20 Hz（`B1_CL4_20.csv`）与 100 Hz（`B1_CL4_100.csv`）两个版本**，各写入独立目录，互不覆盖。

每个版本目录下产出三个文件：`processed.csv`（标准化后的数据）、`scaler.npz`（标准化参数，供缺失编码还原原始域）、`meta.yaml`（来源文件 / 行数 / 输入与目标信号名）。

训练侧只读取其中一份文件，不再参与任何数据处理：在 `config.yaml` 的 `data.processed_file` 中切换即可（目标信号自动从同目录 `meta.yaml` 读取）：

```yaml
data:
  processed_file: "data/processed/B1_CL4_20/processed.csv"   # 换成 .../B1_CL4_100/processed.csv 即切到 100 Hz
```

20 Hz 版本时间覆盖更全（2022-07 ~ 2023-08），100 Hz 版本时域分辨率更高（2022-07 ~ 2023-06）。

### `data/dataset.py`
- `WindTurbineDataset`：按 `window_size` / `stride` 滑动加窗，每个窗口预测最后一个时间步的塔底响应（TMBNS / TMBEW / TMBTOR）；根据场景配置对失效 / 稀疏通道置零。稀疏屏蔽层级由 `scenario.level` 决定（三选一）：
  - `channel`：按通道随机屏蔽，保留 `observable_ratio` 比例的候选通道，被屏蔽通道整条时间轴不可见（测点未安装 / 长期失效）；
  - `timestep`：按时间点屏蔽，随机保留 `observable_ratio` 比例的时间步，被屏蔽时间步上所有候选通道同时缺失（采样丢包 / 间歇采集）；
  - `segment`：按时间段整段置空，候选通道在一段连续时间内全部缺失，缺失长度 `(1 - observable_ratio) × window_size`，起点随机（通信中断 / 停机；可选 `n_segments: K` 将窗口均分为 K 段后随机置空连续若干段）。

  mask 在数据集创建时按窗口固定（不随 epoch 变化），保证同一窗口在不同 epoch 中的输入条件一致、结果可复现。
- 输入维度程序化推导：输入 x 只含非目标通道（`input_dim = len(input_cols)`，本项目 34 = 桨叶 6 + 传动轴 1 + 机舱 7 + 转速 2 + 发电机 2 + 气象 6 + 控制/电网 10），目标信号只作标签，从结构上杜绝目标泄漏；`config.yaml` 中 `model.input_dim / output_dim` 设为 `null` 由数据自动填充，硬编码不一致时直接报错。
- `build_split_datasets`：**先按时间顺序切成 train / val / test 三段连续区间**（相邻区间之间丢弃 `split_gap` ≥ `window_size` 行作为隔离带），**再在各区间内独立滑窗**，不做窗口级随机打乱，返回 `(train, val, test, bounds)`。`stride(10) << window_size(100)` 时窗口高度重叠，若先滑窗再切分，训练段末尾窗口会与验证/测试段开头窗口共享原始数据，导致指标虚高。
- 缺失值编码（由 `preprocessing.missing_mode` / `missing_indicator` 控制）：`processed.csv` 已整体标准化，直接置零 = 填通道均值，模型无法区分“传感器缺失”与“读数恰好等于均值”。默认 `raw_zero`：缺失位置写入“原始域 0”对应的标准化值 `(0 - mean) / scale`（本数据 |哨兵值| ∈ [0.11, 176.5]，均远离均值），等价于断线测点在原始域读到 0；`missing_indicator: true` 时再为每个输入通道附加一条 0/1 观测指示通道（模型输入 34 → 68），使缺失在任何编码下都严格可区分。`norm_zero` 仅用于消融对照。

> ⚠️ 已知限制：`processed.csv` 并非单一连续时间序列，而是由**多个不相邻时间段**拼接而成
> （100 Hz 版 9 段，最大断点约 245 天，可用 `check_frame.py` 复现）。因此滑窗可能跨段
> 生成物理上不存在的样本，train / val / test 的按行切分也会把不同时间段分入不同集合。
> 该问题尚未在代码侧处理，解读跨段指标时需留意。

### `models/base.py`
四种数据驱动 backbone，统一接口 `forward(x) -> (batch, 3)`：
- `LSTMModel`：LSTM 取最后时间步输出。
- `GRUModel`：GRU 取最后时间步输出。
- `TCNModel`：因果空洞卷积时序网络。
- `TransformerModel`：Transformer 编码器 + 线性投影。
- `build_model(config)`：按 `config["model"]["backbone"]` 构建对应模型。

### `models/physics.py`
`physics_loss(pred, inputs, feature_names, target_names, phys_module)`：基于真实物理关系的软一致性残差（**物理约束，不是数值正则**），共三条：

| 约束 | 形式 | 说明 |
|------|------|------|
| 1 | `M_fa ↔ 推力代理` | 式 (1)(2) `M_base ≈ T·h`。**只作用于顺风向分量**；推力 T 不可直接测量，代理按 `RST2`（Q ∝ V² ∝ T）→ `ω²`（λ = ωR/V 恒定）→ `V²`（WSN / WS30 / WindEst）依次回退 |
| 2 | `M_ss` 不约束 | 侧向分量由横风湍流与塔架摆振主导（实测 R² 仅 0.22），施加推力先验属于错误先验 |
| 3 | `TMBTOR ↔ ω²` | 气动扭矩 Q ∝ ω²，与式 (3) 同源 |

实现要点：

- **必须用 `YP` 旋转**：`TMBNS / TMBEW` 是地理坐标系下的分量，随机舱偏航的机舱坐标系
  与之存在时变夹角。取窗口最后时刻的 `YP`（mod 360）得到 ψ，再算
  `M_fa = M_NS·cosψ + M_EW·sinψ`、`M_ss = −M_NS·sinψ + M_EW·cosψ`。
  逐通道施加约束是错误先验，逐通道 R² 也会随风向窗口漂移。
- **矢量运算在去标准化后的原始域进行**：需要 `scaler.npz` 中的 `TMBNS / TMBEW / YP` 参数，
  由 `load_scaler()` 在 `main.py` 读入并注入 `PINNWrapper → PhysicsConstraints`。
  σ_NS / σ_EW ≈ 0.64，若在标准化域直接合成幅值会扭曲两个轴的量纲权重。
- **残差取 1 − ρ²**：逐 batch 去均值后计算相关系数，取值 ∈ [0, 1]，量纲无关且对仿射变换
  不变，等价于逐 batch 拟合一次未知标定，因此不再需要可学习比例系数（`k_bending` /
  `k_torsion` 已被逐 batch 拟合的斜率完全吸收且不可辨识，已移除）。
- **场景感知**：驱动通道被 mask（去标准化后在 batch 内近似恒定）时该条约束自动跳过，
  故传感器失效越严重、可用约束越少，不会对目标施加错误先验。
- **优雅降级**：`YP` 缺失或被 mask → 跳过弯矩约束（**不**退化为逐通道约束）；
  缺 `scaler` → 打印告警并跳过。

已移除的旧项：机舱惯性 `|M| ↔ 加速度`、`|M| ↔ ω`、以及 batch 内相邻差分的“时序平滑”
（`DataLoader(shuffle=True)` 使相邻样本在时间上并不相邻，该实现实际惩罚的是批内输出方差）。

理论依据（简化假设 A1–A7、量纲一致性推导、误差范围：稳态 ±20%、极端工况 30%–40%）见 **[`physics.md`](physics.md)**。

### `models/pinn.py`
`PINNWrapper(config, scaler=None)`：包裹任意 backbone（通过 `build_model`），`compute_loss`
返回 `(data_loss, physics_loss)`，供训练时与数据损失联合优化。`scaler` 由 `main.py` 从
`processed` 目录的 `scaler.npz` 加载后注入，物理约束据此在原始域完成旋转与平方运算。

### `utils/metrics.py`
`compute_metrics(y_true, y_pred, target_names)`：计算 RMSE / MAE / MAPE / SMAPE / R² 的
总体指标（`overall_*`）与每个目标变量的分项（`{name}_*`，如 `TMBNS_r2`）。

指标在**标准化域**上计算。`TMBNS / TMBEW` 是同一矢量的两个分量，逐通道 R² 会随机舱方位
（风向）变化，建议配合物理域的矢量误差（幅值误差 Δ|M|、方向误差 Δφ）一起解读；
该矢量指标尚未实现，见 `research_direction.md` §9。

### `utils/visualize.py`
`plot_predictions` / `plot_scatter` / `plot_error_distribution`：分别绘制预测-真值对比、散点图、误差分布图并保存到输出目录。

### `main.py`
训练入口，支持单场景与多场景批量实验：
- 读取 config，`--scenario` 可传入一个或多个场景；
- `config.yaml` 中 `model.backbone` 和 `model.use_pinn` 支持单个值或列表，列表会自动展开为笛卡尔积组合；
- 同理 `seed` 支持单个值或列表：列表表示重复实验，每个 seed 会把所有 `(场景 × backbone × use_pinn)` 组合完整跑一遍，输出目录附加 `_seed{seed}` 后缀；命令行 `--seed 42 2026 916` 可覆盖配置；分析脚本按 `(scenario, backbone, use_pinn)` 聚合多次 seed 的结果（均值 ± 标准差）；
- 训练循环含早停（`early_stopping_patience`），以验证集损失选最优模型；
- 测试阶段评估并保存 `results.npz`、`metrics.yaml` 与 3 张可视化图到带时间戳的输出目录；
- 所有实验结果自动汇总到 `results/metrics_summary.csv`；
- `effective_physics_weight`：自适应物理权重（数据越充足物理约束自动退场）。

### `analyze_results.py`
结果分析入口，读取一个或多个已完成实验的输出目录，生成自包含 HTML 报表（默认写到该目录下的 `analysis_report.html`）：
- 实验总览与配置摘要（`physics_weight` / `adaptive_physics` 等）；
- 各场景、各 backbone、各 run 的指标总览与 run 索引（含可视化图链接）；
- 物理约束增益：PINN 与同 backbone baseline 配对比较（整体 + 分目标 TMBNS/TMBEW/TMBTOR，判定死区 ±`--tol`%）；
- 增益边界分析：增益随 `observable_ratio` 的变化曲线（内嵌 SVG）、增益转正的**临界稀疏度 r\***（相邻采样点线性插值，并标注不确定度）；
- 负迁移归因：驱动通道可用性、baseline 拟合程度、分目标方向不一致、自适应权重上限等证据链。

> ⚠️ 归因中的"驱动通道可用性"来自 `utils/analysis.py: CONSTRAINT_DRIVERS`，该映射仍引用
> 重构前的约束编号（C1 加速度 / C3 / C4），与 `models/physics.py` 现行约束不一致，
> 需同步后才能正确归因。

### `utils/analysis.py`
支撑上述分析的数据层：`collect_runs` 汇总 run（支持 `metrics.yaml` / `results.npz` 回退、多次重复实验聚合）、`build_gain_rows` 配对计算增益、`boundary_analysis` 求临界稀疏度、`negative_transfer` 负迁移归因、`build_report` 组装报表数据。
`CONSTRAINT_DRIVERS` 定义了"每条物理约束依赖哪些输入通道"，供 `constraint_activity` 计算各场景下约束的驱动可用性（需与 `models/physics.py` 保持同步）。

### `serve/`（模型服务）
把一份训练好的 `best_model.pt` 包装成对外推理 API，输入**原始工程量纲**的采样点，输出塔底响应 `TMBNS / TMBEW / TMBTOR`（同时给出标准化域与原始物理量纲两套结果）。

**加载哪一份权重由 `serve.yaml` 显式指定，服务不做任何自动选优**；`model` 段只写一条 `best_model.pt` 路径，其余（backbone / PINN / 场景 / seed）由该权重所在目录与实验 `config.yaml` 还原。参数优先级：命令行 > `serve.yaml` > 训练产物里的实验 `config.yaml` > 内置默认值：

```yaml
model:
  checkpoint: results/20260916064037_100hz_full/s2_severe_transformer_pinn_seed2026/best_model.pt
  config_file: null           # null = 从权重所在目录向上自动查找（.pt 被单独拷走时才需显式指定）

data:                         # 输入数据契约：不满足 = 输入分布与训练不一致
  processed_file: data/processed/B1_CL4_100/processed.csv   # 通道顺序 + scaler.npz 的来源
  input_channels: [PAB1, PAB2, ... Pwaste]                  # 34 个，顺序即模型输入顺序
  target_signals: [TMBNS, TMBEW, TMBTOR]
  units: raw                  # 只接受原始工程量纲，不接受标准化后的 z 值
  requirements:
    scenario: s2_severe       # 与权重所在 run 的场景不一致时启动报错
    observable_ratio: 0.4     # 典型可见通道比例（34 × 0.4 ≈ 14 路可见）
    tolerance: 1.2            # 可见通道数偏离 ±20% 即判为分布不一致
    blocked: []               # 训练时永久失效的通道（传了值也按缺失处理）
    on_violation: warn        # warn = 结果带 warnings；reject = 直接 400

preprocessing:                # 必须与训练一致，启动时与实验 config.yaml 核对
  strict: true                # 不一致时直接退出（false 只告警，结果不可与离线指标对照）
  window_size: 100
  missing_mode: raw_zero
  missing_indicator: true

server:
  host: 127.0.0.1
  port: 8000
  device: null                # null = 自动（cuda 可用则用 cuda）
```

启动时会做三件核对，任一不通过直接退出：**①** `preprocessing` 与实验 `config.yaml` 是否一致；**②** `data.input_channels` / `target_signals` 与 `processed.csv` 推导出的通道是否一致；**③** `data.requirements`（场景、可见比例、失效通道）与该 run 的训练场景是否一致。运行期每个请求再按 `requirements` 校验：可见通道数是否接近 `observable_ratio × 通道数`（偏离超 `tolerance` 倍即告警，或 `on_violation: reject` 时返回 400），请求里出现未知通道名也会告警。`/schema` 会原样返回这些要求。

命令行可临时覆盖配置而不动文件：`--checkpoint` / `--run-dir`（等价于该目录下的 `best_model.pt`）/ `--processed-file` / `--host` / `--port` / `--device`。

> 权重与当前代码版本不一致时的兼容处理：`PhysicsConstraints` 已移除 `k_bending` / `k_torsion`
> 两个不可辨识的可学习参数（`models/physics.py` 文件头第 4 条），旧权重里的这两个键会在加载时
> 被丢弃并打印告警；它们不参与推理期计算，不影响输出。其它任何键不匹配都会直接报错。

**只有没有 `serve.yaml` 时才退回按指标选优**（`serve/registry.py`，默认 `metric=overall_rmse`、`aggregate=mean`）：

1. 只考虑含 `best_model.pt` 的 run（未跑完的直接排除）；
2. 按 `(scenario, backbone, use_pinn)` 聚合多种子重复实验，用**指标均值**排序（`*_r2` 越大越好，其余越小越好）；`--aggregate single` 则直接按单次 run 排序；
3. 胜出组内再取指标最好的那一次 run 的权重 —— 部署必须落到具体的一份权重上。

> ⚠️ 跨场景比较不是同口径：`s0_full`（完整监测）天然比 `s2_severe`（严重稀疏）精度高，
> 默认取全局最优等价于"挑最好测点条件下的模型"。部署到真实稀疏场景时应显式
> `--scenario` 限定，否则服务的输入语义（哪些通道缺失）与训练场景不一致。
> `--list` 在候选跨多个场景时会打印该提示。

**输入编码必须与训练一致**，否则服务结果无法与离线指标对照（`serve/preprocess.py`）：通道顺序取 `processed.csv` 表头（去掉 `Time` 与目标信号），标准化用同一份 `scaler.npz`；缺失通道按 `missing_mode` 填哨兵值（默认 `raw_zero` = 原始域 0 的标准化值）；`missing_indicator=true` 时在末尾追加 34 条 0/1 观测指示通道。

**接口一览**（`serve/server.py`，启动后见 `/docs`）：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活状态、当前模型名、推理设备、流式会话数 |
| GET | `/model` | 当前服务的模型：run 名、场景配置、backbone / PINN / seed、选优指标与测试集指标 |
| GET | `/schema` | 输入通道清单、窗口长度、缺失编码语义、目标信号、**数据要求**（可见比例 / 容差 / 违规处理） |
| POST | `/predict` | 单窗口预测；带 `session_id` 时追加进滚动缓存，凑够一个窗口才出结果 |
| POST | `/predict/batch` | 多窗口批量预测，逐窗口返回，失败项放进 `errors` 不影响其余结果 |
| GET | `/sessions` | 流式会话的缓存长度 |
| DELETE | `/sessions/{id}` | 清空某个会话缓存（换风机 / 断流重连时调用） |

请求体中每个采样点是 `{通道名: 原始值}`；**通道缺省、值为 `null`、`NaN`、`inf` 均视为缺失**，服务会按训练时的语义编码，不会因为少传一个通道而报 400。

## 使用方式

```bash
# 环境准备（首次）
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 数据处理阶段（按 process.yaml 产出 20 Hz 与 100 Hz 两个版本的 processed 数据）
python preprocessing/prepare_data.py
python preprocessing/prepare_data.py --dataset b1_cl4_100   # 只处理指定版本

# 切换训练所用版本：只改 config.yaml 的 data.processed_file，无需重新处理数据

# 单次训练（backbone 与 PINN 开关由 config.yaml 的 model 段控制）
python main.py --scenario s0_full

# 多种子重复实验（列表中的多个值 = 重复多次运行）
# config.yaml 里 seed: [42, 2026, 916] 会把所有 (场景 × backbone × use_pinn) 组合各跑 3 遍，
# 输出目录附加 _seed{seed} 后缀；命令行 --seed 可覆盖配置
python main.py --scenario s0_full --seed 42 2026 916

# 多场景批量实验（config.yaml 中 backbone/use_pinn 可设为列表）
# 默认 backbone: ["lstm", "gru", "tcn", "transformer"], use_pinn: [false, true]
python main.py --scenario s0_full s1_medium s2_severe s3_meteo_failure s4_nacelle_accel_failure s5_shaft_failure

# 实验结果汇总
# results/{timestamp}_{name}/metrics_summary.csv

# 诊断 1：把 TMBNS/TMBEW 旋转到 fore-aft / side-side，对照各分量可预测性
python diagnose_ns_ew.py --only b1_cl4_100   # 只跑 100 Hz（约 33 万行，秒级）
python diagnose_ns_ew.py                     # 跑全部版本（含 20 Hz，523 万行，分块读取）

# 诊断 2：扫描旋转角确认 fore-aft / side-side 约定，并检查数据时间是否连续
python check_frame.py

# 结果分析（读取已完成实验目录，生成 HTML 报表）
python analyze_results.py --results_dir results/20260825104437_full_experiment

# 多个实验目录合并分析，并可调整增益判定死区与输出路径
python analyze_results.py --results_dir results/exp1 results/exp2 --tol 2 --out results/report.html

# ---- 推理服务 ----
pip install fastapi uvicorn                     # 仅服务需要，训练/分析不依赖

# 服务加载 serve.yaml 里指定的权重，不做选优；换模型改配置里的 model.run_dir
python -m serve.server --config serve.yaml      # 等价于 python -m serve.server
python -m serve.server --checkpoint /path/to/other.pt --port 9000   # 命令行临时覆盖

# 打印当前配置摘要（不启动服务）
python -m serve.server --list

# 端到端自检：取 processed.csv 末尾一个窗口反标准化回原始量纲，走完整链路（不起服务）
python -m serve.server --check

# 没有 serve.yaml 时才按指标选优（排行榜 / 指定场景或骨架）
python -m serve.server --results-dir results --list --top 10
python -m serve.server --results-dir results --scenario s2_severe --use-pinn true

# 冒烟测试（另开终端，服务已启动；地址从 serve.yaml 解析，--url 可覆盖）
python -m serve.client_example                        # 完整窗口一次性预测
python -m serve.client_example --ratio 0.4            # 按训练场景只可见 40% 通道
python -m serve.client_example --drop NAX1 NAX2 NAY1  # 模拟机舱加速度失效
python -m serve.client_example --stream               # 逐点推送，演示滚动缓存预热
```

### 调用示例

```bash
# 服务元信息与输入契约
curl http://127.0.0.1:8000/model
curl http://127.0.0.1:8000/schema

# 预测：samples 为按时间顺序排列的 window_size 个采样点，值是原始工程量纲
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "samples": [
    {"PAB1": 12.3, "PAB2": 11.8, "PAB3": 12.1, "NAX1": 0.04, "WSN": 8.7, "YP": 213.5},
    {"PAB1": 12.5, "PAB2": 12.0, "PAB3": 12.4, "NAX1": 0.05, "WSN": 8.9, "YP": 213.6}
  ]
}'
```

响应（节选）：

```json
{
  "ready": true,
  "status": "ok",
  "physical":  {"TMBNS": 1234.5, "TMBEW": -210.3, "TMBTOR": 88.1},
  "standardized": {"TMBNS": 0.31, "TMBEW": -0.42, "TMBTOR": 0.15},
  "model": {"run_name": "s2_severe_transformer_pinn_seed2026", "scenario": "s2_severe",
            "backbone": "transformer", "use_pinn": true, "seed": 2026},
  "diagnostics": {"window_size": 100, "n_observed": 30, "n_missing": 4,
                  "missing_channels": ["NAX1", "NAX2", "NAY1", "NAY2"]},
  "warnings": []
}
```

- `physical` 是原始工程量纲（可直接对接 SCADA / 报警阈值），`standardized` 可直接与该 run 的离线指标对照；
- 样本不足（≥1 个但不够一个窗口）时返回 `ready=false` + `status=warming_up` 与 `buffer_length`，继续推即可；
- `warnings` 会提示"输入可见通道数明显多于该稀疏场景训练时的典型值"这类分布不一致问题。

不作为 Web 服务时，`serve/predictor.py` 可直接当库用：

```python
from serve.config import ServeConfig
from serve.predictor import BestModelPredictor

p = BestModelPredictor.from_config(ServeConfig.from_yaml("serve.yaml"))
out = p.predict(samples)      # samples: window_size 个 {通道名: 原始值}
print(out["physical"])        # {"TMBNS": ..., "TMBEW": ..., "TMBTOR": ...}
```
