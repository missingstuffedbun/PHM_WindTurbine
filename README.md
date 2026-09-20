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
```
