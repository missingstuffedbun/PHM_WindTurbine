# PHM_WindTurbine

基于稀疏传感与物理约束的风机塔架结构健康监测项目。

## 文件结构

```
PHM_WindTurbine/
│
├── README.md
├── physics.md             # 物理约束说明：塔底弯矩-推力/倾覆力矩平衡式、简化假设、量纲推导、误差范围
├── config.yaml            # 实验参数（数据预处理、场景名称列表、模型、训练、损失）
├── config/
│   └── scenarios.yaml     # 场景策略定义（observable_ratio / level / blocked_signals）
├── main.py                # 训练入口，支持单场景或多场景批量实验，接受 --config / --scenario
├── analyze_results.py     # 结果分析入口：读取已完成实验目录，生成 HTML 报表（含物理约束增益边界分析）
│
├── data/
│   ├── raw/               # 原始 Björkö 风机数据
│   ├── processed/         # 清洗 + 标准化后的 processed.csv 与 scaler.npz
│   └── dataset.py         # WindTurbineDataset（加窗 + 场景 mask）/ build_split_datasets
│
├── preprocessing/
│   └── prepare_data.py    # 数据清洗、归一化、加窗、场景 mask 构造
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

### `preprocessing/prepare_data.py`
读取原始数据，按 metadata 过滤不可靠信号，清洗、标准化，输出 `data/processed/processed.csv` 与 `data/processed/scaler.npz`。
默认优先使用 `B1_CL4_20.csv`（20 Hz，72.75 小时，覆盖 2022-07 至 2023-08），因其数据量与时间覆盖均优于 100 Hz 文件。

### `data/dataset.py`
- `WindTurbineDataset`：按 `window_size` / `stride` 滑动加窗，每个窗口预测最后一个时间步的塔底响应（TMBNS / TMBEW / TMBTOR）；根据场景配置对失效 / 稀疏通道置零。稀疏屏蔽层级由 `scenario.level` 决定（三选一）：
  - `channel`：按通道随机屏蔽，保留 `observable_ratio` 比例的候选通道，被屏蔽通道整条时间轴不可见（测点未安装 / 长期失效）；
  - `timestep`：按时间点屏蔽，随机保留 `observable_ratio` 比例的时间步，被屏蔽时间步上所有候选通道同时缺失（采样丢包 / 间歇采集）；
  - `segment`：按时间段整段置空，候选通道在一段连续时间内全部缺失，缺失长度 `(1 - observable_ratio) × window_size`，起点随机（通信中断 / 停机；可选 `n_segments: K` 将窗口均分为 K 段后随机置空连续若干段）。

  mask 在数据集创建时按窗口固定（不随 epoch 变化），保证同一窗口在不同 epoch 中的输入条件一致、结果可复现。
- 输入维度程序化推导：输入 x 只含非目标通道（`input_dim = len(input_cols)`，本项目 34 = 桨叶 6 + 传动轴 1 + 机舱 7 + 转速 2 + 发电机 2 + 气象 6 + 控制/电网 10），目标信号只作标签，从结构上杜绝目标泄漏；`config.yaml` 中 `model.input_dim / output_dim` 设为 `null` 由数据自动填充，硬编码不一致时直接报错。
- `build_split_datasets`：**先按时间顺序切成 train / val / test 三段连续区间**（相邻区间之间丢弃 `split_gap` ≥ `window_size` 行作为隔离带），**再在各区间内独立滑窗**，不做窗口级随机打乱，返回 `(train, val, test, bounds)`。`stride(10) << window_size(100)` 时窗口高度重叠，若先滑窗再切分，训练段末尾窗口会与验证/测试段开头窗口共享原始数据，导致指标虚高。
- 缺失值编码（由 `preprocessing.missing_mode` / `missing_indicator` 控制）：`processed.csv` 已整体标准化，直接置零 = 填通道均值，模型无法区分“传感器缺失”与“读数恰好等于均值”。默认 `raw_zero`：缺失位置写入“原始域 0”对应的标准化值 `(0 - mean) / scale`（本数据 |哨兵值| ∈ [0.11, 176.5]，均远离均值），等价于断线测点在原始域读到 0；`missing_indicator: true` 时再为每个输入通道附加一条 0/1 观测指示通道（模型输入 34 → 68），使缺失在任何编码下都严格可区分。`norm_zero` 仅用于消融对照。

### `models/base.py`
四种数据驱动 backbone，统一接口 `forward(x) -> (batch, 3)`：
- `LSTMModel`：LSTM 取最后时间步输出。
- `GRUModel`：GRU 取最后时间步输出。
- `TCNModel`：因果空洞卷积时序网络。
- `TransformerModel`：Transformer 编码器 + 线性投影。
- `build_model(config)`：按 `config["model"]["backbone"]` 构建对应模型。

### `models/physics.py`
`physics_loss(pred, inputs, feature_names, target_names)`：基于真实物理关系的软一致性残差（**物理约束，不是数值正则**）——塔底弯矩幅值与机舱加速度幅值一致、塔底扭矩与转子/发电机转速一致；被 mask 的驱动信号自动跳过。

物理约束的理论依据为塔底弯矩与推力 / 倾覆力矩的平衡式 `M_base ≈ T·h`（`T = ½ρA·C_T·V²`，风速不可用时由 `T = Q·ω·C_T/(V·C_P)` 反推）。
其简化假设、量纲一致性推导与误差范围（稳态 ±20%，极端工况 30%–40%）见 **[`physics.md`](physics.md)**。

### `models/pinn.py`
`PINNWrapper`：包裹任意 backbone（通过 `build_model`），`compute_loss` 返回 `(data_loss, physics_loss)`，供训练时与数据损失联合优化。

### `utils/metrics.py`
`compute_metrics(y_true, y_pred, target_names)`：计算 RMSE / MAE / MAPE / SMAPE / R² 的总体指标及每个目标变量分项。

### `utils/visualize.py`
`plot_predictions` / `plot_scatter` / `plot_error_distribution`：分别绘制预测-真值对比、散点图、误差分布图并保存到输出目录。

### `main.py`
训练入口，支持单场景与多场景批量实验：
- 读取 config，`--scenario` 可传入一个或多个场景；
- `config.yaml` 中 `model.backbone` 和 `model.use_pinn` 支持单个值或列表，列表会自动展开为笛卡尔积组合；
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

### `utils/analysis.py`
支撑上述分析的数据层：`collect_runs` 汇总 run（支持 `metrics.yaml` / `results.npz` 回退、多次重复实验聚合）、`build_gain_rows` 配对计算增益、`boundary_analysis` 求临界稀疏度、`negative_transfer` 负迁移归因、`build_report` 组装报表数据。

## 使用方式

```bash
# 数据预处理
python preprocessing/prepare_data.py

# 单次训练（backbone 与 PINN 开关由 config.yaml 的 model 段控制）
python main.py --scenario s0_full

# 多场景批量实验（config.yaml 中 backbone/use_pinn 可设为列表）
# 默认 backbone: ["lstm", "gru", "tcn", "transformer"], use_pinn: [false, true]
python main.py --scenario s0_full s1_medium s2_severe s3_meteo_failure s4_nacelle_accel_failure s5_shaft_failure

# 实验结果汇总
# results/{timestamp}_{name}/metrics_summary.csv

# 结果分析（读取已完成实验目录，生成 HTML 报表）
python analyze_results.py --results_dir results/20260825104437_full_experiment

# 多个实验目录合并分析，并可调整增益判定死区与输出路径
python analyze_results.py --results_dir results/exp1 results/exp2 --tol 2 --out results/report.html
```
