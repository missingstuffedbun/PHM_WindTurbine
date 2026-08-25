# PHM_WindTurbine

基于稀疏传感与物理约束的风机塔架结构健康监测项目。

## 文件结构

```
PHM_WindTurbine/
│
├── README.md
├── config.yaml            # 实验参数（数据预处理、场景名称列表、模型、训练、损失）
├── config/
│   └── scenarios.yaml     # 场景策略定义（observable_ratio / blocked_signals）
├── main.py                # 训练入口，支持单场景或多场景批量实验，接受 --config / --scenario
│
├── data/
│   ├── raw/               # 原始 Björkö 风机数据
│   ├── processed/         # 清洗 + 标准化后的 processed.csv 与 scaler.npz
│   └── dataset.py         # WindTurbineDataset（加窗 + 场景 mask）/ split_dataset
│
├── preprocessing/
│   └── prepare_data.py    # 数据清洗、归一化、加窗、场景 mask 构造
│
├── models/
│   ├── base.py            # 数据驱动 backbone：LSTMModel / TransformerModel / MLPModel / build_model
│   ├── pinn.py            # PINNWrapper：包装任意 backbone，compute_loss 返回 data_loss + physics_loss
│   └── physics.py         # 物理约束残差定义
│
├── utils/
│   ├── metrics.py         # RMSE / MAE / R² / SMAPE，分目标变量评价
│   └── visualize.py       # 预测对比图、散点图、误差分布图
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

### `data/dataset.py`
- `WindTurbineDataset`：按 `window_size` / `stride` 滑动加窗，每个窗口预测最后一个时间步的塔底响应（TMBNS / TMBEW / TMBTOR）；在 `__getitem__` 中根据场景配置动态对失效通道置零（稀疏/故障 mask）。
- `split_dataset`：按时间顺序前向切分训练 / 验证 / 测试集（默认 70% / 15% / 15%）。

### `models/base.py`
三种数据驱动 backbone，统一接口 `forward(x) -> (batch, 3)`：
- `LSTMModel`：LSTM 取最后时间步输出。
- `TransformerModel`：Transformer 编码器 + 线性投影。
- `MLPModel`：取窗口最后时间步的 MLP。
- `build_model(config)`：按 `config["model"]["backbone"]` 构建对应模型。

### `models/physics.py`
`physics_loss(pred, inputs, feature_names, target_names)`：基于真实物理关系的软一致性残差——塔底弯矩幅值与机舱加速度幅值一致、塔底扭矩与转子/发电机转速一致；被 mask 的驱动信号自动跳过。

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

## 使用方式

```bash
# 数据预处理
python preprocessing/prepare_data.py

# 单次训练（backbone 与 PINN 开关由 config.yaml 的 model 段控制）
python main.py --scenario s0_full

# 多场景批量实验（config.yaml 中 backbone/use_pinn 可设为列表）
python main.py --scenario s0_full s1_medium s2_severe s3_tower_failure s4_nacelle_failure s5_rotor_failure

# 实验结果汇总
# results/metrics_summary.csv
```
