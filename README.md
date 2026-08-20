# PHM_WindTurbine

WindTurbine_Sparse_SHM/
│
├── README.md
├── config.yaml
├── main.py                 # 训练入口
│
├── data/
│   ├── raw/                # 原始 Björkö 数据
│   ├── processed/          # 清洗、归一化、加窗后的数据
│   └── dataset.py          # PyTorch Dataset / DataLoader
│
├── preprocessing/
│   ├── clean.py            # 数据清洗
│   ├── normalize.py        # 归一化
│   ├── window.py           # 时间窗口构造
│   └── mask.py             # 稀疏/失效场景构造
│
├── models/
│   ├── base.py             # 数据驱动基线（LSTM / Transformer / GNN 等）
│   ├── pinn.py             # 物理约束模型
│   └── physics.py          # 物理约束定义
│
├── utils/
│   ├── metrics.py          # 评价指标
│   └── visualize.py        # 结果可视化
│
└── results/
    ├── models/             # 保存模型权重
    ├── metrics/            # 保存评价结果
    └── figures/            # 保存可视化图片

