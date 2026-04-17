# 策略实验记录

## 目录结构

```
experiments/
├── README.md                   # 本文件
├── hs300_topk_comparison/      # top_k对比实验
├── hs300_rebalance_comparison/ # 调仓频率对比
├── factor_combinations/        # 因子组合实验
└── time_range_validation/      # 时间区间验证
```

## 实验记录模板

每个实验目录包含：

- `config.toml`      # 实验配置
- `results.json`     # 回测结果
- `notes.md`         # 实验笔记
- `comparison.md`    # 对比分析

## 当前计划

1. top_k对比：5, 10, 15, 20
2. rebalance对比：3, 5, 10, 15
3. 时间区间验证：不同训练/测试区间
