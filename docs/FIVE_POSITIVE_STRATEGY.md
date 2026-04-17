# 五连阳选股策略使用说明

> 策略文件：`configs/native/five_positive_strategy.toml`

---

## 策略说明

### 选股条件

| 条件 | 说明 |
|------|------|
| **1. 五连阳形态** | 最近5日内没有阴线，最多1根十字星 |
| **2. 涨幅限制** | 每根K线涨幅在 0%-10% 之间 |
| **3. 无跳空** | 无向上跳空高开 |
| **4. 收盘递增** | 每日收盘价逐日递增 |
| **5. 均线多头** | MA5 > MA10 > MA20 > MA30 |
| **6. 站稳两线** | 股价在 MA5 和 MA10 之上 |

### 调仓规则

| 规则 | 设置 |
|------|------|
| **调仓频率** | 每日 |
| **持仓数量** | 最多1只股票 |
| **单只上限** | 100% |

### 风控设置

| 风控 | 设置 |
|------|------|
| **止损** | 5% |
| **止盈** | 9% |
| **最大回撤** | 7% |

---

## 使用方法

### 方法一：CLI 命令

```bash
# 步骤1：计算五连阳因子
cd ~/.openclaw/workspace/ashare-lowfreq-research
PYTHONPATH=src python3 -c "
from ashare_backtest.factors.five_positive_bars import build_factor_panel_for_backtest
result = build_factor_panel_for_backtest()
print(f'因子信号数: {result[\"signal_count\"]}')
"

# 步骤2：运行回测
PYTHONPATH=src python3 -m ashare_backtest.cli.main run-model-backtest \
  --scores-path research/native/five_positive/factors/five_positive_scores.parquet \
  --storage-root storage \
  --start-date 2026-01-02 \
  --end-date 2026-04-16 \
  --top-k 1 \
  --rebalance-every 1 \
  --min-daily-amount 50000000 \
  --initial-cash 1000000 \
  --output-dir results/native/five_positive_backtest
```

### 方法二：Web 控制台

1. **启动 Web 服务**
   ```bash
   PYTHONPATH=src python3 -m ashare_backtest.web.app
   ```

2. **访问回测控制台**
   - URL: http://127.0.0.1:8888/backtest
   - 切换工作区：native

3. **选择分数文件**
   - 下拉框选择：`five_positive_scores.parquet`

4. **设置参数**
   - 开始日期：2026-01-02
   - 结束日期：2026-04-16
   - 初始资金：1000000

5. **点击"开始回测"**

---

## 回测结果（示例）

### 测试区间：2026-01-02 ~ 2026-04-16

| 指标 | 值 |
|------|------|
| **总收益率** | 31.93% |
| **年化收益** | 160.27% |
| **最大回撤** | 27.42% |
| **夏普比率** | 1.55 |
| **胜率** | 48.48% |
| **盈亏比** | 1.78 |
| **交易次数** | 96次 |
| **换手率** | 52.72 倍/年 |

---

## 参数调整建议

### 调整持仓数量

```bash
# 持仓2只股票
--top-k 2

# 持仓5只股票
--top-k 5
```

### 调整调仓频率

```bash
# 每3天调仓
--rebalance-every 3

# 每5天调仓
--rebalance-every 5
```

### 调整流动性要求

```bash
# 日均成交额 >= 1亿
--min-daily-amount 100000000

# 日均成交额 >= 2亿
--min-daily-amount 200000000
```

---

## 策略文件结构

```
configs/native/five_positive_strategy.toml  # 策略配置
src/ashare_backtest/factors/
  └── five_positive_bars.py                 # 因子计算模块
research/native/five_positive/factors/
  └── five_positive_scores.parquet          # 分数文件（运行时生成）
results/native/five_positive_backtest/
  ├── summary.json                          # 回测摘要
  ├── equity_curve.csv                      # 资金曲线
  └── trades.csv                            # 交易记录
```

---

## 注意事项

### 1. 数据要求

- 需要 **最近5日** 的日线数据
- 需要 **30日** 均线数据（至少30个交易日历史）

### 2. 流动性风险

- 设置 `min_daily_amount` 避免流动性差的股票
- 默认：日均成交额 >= 5000万

### 3. 回测时间

- **每日调仓** 回测较慢
- 建议先用 **短期测试**（1-3个月）
- 再扩展到 **长期验证**（1年以上）

### 4. 信号频率

- 五连阳形态较稀有
- 每日信号数：0-200个不等
- 需要配合其他条件（均线、流动性）筛选

---

## 优化方向

| 方向 | 说明 |
|------|------|
| **增加持仓** | top_k=2~5，分散风险 |
| **延长持有** | rebalance_every=3~5，降低换手 |
| **增加止损** | 在回测后处理中添加止损逻辑 |
| **组合其他因子** | 与动量、估值因子组合 |

---

## 技术支持

- 策略文件：`configs/native/five_positive_strategy.toml`
- 因子模块：`src/ashare_backtest/factors/five_positive_bars.py`
- GitHub：https://github.com/ingezhang2023/ashare-lowfreq-research

---

> 最后更新：2026-04-17