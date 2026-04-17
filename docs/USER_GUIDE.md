# A股低频量化研究工作台 - 使用说明

> 项目地址：https://github.com/ingezhang2023/ashare-lowfreq-research

---

## 目录

1. [项目概述](#项目概述)
2. [快速开始](#快速开始)
3. [数据导入](#数据导入)
4. [三大模块使用](#三大模块使用)
5. [CLI命令参考](#cli命令参考)
6. [常见问题](#常见问题)

---

## 项目概述

### 能做什么

| 功能 | 说明 |
|------|------|
| ✅ 导入通达信历史数据 | 支持 .day 文件批量导入 |
| ✅ 因子研究与模型训练 | LightGBM walk-forward 训练 |
| ✅ 策略回测 | 计算收益、风险指标 |
| ✅ 模拟成交演练 | 生成盘前方案、模拟下单 |

### 不能做什么

| 功能 | 说明 |
|------|------|
| ❌ 实时行情获取 | 无 mootdx/akshare 实时接口 |
| ❌ 实盘交易 | 无券商 API 对接 |
| ❌ 日内交易 | 只支持日线频率 |

### 系统要求

| 项目 | 要求 |
|------|------|
| Python | 3.11+ |
| 操作系统 | Linux / macOS / Windows |
| 数据源 | 通达信 .day 文件 或 Tushare |

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/ingezhang2023/ashare-lowfreq-research.git
cd ashare-lowfreq-research
```

### 2. 安装依赖

```bash
pip install -e ".[dev]"
pip install tqdm pyarrow pandas lightgbm scikit-learn
```

### 3. 启动 Web 控制台

```bash
PYTHONPATH=src python3 -m ashare_backtest.web.app
```

访问：http://127.0.0.1:8888

### 4. 查看内置 Demo

项目自带 Demo 数据（30只股票，9380条日线），可直接体验。

---

## 数据导入

### 方式一：通达信导入（推荐）

#### 找到通达信数据目录

```
通达信安装目录/vipdoc/
  ├── sh/lday/  # 上海市场 .day 文件
  ├── sz/lday/  # 深圳市场 .day 文件
  └── bj/lday/  # 北京市场 .day 文件
```

#### 执行导入

```bash
# 单只股票
python scripts/import_single_day.py \
  --day-file /path/to/sh600519.day \
  --output-root storage

# 批量导入（使用项目内置脚本）
python -c "
from ashare_backtest.data.tdx_parser import TDXDayParser
from ashare_backtest.data.tdx_cleaner import TDXDataCleaner
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

parser = TDXDayParser()
cleaner = TDXDataCleaner()

# 处理所有 .day 文件
def process_file(f):
    df, _ = parser.parse_file(str(f))
    if df is None: return None
    code = parser.standardize_code(parser.extract_code_from_filename(f.name))
    df_clean, _ = cleaner.clean(df, code)
    df_clean['symbol'] = code
    df_clean = df_clean.rename(columns={'date': 'trade_date'})
    return df_clean[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']]

# 获取文件列表
sh_files = list(Path('/mnt/d/zstxd/vipdoc/sh/lday').glob('sh*.day'))
sz_files = list(Path('/mnt/d/zstxd/vipdoc/sz/lday').glob('sz*.day'))

# 并行处理
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_file, sh_files + sz_files))

# 合并保存
combined = pd.concat([r for r in results if r], ignore_index=True)
combined.to_parquet('storage/parquet/bars/daily.parquet', index=False)
print(f'导入完成: {len(combined)} 条')
"
```

#### 导入后处理

```bash
# 更新交易日历（含周末/节假日）
python scripts/update_calendar.py

# 更新股票列表
python scripts/update_instruments.py
```

### 方式二：Tushare 同步（需要 Token）

```bash
# 设置 Token
export TUSHARE_TOKEN=your_token_here

# 同步数据
ashare-backtest sync-tushare-sqlite

# 导入 Parquet
ashare-backtest import-sqlite --storage-root storage
```

### 数据目录结构

```
storage/
├── parquet/
│   ├── bars/
│   │   └── daily.parquet        # 日线数据
│   ├── calendar/
│   │   └── ashare_trading_calendar.parquet  # 交易日历
│   ├── instruments/
│   │   └── ashare_instruments.parquet  # 股票列表
│   └── universe/
│       └── memberships.parquet  # 股票池
├── source/
│   └── ashare_arena_sync.db     # SQLite 数据库
└── catalog.json                 # 数据索引
```

---

## 三大模块使用

### 模块一：数据看板

**路径**：http://127.0.0.1:8888/

**功能**：

| 功能 | 说明 |
|------|------|
| 核心指标 | 股票数、日线条数、日期范围 |
| 交易日历 | 可视化开盘/休市日 |
| 数据状态 | SQLite、Parquet 数据覆盖 |

**操作步骤**：

1. 打开首页
2. 切换工作区（native / qlib）
3. 查看数据覆盖状态

---

### 模块二：研究台

**路径**：http://127.0.0.1:8888/research

**功能**：完整研究流水线

```
配置文件 → 因子计算 → 模型训练 → 分数生成 → 分层分析 → 回测
```

**操作步骤**：

1. 切换工作区 → native
2. 选择策略配置（下拉框）
3. 编辑 TOML 参数（可选）
4. 点击"运行研究"
5. 查看执行日志
6. 查看产物（分数、指标）

**配置文件**：

| 文件 | 说明 |
|------|------|
| `configs/native/demo_strategy.toml` | Demo 策略（30只股票） |
| `configs/native/full_market_strategy.toml` | 全市场策略（9000+只） |

**关键参数**：

```toml
[storage]
root = "storage"              # 数据源

[factor_spec]
universe_name = "all_active"  # 股票池

[training]
test_start_month = "2025-01"  # 测试区间
test_end_month = "2026-04"
top_k = 10                    # 持仓数量
```

**产物位置**：

```
research/native/{strategy_name}/
  ├── factors/
  │   └── {strategy}.parquet    # 因子面板
  └── models/
      ├── {strategy}_scores.parquet  # 预测分数
      ├── {strategy}_metrics.json    # 训练指标
      └── {strategy}_layer.json      # 分层分析
```

---

### 模块三：回测控制台

**路径**：http://127.0.0.1:8888/backtest

**功能**：用已有分数快速回测

**操作步骤**：

1. 选择分数文件（下拉框）
2. 设置日期范围
3. 设置初始资金
4. 点击"开始回测"
5. 查看结果

**输出**：

```
results/native/{backtest_name}/
  ├── equity_curve.csv      # 资金曲线
  ├── trades.csv            # 交易记录
  └── summary.json          # 回测指标
```

**关键指标**：

| 指标 | 说明 |
|------|------|
| 总收益率 | 策略收益百分比 |
| 最大回撤 | 最大亏损幅度 |
| 夏普比率 | 风险调整收益 |
| 换手率 | 持仓调整频率 |

---

### 模块四：模拟成交台

**路径**：http://127.0.0.1:8888/simulation

**功能**：模拟实盘操作流程

**⚠️ 重要说明**：

| 项目 | 状态 |
|------|------|
| 数据来源 | 通达信历史数据（非实时） |
| 成交价格 | 历史开盘价（非实时价） |
| 资金 | 虚拟资金（非真钱） |
| 下单 | 模拟计算（非真实） |

**操作流程**：

```
创建账户 → 生成盘前计划 → 模拟下单 → 继续下一日
```

**步骤详解**：

1. **创建模拟账户**
   - 选择策略配置
   - 设置信号日期（如 2026-04-16）
   - 设置初始资金（如 100万）

2. **生成盘前计划**
   - 系统根据分数计算目标持仓
   - 生成买卖列表

3. **模拟下单**
   - 使用历史开盘价成交
   - 更新账户状态

4. **继续推进**
   - 生成下一交易日计划
   - 重复执行

**输出**：

```
simulation_accounts/{account_id}/
  ├── plan_{date}.json       # 盘前计划
  ├── execution_{date}.json  # 执行记录
  └── state_{date}.json      # 账户状态
```

---

## CLI命令参考

### 数据管理

```bash
# 列出股票池
ashare-backtest list-universes --storage-root storage

# Tushare 同步
ashare-backtest sync-tushare-sqlite --token YOUR_TOKEN

# SQLite 导入 Parquet
ashare-backtest import-sqlite --storage-root storage

# 同步基准指数
ashare-backtest sync-tushare-benchmark --symbol 000300.SH
```

### 研究训练

```bash
# 运行完整研究流水线
ashare-backtest run-research-config configs/native/demo_strategy.toml

# 训练 walk-forward 模型
ashare-backtest train-lgbm-walk-forward \
  --factor-panel-path research/factors/demo.parquet \
  --test-start-month 2025-01 \
  --test-end-month 2026-04

# 生成单日分数
ashare-backtest train-lgbm-walk-forward-as-of-date \
  --factor-panel-path research/factors/demo.parquet \
  --as-of-date 2026-04-16
```

### 回测运行

```bash
# 模型分数回测
ashare-backtest run-model-backtest \
  --scores-path research/native/demo/models/demo_scores.parquet \
  --storage-root storage \
  --start-date 2025-01-02 \
  --end-date 2026-04-16 \
  --top-k 10
```

### 模拟成交

```bash
# 生成盘前参考
ashare-backtest generate-premarket-reference \
  --scores-path research/native/demo/models/demo_scores.parquet \
  --trade-date 2026-04-17

# 生成策略状态
ashare-backtest generate-strategy-state \
  --config-path configs/native/demo_strategy.toml \
  --trade-date 2026-04-17
```

---

## 常见问题

### Q1：数据导入后交易日历全是开盘日？

**原因**：通达信 .day 文件只包含交易日，缺少周末/节假日

**解决**：
```bash
python scripts/update_calendar.py
```

### Q2：Dashboard 显示股票数不对？

**原因**：Catalog 未更新

**解决**：
```bash
python scripts/update_instruments.py
```

### Q3：模拟成交台提示"行情尚未落地"？

**原因**：执行日数据未导入

**解决**：
- 通达信只能导入历史数据（昨天及之前）
- 今日数据需要等待通达信更新后重新导入

### Q4：研究台执行失败？

**检查**：
```bash
# 1. 数据是否存在
ls storage/parquet/bars/daily.parquet

# 2. 配置文件是否正确
cat configs/native/demo_strategy.toml

# 3. 工作区是否匹配
# Web界面右上角切换 native/qlib
```

### Q5：全市场策略训练太慢？

**优化**：
- 减少 `test_start_month` 到 `test_end_month` 的月数
- 增加 `min_daily_amount` 过滤流动性差的股票
- 使用并行训练（配置 `parallel_workers`）

### Q6：能否实盘交易？

**答案**：不能

| 条件 | 当前状态 |
|------|----------|
| 券商 API | ❌ 未对接 |
| 实时行情 | ❌ 无接口 |
| 下单权限 | ❌ 无 |

**用途**：
- 回测验证策略
- 生成盘前方案（人工下单参考）
- 流程演练准备

---

## 数据流向图

```
通达信 .day 文件
    ↓ import-tdx-day
storage/parquet/bars/daily.parquet
    ↓ update_calendar / update_instruments
完整数据集（日历、股票池）
    ↓ 研究台
因子 → 模型 → 分数
    ↓ 回测控制台
回测结果（收益、风险）
    ↓ 模拟成交台
盘前方案（买卖列表）
    ↓ 人工执行
实盘下单（券商APP）
```

---

## 配置参数详解

### 策略配置结构

```toml
[storage]
root = "storage"              # 数据目录

[factor_spec]
id = "strategy_name"          # 策略标识
universe_name = "all_active"  # 股票池名称
start_date = "2020-04-16"     # 数据起始

[research_snapshot]
as_of_date = "2026-04-16"     # 当前快照日期

[factors]
output_path = "research/native/strategy/factors/factor.parquet"

[training]
label_column = "fwd_return_5"  # 预测目标（未来5日收益）
train_window_months = 12       # 训练窗口
validation_window_months = 1   # 验证窗口
test_start_month = "2025-01"   # 测试起始
test_end_month = "2026-04"     # 测试结束

[model_backtest]
output_dir = "results/native/strategy_backtest"
start_date = "2025-01-02"
end_date = "2026-04-16"
top_k = 10                     # 持仓数量
rebalance_every = 5            # 调仓间隔（交易日）
initial_cash = 1000000         # 初始资金
commission_rate = 0.0003       # 佣金率
stamp_tax_rate = 0.001         # 印花税率
slippage_rate = 0.0005         # 滑点率
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 10 | 持仓股票数量 |
| `rebalance_every` | 5 | 每5个交易日调仓 |
| `min_hold_bars` | 5 | 最少持有5个交易日 |
| `keep_buffer` | 2 | 持仓缓冲区（避免频繁换股） |
| `min_daily_amount` | 1000000 | 日成交额过滤（流动性） |

---

## 技术支持

- GitHub: https://github.com/ingezhang2023/ashare-lowfreq-research
- 文档: https://cyecho-io.github.io/ashare-lowfreq-research/
- 社区: https://discord.com/invite/clawd

---

## 版本历史

| 版本 | 日期 | 更新 |
|------|------|------|
| v0.1.0 | 2026-04 | 基础功能：回测、研究、模拟 |
| v0.2.0 | 2026-04 | 新增：通达信数据导入 |

---

> 最后更新：2026-04-17