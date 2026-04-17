"""
五连阳选股因子

策略条件：
1. 最近5日内没有阴线（最多1根十字星）
2. 每根K线涨幅在 0%-10% 之间
3. 无向上跳空高开
4. 每日收盘价逐日递增
5. 均线多头排列（MA5 > MA10 > MA20 > MA30）
6. 股价在 MA5 和 MA10 之上
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_five_positive_bars_factor(
    bars_path: str | Path,
    output_path: str | Path | None = None,
    lookback_days: int = 5,
    min_positive_count: int = 4,
    max_doji_count: int = 1,
    min_return_pct: float = 0.0,
    max_return_pct: float = 10.0,
    ma_periods: list[int] = [5, 10, 20, 30],
) -> pd.DataFrame:
    """
    计算五连阳因子
    
    Args:
        bars_path: 日线数据路径
        output_path: 输出路径（可选）
        lookback_days: 回看天数（默认5）
        min_positive_count: 最少阳线数（默认4）
        max_doji_count: 最大十字星数（默认1）
        min_return_pct: 最小涨幅百分比（默认0）
        max_return_pct: 最大涨幅百分比（默认10）
        ma_periods: 均线周期列表
    
    Returns:
        因子DataFrame，包含 trade_date, symbol, score 列
    """
    # 读取日线数据
    bars = pd.read_parquet(bars_path)
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    bars = bars.sort_values(['symbol', 'trade_date'])
    
    # 计算基础指标
    bars['return_pct'] = (bars['close'] - bars['open']) / bars['open'] * 100
    bars['is_positive'] = bars['close'] > bars['open']  # 阳线
    bars['is_doji'] = np.abs(bars['close'] - bars['open']) / bars['open'] < 0.001  # 十字星
    bars['is_negative'] = bars['close'] < bars['open']  # 阴线
    
    # 计算均线
    for period in ma_periods:
        bars[f'ma{period}'] = bars.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=period, min_periods=period).mean()
        )
    
    # 计算跳空（今日开盘 > 昨日最高）
    bars['prev_high'] = bars.groupby('symbol')['high'].shift(1)
    bars['gap_up'] = bars['open'] > bars['prev_high']
    
    # 计算收盘递增
    bars['prev_close'] = bars.groupby('symbol')['close'].shift(1)
    bars['close_increasing'] = bars['close'] > bars['prev_close']
    
    # 滚动计算条件（最近lookback_days天）
    def check_five_positive_pattern(group):
        """
        检查五连阳形态
        """
        results = []
        
        for i in range(len(group)):
            if i < lookback_days - 1:
                results.append(False)
                continue
            
            # 获取最近lookback_days天的数据
            window = group.iloc[i - lookback_days + 1:i + 1]
            
            # 条件1: 最近5日内没有阴线（最多1根十字星）
            negative_count = window['is_negative'].sum()
            if negative_count > 0:
                results.append(False)
                continue
            
            positive_count = window['is_positive'].sum()
            doji_count = window['is_doji'].sum()
            if positive_count + doji_count < min_positive_count:
                results.append(False)
                continue
            
            if doji_count > max_doji_count:
                results.append(False)
                continue
            
            # 条件2: 每根K线涨幅在 0%-10% 之间
            if not ((window['return_pct'] >= min_return_pct) & 
                    (window['return_pct'] <= max_return_pct)).all():
                results.append(False)
                continue
            
            # 条件3: 无向上跳空高开
            if window['gap_up'].any():
                results.append(False)
                continue
            
            # 条件4: 每日收盘价逐日递增
            if not window['close_increasing'].iloc[1:].all():  # 从第2天开始检查
                results.append(False)
                continue
            
            # 条件5: 均线多头排列
            current = group.iloc[i]
            if not (current['ma5'] > current['ma10'] and 
                    current['ma10'] > current['ma20'] and 
                    current['ma20'] > current['ma30']):
                results.append(False)
                continue
            
            # 条件6: 股价在 MA5 和 MA10 之上
            if not (current['close'] > current['ma5'] and 
                    current['close'] > current['ma10']):
                results.append(False)
                continue
            
            # 所有条件满足
            results.append(True)
        
        return results
    
    # 应用检查函数
    bars['five_positive_signal'] = bars.groupby('symbol').apply(
        check_five_positive_pattern
    ).reset_index(level=0, drop=True)
    
    # 创建因子面板
    factor_panel = bars[bars['five_positive_signal']].copy()
    
    # 计算得分（满足条件时给1分）
    factor_panel['score'] = 1.0
    factor_panel['factor_name'] = 'five_positive_bars'
    
    # 选择输出列
    output_cols = [
        'trade_date', 'symbol', 'score', 'factor_name',
        'close', 'return_pct', 'ma5', 'ma10', 'ma20', 'ma30',
        'open', 'high', 'low', 'volume', 'amount'
    ]
    
    factor_panel = factor_panel[output_cols].copy()
    
    # 保存（如果指定输出路径）
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        factor_panel.to_parquet(output_path, index=False)
        print(f'✅ 因子面板已保存: {output_path}')
        print(f'   信号数: {len(factor_panel)}')
        print(f'   股票数: {factor_panel["symbol"].nunique()}')
        print(f'   日期范围: {factor_panel["trade_date"].min()} ~ {factor_panel["trade_date"].max()}')
    
    return factor_panel


def build_factor_panel_for_backtest(
    bars_path: str | Path = 'storage/parquet/bars/daily.parquet',
    output_dir: str | Path = 'research/native/five_positive',
) -> dict:
    """
    构建因子面板用于回测
    
    Returns:
        包含因子路径等信息的字典
    """
    output_path = Path(output_dir) / 'factors' / 'five_positive.parquet'
    
    factor_panel = calculate_five_positive_bars_factor(
        bars_path=bars_path,
        output_path=output_path,
    )
    
    return {
        'factor_path': str(output_path),
        'signal_count': len(factor_panel),
        'symbol_count': factor_panel['symbol'].nunique(),
        'start_date': str(factor_panel['trade_date'].min()),
        'end_date': str(factor_panel['trade_date'].max()),
    }


if __name__ == '__main__':
    # 测试运行
    result = build_factor_panel_for_backtest()
    print()
    print('因子构建结果:')
    for key, value in result.items():
        print(f'  {key}: {value}')