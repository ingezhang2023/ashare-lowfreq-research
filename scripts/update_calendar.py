"""
更新交易日历脚本

功能：从日线数据生成完整交易日历（包含非开盘日）

使用：
    python scripts/update_calendar.py
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta
import json


def update_calendar():
    """从 bars 数据生成完整交易日历"""
    
    # 读取日线数据
    bars_path = Path('storage/parquet/bars/daily.parquet')
    if not bars_path.exists():
        print("❌ 日线数据不存在: storage/parquet/bars/daily.parquet")
        return
    
    bars = pd.read_parquet(bars_path)
    
    # 获取开盘日期
    trade_dates = set(bars['trade_date'].dt.strftime('%Y-%m-%d'))
    
    # 日期范围
    min_date = bars['trade_date'].min()
    max_date = bars['trade_date'].max()
    
    print(f"数据范围: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
    
    # 生成完整日历
    all_dates = []
    current = min_date
    while current <= max_date:
        is_open = current.strftime('%Y-%m-%d') in trade_dates
        all_dates.append({
            'trade_date': current,
            'is_open': is_open
        })
        current += timedelta(days=1)
    
    calendar = pd.DataFrame(all_dates)
    
    # 保存
    cal_path = Path('storage/parquet/calendar/ashare_trading_calendar.parquet')
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(cal_path, index=False)
    
    # 更新 catalog
    catalog_path = Path('storage/catalog.json')
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
        for ds in catalog.get('datasets', []):
            if ds['name'] == 'calendar.ashare':
                ds['rows'] = len(calendar)
                ds['min_date'] = calendar['trade_date'].min().strftime('%Y-%m-%d')
                ds['max_date'] = calendar['trade_date'].max().strftime('%Y-%m-%d')
        catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    
    # 输出结果
    print(f"✅ 交易日历已更新")
    print(f"   总天数: {len(calendar)}")
    print(f"   开盘日: {calendar['is_open'].sum()}")
    print(f"   非开盘日: {len(calendar) - calendar['is_open'].sum()}")
    print(f"   保存路径: {cal_path}")


if __name__ == '__main__':
    update_calendar()