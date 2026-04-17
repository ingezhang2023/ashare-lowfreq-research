"""
更新股票列表脚本

功能：从日线数据生成股票列表（instruments）

使用：
    python scripts/update_instruments.py
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def update_instruments():
    """从 bars 数据生成股票列表"""
    
    # 读取日线数据
    bars_path = Path('storage/parquet/bars/daily.parquet')
    if not bars_path.exists():
        print("❌ 日线数据不存在: storage/parquet/bars/daily.parquet")
        return
    
    bars = pd.read_parquet(bars_path)
    
    # 生成股票列表
    symbols = bars.groupby('symbol').agg({
        'trade_date': ['min', 'max', 'count']
    }).reset_index()
    symbols.columns = ['symbol', 'listing_date', 'delisting_date', 'bar_count']
    
    # 添加交易所信息
    symbols['exchange'] = symbols['symbol'].str.split('.').str[1]
    symbols['name'] = 'Stock_' + symbols['symbol']
    symbols['board'] = symbols['exchange'].map({'SH': '主板', 'SZ': '主板', 'BJ': '北交所'})
    symbols['industry'] = ''
    symbols['is_st'] = False
    symbols['is_active'] = True
    
    # 保存
    inst_path = Path('storage/parquet/instruments/ashare_instruments.parquet')
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    symbols.to_parquet(inst_path, index=False)
    
    # 更新 catalog
    catalog_path = Path('storage/catalog.json')
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
        for ds in catalog.get('datasets', []):
            if ds['name'] == 'instruments.ashare':
                ds['rows'] = len(symbols)
                ds['min_date'] = symbols['listing_date'].min().strftime('%Y-%m-%d')
                ds['max_date'] = symbols['delisting_date'].max().strftime('%Y-%m-%d')
        catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    
    # 输出结果
    print(f"✅ 股票列表已更新")
    print(f"   股票数量: {len(symbols)}")
    print(f"   上海: {len(symbols[symbols['exchange'] == 'SH'])}")
    print(f"   深圳: {len(symbols[symbols['exchange'] == 'SZ'])}")
    print(f"   北京: {len(symbols[symbols['exchange'] == 'BJ'])}")
    print(f"   保存路径: {inst_path}")


if __name__ == '__main__':
    update_instruments()