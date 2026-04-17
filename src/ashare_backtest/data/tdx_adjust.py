"""
前复权处理模块

复权公式：
- 复权后价格 = 原始价格 × 复权因子
- 复权后成交量 = 原始成交量 ÷ 复权因子
- 复权后成交额 = 原始成交额 ÷ 复权因子
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class TDXAdjuster:
    """前复权处理器"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.warnings: List[Dict] = []
        self.errors: List[Dict] = []
    
    def load_adj_factor(
        self, 
        adj_factor_dir: str, 
        code: str
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        加载复权因子
        
        Args:
            adj_factor_dir: 复权因子目录
            code: 股票代码（如 000001.SZ）
            
        Returns:
            (adj_factor_df, metadata) - 复权因子 DataFrame 和元信息
        """
        adj_dir = Path(adj_factor_dir)
        
        # 尝试两种文件格式
        possible_files = [
            adj_dir / f'{code}.parquet',
            adj_dir / f'{code}.csv',
            adj_dir / f'{code.replace(".", "")}.parquet',
            adj_dir / f'{code.replace(".", "")}.csv',
        ]
        
        adj_file = None
        for f in possible_files:
            if f.exists():
                adj_file = f
                break
        
        if adj_file is None:
            return None, {
                'error': f'复权因子文件不存在: {code}',
                'searched': [str(f) for f in possible_files]
            }
        
        try:
            if adj_file.suffix == '.parquet':
                adj_df = pd.read_parquet(adj_file)
            else:
                adj_df = pd.read_csv(adj_file)
            
            # 验证必需列
            if 'date' not in adj_df.columns:
                return None, {'error': '复权因子文件缺少 date 列'}
            if 'adj_factor' not in adj_df.columns:
                return None, {'error': '复权因子文件缺少 adj_factor 列'}
            
            # 确保日期为 datetime 类型
            adj_df['date'] = pd.to_datetime(adj_df['date'])
            
            # 排序
            adj_df = adj_df.sort_values('date', ascending=True).reset_index(drop=True)
            
            metadata = {
                'filepath': str(adj_file),
                'records': len(adj_df),
                'date_range': f'{adj_df["date"].min()} ~ {adj_df["date"].max()}'
            }
            
            return adj_df, metadata
            
        except Exception as e:
            return None, {'error': f'读取复权因子文件失败: {type(e).__name__}: {e}'}
    
    def adjust(
        self, 
        df: pd.DataFrame, 
        adj_factor_df: pd.DataFrame,
        code: str = ''
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        应用前复权
        
        Args:
            df: 原始数据 DataFrame
            adj_factor_df: 复权因子 DataFrame
            code: 股票代码（用于日志）
            
        Returns:
            (adjusted_df, stats) - 复权后的 DataFrame 和统计信息
        """
        stats = {
            'original_records': len(df),
            'matched_factors': 0,
            'ffilled_factors': 0,
            'default_factors': 0,
            'warnings': []
        }
        
        # 合并复权因子
        df_merged = df.merge(adj_factor_df, on='date', how='left')
        
        # 向前填充复权因子
        df_merged['adj_factor'] = df_merged['adj_factor'].ffill()
        
        # 统计填充情况
        matched = df_merged[df_merged['adj_factor'].notna() & 
                          (df_merged['date'] >= adj_factor_df['date'].min())]
        stats['matched_factors'] = len(matched)
        
        # 未匹配到复权因子的日期（在复权因子日期范围之前的）
        earliest_dates = df_merged[df_merged['adj_factor'].isna()]
        if len(earliest_dates) > 0:
            # 使用最早的复权因子或默认值 1.0
            first_adj_factor = adj_factor_df.iloc[0]['adj_factor']
            df_merged.loc[earliest_dates.index, 'adj_factor'] = first_adj_factor
            
            stats['default_factors'] = len(earliest_dates)
            
            if self.verbose:
                for _, row in earliest_dates.iterrows():
                    self.warnings.append({
                        'code': code,
                        'date': str(row['date']),
                        'warning': f'复权因子缺失，使用默认值 {first_adj_factor}'
                    })
        
        # 确保复权因子不为 NaN
        df_merged['adj_factor'] = df_merged['adj_factor'].fillna(1.0)
        
        # 应用复权公式
        df_merged['open'] = df_merged['open'] * df_merged['adj_factor']
        df_merged['high'] = df_merged['high'] * df_merged['adj_factor']
        df_merged['low'] = df_merged['low'] * df_merged['adj_factor']
        df_merged['close'] = df_merged['close'] * df_merged['adj_factor']
        df_merged['volume'] = df_merged['volume'] / df_merged['adj_factor']
        df_merged['amount'] = df_merged['amount'] / df_merged['adj_factor']
        
        # 删除辅助列
        df_adjusted = df_merged.drop(columns=['adj_factor'])
        
        # 确保数据类型正确
        df_adjusted['volume'] = df_adjusted['volume'].astype(int)
        
        stats['final_records'] = len(df_adjusted)
        
        return df_adjusted, stats
    
    def get_warnings(self) -> List[Dict]:
        """获取警告"""
        return self.warnings
    
    def get_errors(self) -> List[Dict]:
        """获取错误"""
        return self.errors
    
    def clear_logs(self):
        """清除日志"""
        self.warnings = []
        self.errors = []