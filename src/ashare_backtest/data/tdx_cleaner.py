"""
通达信数据清洗模块

实现严格的数据清洗规则：
1. 排序与去重
2. 价格完整性检查
3. 价格逻辑一致性检查
4. 涨跌幅异常检测（仅警告）
5. 缺失值处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class TDXDataCleaner:
    """通达信数据清洗器"""
    
    # 涨跌幅警告阈值
    PRICE_CHANGE_WARNING_THRESHOLD = 15.0  # 15%
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.cleaning_stats = {
            'original_count': 0,
            'final_count': 0,
            'removed_count': 0,
            'price_zero_or_negative': 0,
            'high_low_logic_error': 0,
            'missing_values': 0,
            'duplicate_dates': 0,
            'price_change_warnings': 0
        }
        self.warnings: List[Dict] = []
    
    def clean(self, df: pd.DataFrame, code: str = '') -> Tuple[pd.DataFrame, Dict]:
        """
        清洗数据
        
        Args:
            df: 原始 DataFrame，必须包含 date, open, high, low, close, volume, amount 列
            code: 股票代码（用于日志）
            
        Returns:
            (cleaned_df, stats) - 清洗后的 DataFrame 和统计信息
        """
        self.cleaning_stats = {
            'original_count': len(df),
            'final_count': 0,
            'removed_count': 0,
            'price_zero_or_negative': 0,
            'high_low_logic_error': 0,
            'missing_values': 0,
            'duplicate_dates': 0,
            'price_change_warnings': 0
        }
        
        if df.empty:
            return df, self.cleaning_stats
        
        # 1. 排序与去重
        df = self._sort_and_dedupe(df, code)
        
        # 2. 价格完整性检查
        df = self._check_price_completeness(df, code)
        
        # 3. 价格逻辑一致性检查
        df = self._check_price_logic(df, code)
        
        # 4. 涨跌幅异常检测
        self._check_price_change(df, code)
        
        # 5. 缺失值处理
        df = self._handle_missing_values(df, code)
        
        # 最终统计
        self.cleaning_stats['final_count'] = len(df)
        self.cleaning_stats['removed_count'] = self.cleaning_stats['original_count'] - len(df)
        
        return df, self.cleaning_stats
    
    def _sort_and_dedupe(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """排序与去重"""
        # 按日期升序排序
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        # 删除重复日期（保留第一条）
        duplicates = df[df.duplicated('date', keep='first')]
        if len(duplicates) > 0:
            self.cleaning_stats['duplicate_dates'] = len(duplicates)
            if self.verbose:
                for _, row in duplicates.iterrows():
                    self.warnings.append({
                        'code': code,
                        'date': str(row['date']),
                        'warning': f'重复日期，已删除'
                    })
            df = df.drop_duplicates('date', keep='first').reset_index(drop=True)
        
        return df
    
    def _check_price_completeness(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        价格完整性检查
        
        条件：
        - 开盘价 > 0
        - 最高价 > 0
        - 最低价 > 0
        - 收盘价 > 0
        - 成交量 >= 0
        - 成交额 >= 0
        """
        original_len = len(df)
        
        # 检查价格 <= 0
        invalid_mask = (
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0) |
            (df['volume'] < 0) |
            (df['amount'] < 0)
        )
        
        invalid_rows = df[invalid_mask]
        if len(invalid_rows) > 0:
            self.cleaning_stats['price_zero_or_negative'] = len(invalid_rows)
            if self.verbose:
                for _, row in invalid_rows.iterrows():
                    self.warnings.append({
                        'code': code,
                        'date': str(row['date']),
                        'warning': f'价格<=0删除: open={row["open"]}, high={row["high"]}, low={row["low"]}, close={row["close"]}'
                    })
        
        df = df[~invalid_mask].reset_index(drop=True)
        
        return df
    
    def _check_price_logic(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        价格逻辑一致性检查
        
        条件：
        - 最高价 >= 开盘价
        - 最高价 >= 收盘价
        - 最低价 <= 开盘价
        - 最低价 <= 收盘价
        """
        original_len = len(df)
        
        # 检查逻辑一致性
        invalid_mask = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        invalid_rows = df[invalid_mask]
        if len(invalid_rows) > 0:
            self.cleaning_stats['high_low_logic_error'] = len(invalid_rows)
            if self.verbose:
                for _, row in invalid_rows.iterrows():
                    self.warnings.append({
                        'code': code,
                        'date': str(row['date']),
                        'warning': f'逻辑不一致删除: high={row["high"]}, low={row["low"]}, open={row["open"]}, close={row["close"]}'
                    })
        
        df = df[~invalid_mask].reset_index(drop=True)
        
        return df
    
    def _check_price_change(self, df: pd.DataFrame, code: str) -> None:
        """
        涨跌幅异常检测（仅警告，不删除）
        
        计算当日涨跌幅，如果绝对值超过 15%，记录警告
        """
        if len(df) < 2:
            return
        
        # 计算涨跌幅
        df_temp = df.copy()
        df_temp['prev_close'] = df_temp['close'].shift(1)
        df_temp['price_change'] = (
            (df_temp['close'] - df_temp['prev_close']) / df_temp['prev_close'] * 100
        )
        
        # 检测异常涨跌幅
        abnormal = df_temp[abs(df_temp['price_change']) > self.PRICE_CHANGE_WARNING_THRESHOLD]
        
        if len(abnormal) > 0:
            self.cleaning_stats['price_change_warnings'] = len(abnormal)
            for _, row in abnormal.iterrows():
                self.warnings.append({
                    'code': code,
                    'date': str(row['date']),
                    'warning': f'涨跌幅{row["price_change"]:.1f}%超过15%'
                })
    
    def _handle_missing_values(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        缺失值处理
        
        规则：
        - 价格字段（开/高/低/收）有 NaN → 删除整行
        - 成交量或成交额为 NaN → 删除整行
        - 不进行任何填充
        """
        original_len = len(df)
        
        # 检查 NaN
        invalid_mask = (
            df['open'].isna() |
            df['high'].isna() |
            df['low'].isna() |
            df['close'].isna() |
            df['volume'].isna() |
            df['amount'].isna()
        )
        
        invalid_rows = df[invalid_mask]
        if len(invalid_rows) > 0:
            self.cleaning_stats['missing_values'] = len(invalid_rows)
            if self.verbose:
                for _, row in invalid_rows.iterrows():
                    self.warnings.append({
                        'code': code,
                        'date': str(row['date']) if pd.notna(row['date']) else 'NaN',
                        'warning': '缺失值删除'
                    })
        
        df = df[~invalid_mask].reset_index(drop=True)
        
        return df
    
    def get_stats(self) -> Dict:
        """获取清洗统计"""
        return self.cleaning_stats
    
    def get_warnings(self) -> List[Dict]:
        """获取清洗警告"""
        return self.warnings
    
    def clear_logs(self):
        """清除日志"""
        self.warnings = []