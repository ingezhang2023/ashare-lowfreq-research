"""
通达信 .day 文件解析器

.day 文件格式（每条记录 32 字节）：
- 日期: 4 字节 (整数，YYYYMMDD 格式)
- 开盘价: 4 字节 (整数，实际价格 = 原始值 / 100)
- 最高价: 4 字节
- 最低价: 4 字节
- 收盘价: 4 字节
- 成交额: 4 字节 (浮点数)
- 成交量: 4 字节 (整数)
- 保留: 4 字节
"""

import struct
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


class TDXDayParser:
    """通达信 .day 文件解析器"""
    
    # 每条记录的字节数
    RECORD_SIZE = 32
    
    # 记录格式：日期、开、高、低、收、成交量、成交额、保留
    # 使用 struct 解析：< 表示小端，I 表示 4 字节无符号整数
    # 注意：成交量在前，成交额在后（都是整数）
    RECORD_FORMAT = '<IIIIIIII'
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.warnings: List[Dict] = []
        self.errors: List[Dict] = []
    
    def parse_file(self, filepath: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        解析单个 .day 文件
        
        Args:
            filepath: .day 文件路径
            
        Returns:
            (DataFrame, metadata) - DataFrame 包含解析后的数据，metadata 包含解析信息
            如果解析失败，DataFrame 为 None
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return None, {'error': f'文件不存在: {filepath}'}
        
        if filepath.stat().st_size == 0:
            return None, {'error': '文件为空'}
        
        # 检查文件大小是否为 32 字节的整数倍
        file_size = filepath.stat().st_size
        if file_size % self.RECORD_SIZE != 0:
            return None, {'error': f'文件大小 {file_size} 不是 32 字节的整数倍'}
        
        records = []
        parse_errors = []
        
        try:
            with open(filepath, 'rb') as f:
                record_num = 0
                while True:
                    data = f.read(self.RECORD_SIZE)
                    if not data:
                        break
                    
                    if len(data) < self.RECORD_SIZE:
                        parse_errors.append({
                            'record': record_num,
                            'error': '记录不完整（少于 32 字节）'
                        })
                        break
                    
                    try:
                        # 解析二进制数据
                        # 格式: 日期、开、高、低、收、成交量、成交额、保留
                        date_int, open_raw, high_raw, low_raw, close_raw, volume, amount, reserved = \
                            struct.unpack(self.RECORD_FORMAT, data)
                        
                        # 转换日期
                        try:
                            date_str = str(date_int)
                            if len(date_str) != 8:
                                # 可能是其他格式，尝试处理
                                if len(date_str) == 6:
                                    # YYMMDD 格式
                                    date_str = '20' + date_str
                                else:
                                    parse_errors.append({
                                        'record': record_num,
                                        'error': f'日期格式无效: {date_int}'
                                    })
                                    continue
                            
                            date = datetime.strptime(date_str, '%Y%m%d')
                        except ValueError as e:
                            parse_errors.append({
                                'record': record_num,
                                'error': f'日期解析失败: {date_int}, {e}'
                            })
                            continue
                        
                        # 转换价格（原始值 / 100）
                        open_price = open_raw / 100.0
                        high_price = high_raw / 100.0
                        low_price = low_raw / 100.0
                        close_price = close_raw / 100.0
                        
                        # 成交量和成交额已经是正确值（整数）
                        records.append({
                            'date': date,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': int(volume),
                            'amount': float(amount)  # 成交额转浮点数
                        })
                        
                    except struct.error as e:
                        parse_errors.append({
                            'record': record_num,
                            'error': f'struct.unpack 失败: {e}'
                        })
                    
                    record_num += 1
                    
        except PermissionError:
            return None, {'error': '文件权限不足，无法读取'}
        except Exception as e:
            return None, {'error': f'读取文件失败: {type(e).__name__}: {e}'}
        
        if not records:
            return None, {
                'error': '未能解析出任何有效记录',
                'parse_errors': parse_errors
            }
        
        # 创建 DataFrame
        df = pd.DataFrame(records)
        
        # 确保日期列为 datetime 类型
        df['date'] = pd.to_datetime(df['date'])
        
        metadata = {
            'filepath': str(filepath),
            'total_records': file_size // self.RECORD_SIZE,
            'parsed_records': len(records),
            'parse_errors': parse_errors
        }
        
        if parse_errors and self.verbose:
            for err in parse_errors:
                self.warnings.append({
                    'file': str(filepath),
                    'record': err['record'],
                    'warning': err['error']
                })
        
        return df, metadata
    
    def extract_code_from_filename(self, filename: str) -> str:
        """
        从文件名提取股票代码
        
        Args:
            filename: 文件名，如 000001.day 或 sh600001.day
            
        Returns:
            股票代码，如 000001
        """
        # 移除扩展名
        name = Path(filename).stem
        
        # 移除市场前缀（如 sh, sz）
        if name.lower().startswith(('sh', 'sz', 'bj')):
            name = name[2:]
        
        return name
    
    def standardize_code(self, code: str) -> str:
        """
        标准化股票代码
        
        Args:
            code: 原始代码，如 000001
            
        Returns:
            标准化代码，如 000001.SZ
        """
        code = code.strip()
        
        # 补齐 6 位
        if len(code) < 6:
            code = code.zfill(6)
        
        # 根据代码前缀判断市场
        prefix = code[:2]
        
        if prefix in ('60', '68'):
            return f'{code}.SH'
        elif prefix in ('00', '30'):
            return f'{code}.SZ'
        elif prefix in ('83', '87', '88'):
            return f'{code}.BJ'
        else:
            # 默认深圳
            return f'{code}.SZ'
    
    def apply_code_mapping(
        self, 
        filename: str, 
        mapping_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        应用代码映射
        
        Args:
            filename: 文件名
            mapping_df: 映射 DataFrame，包含 filename 和 standard_code 列
            
        Returns:
            标准化后的代码
        """
        if mapping_df is not None:
            # 查找映射
            match = mapping_df[mapping_df['filename'] == filename]
            if len(match) > 0:
                return match.iloc[0]['standard_code']
        
        # 无映射时，自动标准化
        code = self.extract_code_from_filename(filename)
        return self.standardize_code(code)
    
    def get_warnings(self) -> List[Dict]:
        """获取解析过程中的警告"""
        return self.warnings
    
    def get_errors(self) -> List[Dict]:
        """获取解析过程中的错误"""
        return self.errors
    
    def clear_logs(self):
        """清除警告和错误日志"""
        self.warnings = []
        self.errors = []