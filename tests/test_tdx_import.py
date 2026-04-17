"""
通达信数据导入模块测试

测试 tdx_parser, tdx_cleaner, tdx_adjust 模块
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import struct

from ashare_backtest.data.tdx_parser import TDXDayParser
from ashare_backtest.data.tdx_cleaner import TDXDataCleaner
from ashare_backtest.data.tdx_adjust import TDXAdjuster


class TestTDXDayParser:
    """测试通达信 .day 文件解析器"""
    
    def test_parse_mock_file(self):
        """测试解析模拟 .day 文件"""
        parser = TDXDayParser()
        
        # 创建临时 .day 文件
        with tempfile.NamedTemporaryFile(suffix='.day', delete=False) as f:
            # 写入 3 条记录
            for i in range(3):
                record = struct.pack(
                    '<IIIIIIII',
                    20240101 + i,
                    100000 + i * 1000,
                    110000 + i * 1000,
                    90000 + i * 1000,
                    105000 + i * 1000,
                    10000 + i * 100,
                    10000000 + i * 100000,
                    0
                )
                f.write(record)
            f.flush()
            
            # 解析文件
            df, meta = parser.parse_file(f.name)
            
            assert df is not None
            assert len(df) == 3
            assert 'parsed_records' in meta
            assert 'filepath' in meta
            
            # 清理临时文件
            Path(f.name).unlink()


class TestTDXDataCleaner:
    """测试数据清洗器"""
    
    def test_clean_valid_data(self):
        """测试清洗有效数据"""
        cleaner = TDXDataCleaner()
        
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'open': [10.0, 11.0, 12.0],
            'high': [11.0, 12.0, 13.0],
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 2000, 3000],
            'amount': [10000, 20000, 30000]
        })
        
        cleaned_df, stats = cleaner.clean(df, '000001.SZ')
        
        assert len(cleaned_df) == 3
        assert stats['removed_count'] == 0
    
    def test_clean_zero_price(self):
        """测试清洗零价格数据"""
        cleaner = TDXDataCleaner()
        
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'open': [10.0, 0.0, 12.0],  # 第二条开盘价为 0
            'high': [11.0, 0.0, 13.0],
            'low': [9.0, 0.0, 11.0],
            'close': [10.5, 0.0, 12.5],
            'volume': [1000, 0, 3000],
            'amount': [10000, 0, 30000]
        })
        
        cleaned_df, stats = cleaner.clean(df, '000001.SZ')
        
        assert len(cleaned_df) == 2  # 第二条被删除
        assert stats['removed_count'] == 1
    
    def test_clean_high_low_logic_error(self):
        """测试清洗高低价逻辑错误"""
        cleaner = TDXDataCleaner()
        
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'open': [10.0, 11.0, 12.0],
            'high': [11.0, 9.0, 13.0],  # 第二条最高价 < 开盘价 (逻辑错误)
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 2000, 3000],
            'amount': [10000, 20000, 30000]
        })
        
        cleaned_df, stats = cleaner.clean(df, '000001.SZ')
        
        assert len(cleaned_df) == 2  # 第二条被删除
        assert stats['removed_count'] == 1


class TestTDXAdjuster:
    """测试前复权处理器"""
    
    def test_load_adj_factor(self):
        """测试加载复权因子"""
        adjuster = TDXAdjuster()
        
        # 创建临时复权因子文件
        with tempfile.TemporaryDirectory() as tmpdir:
            adj_file = Path(tmpdir) / '000001.SZ.parquet'
            
            adj_df = pd.DataFrame({
                'date': pd.to_datetime(['2024-01-01', '2024-06-01']),
                'adj_factor': [1.0, 1.5]  # 6月1日有分红，因子变为1.5
            })
            adj_df.to_parquet(adj_file)
            
            loaded_df, meta = adjuster.load_adj_factor(tmpdir, '000001.SZ')
            
            assert loaded_df is not None
            assert len(loaded_df) == 2
            assert 'date' in loaded_df.columns
            assert 'adj_factor' in loaded_df.columns
    
    def test_apply_adjustment(self):
        """测试应用复权因子"""
        adjuster = TDXAdjuster()
        
        # 价格数据
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-06-02']),
            'open': [10.0, 10.0],
            'high': [11.0, 11.0],
            'low': [9.0, 9.0],
            'close': [10.5, 10.5],
            'volume': [1000, 1000],
            'amount': [10000, 10000]
        })
        
        # 复权因子（注意：列名必须是 'date'）
        adj_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-06-01']),
            'adj_factor': [1.0, 1.5]
        })
        
        # 应用复权
        adjusted_df, stats = adjuster.adjust(df, adj_df, '000001.SZ')
        
        # 验证结果
        assert len(adjusted_df) == 2
        # adj_factor 列已被删除（不包含在输出中）
        assert 'adj_factor' not in adjusted_df.columns
        assert 'date' in adjusted_df.columns
        assert 'close' in adjusted_df.columns
        
        # 1月1日价格不变 (因子=1.0)
        assert adjusted_df.loc[0, 'close'] == 10.5
        
        # 6月2日价格调整 (因子=1.5)
        assert stats['matched_factors'] > 0


class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整处理流程"""
        parser = TDXDayParser()
        cleaner = TDXDataCleaner()
        
        # 创建临时 .day 文件
        with tempfile.NamedTemporaryFile(suffix='.day', delete=False) as f:
            for i in range(10):
                record = struct.pack(
                    '<IIIIIIII',
                    20240101 + i,
                    100000 + i * 1000,
                    110000 + i * 1000,
                    90000 + i * 1000,
                    105000 + i * 1000,
                    10000 + i * 100,
                    10000000 + i * 100000,
                    0
                )
                f.write(record)
            f.flush()
            
            # 解析
            df, meta = parser.parse_file(f.name)
            assert df is not None
            assert len(df) == 10
            
            # 清洗
            cleaned_df, stats = cleaner.clean(df, '000001.SZ')
            assert len(cleaned_df) == 10
            
            # 输出
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = Path(tmpdir) / '000001.SZ.parquet'
                cleaned_df.to_parquet(out_path)
                
                # 验证
                verify_df = pd.read_parquet(out_path)
                assert len(verify_df) == 10
            
            Path(f.name).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])