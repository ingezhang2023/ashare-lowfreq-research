"""
通达信 .day 文件导入 CLI 命令

命令：ashare-backtest import-tdx-day
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    print("请安装 tqdm: pip install tqdm")
    sys.exit(1)

from ashare_backtest.data.tdx_parser import TDXDayParser
from ashare_backtest.data.tdx_cleaner import TDXDataCleaner
from ashare_backtest.data.tdx_adjust import TDXAdjuster


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='导入通达信 .day 文件到 Parquet 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  ashare-backtest import-tdx-day --day-dir ./tdx_data --output-root ./output --adj-factor-dir ./adj
  ashare-backtest import-tdx-day --day-dir ./tdx_data --output-root ./output --no-adjust
  ashare-backtest import-tdx-day --day-dir ./tdx_data --output-root ./output --adj-factor-dir ./adj --start-date 20200101
        """
    )
    
    parser.add_argument('--day-dir', required=True, help='.day 文件所在目录')
    parser.add_argument('--output-root', required=True, help='输出 Parquet 的根目录（会创建 /ashare 子目录）')
    parser.add_argument('--adj-factor-dir', required=False, help='复权因子目录')
    parser.add_argument('--code-mapping', required=False, help='代码映射表 CSV 文件')
    parser.add_argument('--start-date', required=False, help='只导入此日期之后的数据（YYYYMMDD）')
    parser.add_argument('--end-date', required=False, help='只导入此日期之前的数据（YYYYMMDD）')
    parser.add_argument('--parallel', required=False, type=int, default=None, help='并行进程数')
    parser.add_argument('--verbose', action='store_true', help='输出详细日志')
    parser.add_argument('--validate-only', action='store_true', help='只验证文件格式，不实际写入')
    parser.add_argument('--no-adjust', action='store_true', help='跳过复权处理')
    
    return parser


def load_code_mapping(mapping_file: str) -> Optional[pd.DataFrame]:
    """加载代码映射表"""
    if not mapping_file:
        return None
    
    try:
        df = pd.read_csv(mapping_file)
        if 'filename' not in df.columns or 'standard_code' not in df.columns:
            print(f"映射表缺少必需列：filename, standard_code")
            return None
        return df
    except Exception as e:
        print(f"加载映射表失败: {e}")
        return None


def process_single_file(
    filepath: str,
    output_dir: str,
    adj_factor_dir: Optional[str],
    code_mapping: Optional[pd.DataFrame],
    start_date: Optional[str],
    end_date: Optional[str],
    no_adjust: bool,
    verbose: bool
) -> Dict:
    """
    处理单个 .day 文件
    
    Returns:
        处理结果字典
    """
    result = {
        'file': filepath,
        'success': False,
        'code': '',
        'records_parsed': 0,
        'records_after_cleaning': 0,
        'records_removed': 0,
        'warnings': [],
        'error': None
    }
    
    try:
        # 1. 解析文件
        parser = TDXDayParser(verbose=verbose)
        df, metadata = parser.parse_file(filepath)
        
        if df is None:
            result['error'] = metadata.get('error', '解析失败')
            return result
        
        result['records_parsed'] = len(df)
        
        # 2. 提取代码
        filename = Path(filepath).name
        code = parser.apply_code_mapping(filename, code_mapping)
        result['code'] = code
        
        # 3. 日期过滤
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['date'] <= end_dt]
        
        if df.empty:
            result['error'] = '日期过滤后无数据'
            return result
        
        # 4. 数据清洗
        cleaner = TDXDataCleaner(verbose=verbose)
        df, stats = cleaner.clean(df, code)
        
        result['records_after_cleaning'] = stats['final_count']
        result['records_removed'] = stats['removed_count']
        result['warnings'].extend(cleaner.get_warnings())
        
        if df.empty:
            result['error'] = '清洗后无数据'
            return result
        
        # 5. 复权处理
        if not no_adjust and adj_factor_dir:
            adjuster = TDXAdjuster(verbose=verbose)
            adj_df, adj_meta = adjuster.load_adj_factor(adj_factor_dir, code)
            
            if adj_df is None:
                result['error'] = adj_meta.get('error', '复权因子加载失败')
                return result
            
            df, adj_stats = adjuster.adjust(df, adj_df, code)
            result['warnings'].extend(adjuster.get_warnings())
        
        # 6. 输出 Parquet
        output_path = Path(output_dir) / 'ashare' / f'{code}.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保列顺序和数据类型
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        df['date'] = pd.to_datetime(df['date'])
        df['volume'] = df['volume'].astype('int64')
        
        df.to_parquet(output_path, index=False)
        
        # 7. 验证输出
        df_verify = pd.read_parquet(output_path)
        if len(df_verify) != len(df):
            result['error'] = '写入验证失败：行数不一致'
            return result
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {str(e)}'
    
    return result


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 检查必需参数
    if not args.no_adjust and not args.adj_factor_dir:
        print("=" * 50)
        print("错误：必须提供 --adj-factor-dir 或使用 --no-adjust")
        print("=" * 50)
        sys.exit(1)
    
    # 如果使用 --no-adjust，输出警告
    if args.no_adjust:
        print("=" * 50)
        print("WARNING: 未应用复权因子")
        print("除权除息日价格将出现断裂，回测结果不可用于实盘决策")
        print("=" * 50)
    
    # 检查目录
    day_dir = Path(args.day_dir)
    if not day_dir.exists():
        print(f"错误：.day 目录不存在: {day_dir}")
        sys.exit(1)
    
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 加载代码映射
    code_mapping = load_code_mapping(args.code_mapping) if args.code_mapping else None
    
    # 确定并行数
    if args.parallel:
        parallel_count = args.parallel
    else:
        parallel_count = max(1, multiprocessing.cpu_count() - 1)
    
    # 查找所有 .day 文件
    day_files = list(day_dir.glob('*.day'))
    if not day_files:
        print(f"错误：目录中没有 .day 文件: {day_dir}")
        sys.exit(1)
    
    print(f"找到 {len(day_files)} 个 .day 文件")
    print(f"并行进程数: {parallel_count}")
    
    # 初始化报告
    report = {
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_files_found': len(day_files),
        'successful_imports': 0,
        'failed_files': 0,
        'total_records_parsed': 0,
        'total_records_after_cleaning': 0,
        'records_removed_by_cleaning': 0,
        'cleaning_details': {
            'price_zero_or_negative': 0,
            'high_low_logic_error': 0,
            'missing_values': 0
        },
        'adjust_applied': not args.no_adjust,
        'warnings': [],
        'errors': []
    }
    
    # 并行处理
    results = []
    
    with ProcessPoolExecutor(max_workers=parallel_count) as executor:
        futures = {
            executor.submit(
                process_single_file,
                str(f),
                str(output_root),
                args.adj_factor_dir,
                code_mapping,
                args.start_date,
                args.end_date,
                args.no_adjust,
                args.verbose
            ): f for f in day_files
        }
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc='处理进度'):
            result = future.result()
            results.append(result)
            
            if result['success']:
                report['successful_imports'] += 1
                report['total_records_parsed'] += result['records_parsed']
                report['total_records_after_cleaning'] += result['records_after_cleaning']
                report['records_removed_by_cleaning'] += result['records_removed']
            else:
                report['failed_files'] += 1
                report['errors'].append({
                    'file': result['file'],
                    'error': result['error']
                })
            
            # 添加警告
            for w in result['warnings']:
                report['warnings'].append({
                    'file': result['file'],
                    'issue': w.get('warning', '')
                })
    
    # 保存报告
    report_path = output_root / 'tdx_import_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 输出摘要
    print("\n" + "=" * 50)
    print("导入完成")
    print("=" * 50)
    print(f"成功: {report['successful_imports']} / {report['total_files_found']}")
    print(f"失败: {report['failed_files']}")
    print(f"原始记录: {report['total_records_parsed']}")
    print(f"清洗后记录: {report['total_records_after_cleaning']}")
    print(f"删除记录: {report['records_removed_by_cleaning']}")
    print(f"\n报告已保存: {report_path}")
    
    return 0 if report['failed_files'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())