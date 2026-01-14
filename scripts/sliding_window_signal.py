"""
Sliding Window Signal Detection - 滑动窗口信号检测与图像生成

主入口脚本，实现：
1. 获取 ETHUSDT.P 5分钟级别数据
2. 使用滑动窗口遍历数据
3. 对每个窗口的最后一根K线应用信号检测规则
4. 生成标准化图像（信号点在最右侧）
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# 添加脚本目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from okx_utils import fetch_candles, get_top_volume_pairs
from pine_signal_detector import PineSignalDetector, SignalConfig, detect_signals_in_window
from chart_generator import ChartGenerator, ChartConfig, find_adhesion_region


# ============================================================
# 配置
# ============================================================
DEFAULT_SYMBOL = "ETH-USDT-SWAP"
DEFAULT_BAR = "5m"
DEFAULT_LIMIT = 3000  # 获取足够多的历史数据
DEFAULT_WINDOW_SIZE = 60  # 窗口大小（根K线）
DEFAULT_STRIDE = 1  # 滑动步长

OUTPUT_DIR = "data/pine_signals"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")


def setup_dirs():
    """创建输出目录"""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)


def sliding_window_detect(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    signal_config: SignalConfig = None,
    dry_run: bool = False,
) -> List[dict]:
    """
    滑动窗口信号检测
    
    Args:
        df: 完整的K线数据 DataFrame
        symbol: 交易对符号
        window_size: 窗口大小（用于图像生成）
        stride: 滑动步长
        signal_config: 信号检测配置
        dry_run: 如果为 True，只输出信号时间戳，不生成图像
    
    Returns:
        检测到的信号列表
    """
    setup_dirs()
    
    detector = PineSignalDetector(signal_config or SignalConfig())
    chart_gen = ChartGenerator()
    
    signals = []
    n = len(df)
    
    # 先在整个数据上计算所有指标
    print(f"📊 预计算指标...")
    df = detector.calculate_indicators(df)
    df = detector.calculate_stateful_signals(df)
    
    # 需要足够的数据来计算 SMA120
    min_start = max(120, window_size)
    
    print(f"📊 开始滑动窗口检测...")
    print(f"   数据总长度: {n}")
    print(f"   窗口大小: {window_size}")
    print(f"   滑动步长: {stride}")
    print(f"   检测范围: {min_start} - {n}")
    
    detected_count = 0
    
    # 遍历每个K线索引，检查信号
    for current_idx in range(min_start, n, stride):
        
        # 确保有足够数据
        if pd.isna(df['SMA120'].iloc[current_idx]):
            continue
        
        is_long, is_short = detector.check_signal(df, current_idx)
        
        if is_long or is_short:
            signal_type = 'LONG' if is_long else 'SHORT'
            
            # 获取时间戳
            if 'datetime' in df.columns:
                timestamp = df['datetime'].iloc[current_idx]
            else:
                timestamp = df.index[current_idx]
            
            signal_info = {
                'timestamp': str(timestamp),
                'type': signal_type,
                'close': float(df['close'].iloc[current_idx]),
                'df_index': current_idx,
            }
            
            signals.append(signal_info)
            detected_count += 1
            
            # 进度输出
            if detected_count % 10 == 0:
                print(f"   已检测到 {detected_count} 个信号...")
            
            if not dry_run:
                # 提取窗口数据用于图像生成
                # 信号K线之后的第2根K线作为图片最右边
                # signal at current_idx, chart ends at current_idx + 2
                chart_end_idx = current_idx + 2
                if chart_end_idx >= len(df):
                    continue  # 数据不够，跳过
                start_idx = max(0, chart_end_idx - window_size + 1)
                window_df = df.iloc[start_idx:chart_end_idx + 1].copy()
                
                # 生成图像
                _save_signal_chart(window_df, signal_type, timestamp, chart_gen, symbol)
    
    print(f"\n✅ 检测完成！共发现 {len(signals)} 个信号")
    return signals


def _save_signal_chart(
    window_df: pd.DataFrame,
    signal_type: str,
    timestamp,
    chart_gen: ChartGenerator,
    symbol: str = "UNKNOWN",
) -> bool:
    """
    保存信号对应的图表
    
    关键：信号点位于图片最右侧（window_df 的最后一行就是信号K线）
    """
    try:
        # 生成文件名（增加 symbol 前缀防止冲突）
        ts_str = timestamp.strftime('%Y%m%d_%H%M') if hasattr(timestamp, 'strftime') else str(timestamp).replace(':', '').replace('-', '').replace(' ', '_')
        safe_symbol = symbol.replace('-', '').replace('_', '')
        
        base_name = f"{safe_symbol}_{signal_type}_{ts_str}"
        img_path = os.path.join(IMAGE_DIR, base_name + ".png")
        txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
        
        # 生成图像
        chart_gen.generate_chart(
            window_df,
            signal_type=signal_type,
            output_path=img_path,
            show_signal_marker=True
        )
        
        # 生成 YOLO 标签
        start_idx, end_idx = find_adhesion_region(window_df)
        
        # 确定类别ID (LONG=0, SHORT=1)
        class_id = 0 if signal_type == 'LONG' else 1
        
        label = chart_gen.generate_yolo_label(window_df, start_idx, end_idx, class_id=class_id)
        
        with open(txt_path, 'w') as f:
            if label:
                f.write(label + "\n")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ 保存图表失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pine Script 滑动窗口信号检测")
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL,
                        help=f'单交易对符号 (default: {DEFAULT_SYMBOL})')
    parser.add_argument('--symbols', type=str, default=None,
                        help='多交易对符号列表 (逗号分隔)，例如: BTC-USDT-SWAP,ETH-USDT-SWAP')
    parser.add_argument('--top', type=int, default=None,
                        help='自动获取成交量前 N 的币种 (例如 50)，覆盖 symbols 参数')
    
    parser.add_argument('--bar', type=str, default=DEFAULT_BAR,
                        help=f'K线周期 (default: {DEFAULT_BAR})')
    # 默认获取 110000 根 (约1年 5m 数据)
    parser.add_argument('--limit', type=int, default=110000,
                        help=f'获取K线数量 (default: 110000)')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW_SIZE,
                        help=f'窗口大小 (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--stride', type=int, default=DEFAULT_STRIDE,
                        help=f'滑动步长 (default: {DEFAULT_STRIDE})')
    parser.add_argument('--dry-run', action='store_true',
                        help='只输出信号时间戳，不生成图像')
    parser.add_argument('--output-json', type=str, default=None,
                        help='将检测结果保存到 JSON 文件')
    
    # 信号配置参数
    parser.add_argument('--no-strict', action='store_true',
                        help='禁用严格的6均线过滤模式')
    parser.add_argument('--min-ma', type=int, default=4,
                        help='非严格模式下最少需满足的均线数量')
    parser.add_argument('--no-osc-filter', action='store_true',
                        help='禁用动能过滤')
    parser.add_argument('--no-alignment-filter', action='store_true',
                        help='禁用均线排列过滤')
    parser.add_argument('--no-power-filter', action='store_true',
                        help='禁用K线力度过滤')
    
    args = parser.parse_args()
    
    # 创建信号配置
    signal_config = SignalConfig(
        use_strict_filter=not args.no_strict,
        min_ma_confirm=args.min_ma,
        use_osc_filter=not args.no_osc_filter,
        use_alignment_filter=not args.no_alignment_filter,
        use_candle_power=not args.no_power_filter,
    )
    
    print("=" * 60)
    print("Pine Script 滑动窗口信号检测")
    print("=" * 60)
    print(f"📌 交易对: {args.symbols if args.symbols else args.symbol}")
    print(f"📌 K线周期: {args.bar}")
    print(f"📌 获取数量: {args.limit}")
    print(f"📌 窗口大小: {args.window}")
    print(f"📌 滑动步长: {args.stride}")
    print(f"📌 严格模式: {not args.no_strict}")
    print(f"📌 Dry Run: {args.dry_run}")
    print()
    
    # 确定要处理的 symbol 列表
    symbol_list = []
    
    if args.top:
        print(f"🌟 正在获取 OKX 成交量前 {args.top} 的币种...")
        symbol_list = get_top_volume_pairs(args.top)
        print(f"👉 获取到: {len(symbol_list)} 个币种")
        print(f"   列表: {symbol_list[:5]} ...")
    elif args.symbols:
        symbol_list = [s.strip() for s in args.symbols.split(',') if s.strip()]
    else:
        symbol_list = [args.symbol]
    
    total_signals_all = 0
    
    for symbol in symbol_list:
        print("=" * 40)
        print(f"🚀 开始处理: {symbol}")
        print("=" * 40)
        
        # 获取数据
        print(f"⏳ 正在获取 {symbol} {args.bar} 数据...")
        try:
            df = fetch_candles(symbol, bar=args.bar, limit=args.limit)
        except Exception as e:
            print(f"❌ 获取数据异常: {e}")
            continue
        
        if df is None or len(df) < args.window + 120:
            print(f"❌ 数据获取失败或数据不足")
            continue
        
        print(f"✅ 获取到 {len(df)} 根K线")
        if len(df) > 0:
            print(f"   时间范围: {df['datetime'].iloc[0]} - {df['datetime'].iloc[-1]}")
        
        # 滑动窗口检测
        signals = sliding_window_detect(
            df,
            symbol=symbol,
            window_size=args.window,
            stride=args.stride,
            signal_config=signal_config,
            dry_run=args.dry_run,
        )
        
        if signals:
            total_signals_all += len(signals)
            print(f"\n✅ {symbol} 检测到 {len(signals)} 个信号")
            
            # 输出前几个
            print("-" * 40)
            for sig in signals[:5]:
                print(f"   {sig['timestamp']} | {sig['type']:5} | {sig['close']:.4f}")
            if len(signals) > 5:
                print(f"   ... (更多)")
        else:
            print(f"⚠️ {symbol} 未检测到信号")
            
        print("\n")
        
    print(f"\n🎉 所有任务完成！总共发现 {total_signals_all} 个信号。")
    print(f"📁 图像目录: {IMAGE_DIR}")
    print(f"📁 标签目录: {LABEL_DIR}")
    
    if not args.dry_run:
        print(f"\n📁 图像输出目录: {IMAGE_DIR}")
        print(f"📁 标签输出目录: {LABEL_DIR}")


if __name__ == "__main__":
    main()
