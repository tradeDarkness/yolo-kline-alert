import pandas as pd
import numpy as np
import mplfinance as mpf
import os
import random

# --- 配置 ---
output_dir = 'data/kline_dataset'
os.makedirs(f"{output_dir}/train/images", exist_ok=True)
os.makedirs(f"{output_dir}/train/labels", exist_ok=True)

# OKX 风格颜色定义
COLOR_SMA20 = '#000000'   # 黑色
COLOR_SMA60 = '#2962FF'   # 蓝色 (接近 OKX/TradingView)
COLOR_SMA120 = '#8E24AA'  # 紫色

def generate_cluster_dump_data(n=150):
    """
    生成：震荡 -> 均线密集 (Cluster) -> 暴跌 (Dump) 的数据
    """
    # 1. 基础波动 (随机游走)
    changes = np.random.normal(0, 0.8, n)
    
    # 2. 定义关键时间点
    cluster_start = 60
    cluster_end = 90
    dump_start = 91
    
    # --- 关键步骤：造假数据 ---
    
    # A. 制造“均线密集”：在 cluster 区间内，将波动率压缩到极致
    # 这样 SMA 和 EMA 就会被迫粘合在一起
    changes[cluster_start:cluster_end] = np.random.normal(0, 0.15, cluster_end-cluster_start)
    
    # B. 制造“暴跌”：在 cluster 之后，强制向下
    # 生成一系列负数，模拟大阴线
    changes[dump_start:dump_start+20] = np.random.normal(-1.5, 0.5, 20) 
    
    # C. 合成价格曲线
    prices = 100 + np.cumsum(changes)
    
    # D. 构造 OHLC 数据
    df = pd.DataFrame({
        'Close': prices,
        'Open': prices - changes + np.random.normal(0, 0.1, n), # Open 稍微偏离一点
        'High': np.maximum(prices, prices - changes) + np.abs(np.random.normal(0, 0.3, n)),
        'Low': np.minimum(prices, prices - changes) - np.abs(np.random.normal(0, 0.3, n))
    }, index=pd.date_range('2026-01-01', periods=n, freq='15min'))
    
    return df, (cluster_start, cluster_end)

def create_sample(idx):
    # 生成数据
    df, (c_start, c_end) = generate_cluster_dump_data(150)
    
    # 计算指标 (你的 Pine Script 逻辑)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['SMA120'] = df['Close'].rolling(120).mean()
    
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['EMA120'] = df['Close'].ewm(span=120, adjust=False).mean()

    # 裁剪掉最前面的数据(因为均线计算初期是NaN)
    valid_start = 20
    df = df.iloc[valid_start:]
    # 修正坐标偏移
    real_c_start = c_start - valid_start
    real_c_end = c_end - valid_start
    
    # 设置绘图风格 (OKX 风格)
    # 涨=绿，跌=红，背景=白，无网格
    mc = mpf.make_marketcolors(up='#0ecb81', down='#f6465d', edge='inherit', wick='inherit')
    s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, gridstyle='', facecolor='white')

    # 添加均线 (注意 EMA 设置 alpha 透明度)
    apds = [
        mpf.make_addplot(df['SMA20'], color=COLOR_SMA20, width=1.2),
        mpf.make_addplot(df['SMA60'], color=COLOR_SMA60, width=1.2),
        mpf.make_addplot(df['SMA120'], color=COLOR_SMA120, width=1.2),
        # EMA 半透明
        mpf.make_addplot(df['EMA20'], color=COLOR_SMA20, width=0.8, alpha=0.4),
        mpf.make_addplot(df['EMA60'], color=COLOR_SMA60, width=0.8, alpha=0.4),
        mpf.make_addplot(df['EMA120'], color=COLOR_SMA120, width=0.8, alpha=0.4),
    ]

    img_name = f"synth_v2_{idx}.png"
    img_path = os.path.join(output_dir, 'train/images', img_name)
    
    # 绘图：axisoff=True 去掉坐标轴，tight_layout 去掉白边
    mpf.plot(df, type='candle', addplot=apds, style=s,
             savefig=dict(fname=img_path, dpi=100, bbox_inches='tight', pad_inches=0),
             axisoff=True, tight_layout=True, volume=False)

    # --- 自动计算 YOLO 标签 ---
    # 我们知道密集区发生的时间段是 real_c_start 到 real_c_end
    # 我们需要找出这段时间内，价格在Y轴上的范围
    
    # 1. 获取全图数据的极值 (用于归一化)
    # 注意：mplfinance 自动缩放，不仅看 K 线，也看均线。我们要取所有绘图元素的最值。
    all_data_max = max(df[['High', 'SMA20', 'SMA120']].max().max(), df[['High', 'SMA20', 'SMA120']].max().max())
    all_data_min = min(df[['Low', 'SMA20', 'SMA120']].min().min(), df[['Low', 'SMA20', 'SMA120']].min().min())
    plot_range = all_data_max - all_data_min

    # 2. 获取“密集区”内的价格范围
    cluster_slice = df.iloc[real_c_start:real_c_end]
    cluster_high = cluster_slice['High'].max()
    cluster_low = cluster_slice['Low'].min()
    
    # 稍微放大一点框，包住均线
    box_top = cluster_high + (plot_range * 0.02) 
    box_bottom = cluster_low - (plot_range * 0.02)
    
    # 3. 转换为 YOLO 坐标 (cx, cy, w, h) 全部归一化到 0-1
    total_bars = len(df)
    
    # X轴计算
    x_center_idx = (real_c_start + real_c_end) / 2
    cx = x_center_idx / total_bars
    w = (real_c_end - real_c_start) / total_bars
    
    # Y轴计算 (注意：YOLO原点在左上角，而价格原点在下方，所以 y = 1 - price_ratio)
    y_top_ratio = (box_top - all_data_min) / plot_range
    y_bottom_ratio = (box_bottom - all_data_min) / plot_range
    
    h = y_top_ratio - y_bottom_ratio
    cy = 1 - (y_top_ratio + y_bottom_ratio) / 2 # 翻转Y轴
    
    # 写入标签
    txt_path = os.path.join(output_dir, 'train/labels', f"synth_v2_{idx}.txt")
    with open(txt_path, 'w') as f:
        # 类别 0，保留6位小数
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# 执行
print("正在生成 OKX 风格的高仿真数据...")
for i in range(200): # 直接生成 200 张
    create_sample(i)
    if i % 20 == 0: print(f"进度: {i}/200")
print("✅ 完成！请检查 data/kline_dataset/train/images 文件夹")