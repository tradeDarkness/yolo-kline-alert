"""
Chart Generator - 标准化K线图像生成

生成规则：
1. 信号点必须位于图片**最右侧**（模拟实战视角）
2. 必须绘制6条均线
3. K线颜色基于 close >= EMA120 判定
4. 白色背景，无坐标轴
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass  
class ChartConfig:
    """图表配置"""
    # 颜色设置（对应 Pine Script 用户代码）
    # SMA/EMA 20: 黑色
    col_sma20: str = '#000000'    
    col_ema20: str = '#000000'    
    
    # SMA/EMA 60: 蓝色
    col_sma60: str = '#0000FF'    
    col_ema60: str = '#0000FF'    
    
    # SMA/EMA 120: 紫色
    col_sma120: str = '#800080'   
    col_ema120: str = '#800080'   
    
    # K线颜色（基于 source >= MA(100)）
    # 用户指定颜色: #636363
    col_candle_up: str = '#636363'  
    col_candle_dn: str = '#636363'
    
    # 图像设置
    # YOLO 默认输入 640x640，为避免长宽比失真，生成时直接使用正方形画布
    fig_width: float = 6.4
    fig_height: float = 6.4
    dpi: int = 100
    
    # K线设置
    candle_width: float = 0.6
    ma_linewidth: float = 1.0
    ma_alpha: float = 1.0     # SMA 不透明
    ema_alpha: float = 0.5    # EMA 半透明


class ChartGenerator:
    """标准化K线图像生成器"""
    
    def __init__(self, config: ChartConfig = None):
        self.config = config or ChartConfig()
    
    def generate_chart(
        self,
        df: pd.DataFrame,
        signal_type: Optional[str] = None,
        output_path: Optional[str] = None,
        show_signal_marker: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """生成K线图"""
        cfg = self.config
        
        # 必需的列
        required_cols = ['open', 'high', 'low', 'close', 'SMA20', 'SMA60', 'SMA100', 'SMA120', 'EMA20', 'EMA60', 'EMA120']
        for col in required_cols:
            if col not in df.columns:
                # 兼容旧逻辑，如果没有SMA100，尝试用SMA120代替或报错
                if col == 'SMA100' and 'SMA120' in df.columns:
                    df['SMA100'] = df['SMA120']
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # 数据准备
        n = len(df)
        dates = np.arange(n)
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        trend_ma_vals = df['SMA100'].values
        
        # 坐标范围
        ma_cols = ['SMA20', 'SMA60', 'SMA120', 'EMA20', 'EMA60', 'EMA120']
        
        x_min, x_max = 0, n - 1
        y_min_data = min(df['low'].min(), df[ma_cols].min().min())
        y_max_data = max(df['high'].max(), df[ma_cols].max().max())
        y_pad = (y_max_data - y_min_data) * 0.05
        y_min_limit = y_min_data - y_pad
        y_max_limit = y_max_data + y_pad
        
        # 创建图形
        fig = plt.figure(figsize=(cfg.fig_width, cfg.fig_height), dpi=cfg.dpi)
        fig.patch.set_facecolor('white')
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor('white')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min_limit, y_max_limit)
        
        # 绘制均线
        # 1. 绘制 EMA (半透明)
        ema_cols = ['EMA20', 'EMA60', 'EMA120']
        ema_colors = [cfg.col_ema20, cfg.col_ema60, cfg.col_ema120]
        for col, color in zip(ema_cols, ema_colors):
            ax.plot(dates, df[col].values, color=color, 
                   linewidth=cfg.ma_linewidth, alpha=cfg.ema_alpha)
                   
        # 2. 绘制 SMA (不透明)
        sma_cols = ['SMA20', 'SMA60', 'SMA120']
        sma_colors = [cfg.col_sma20, cfg.col_sma60, cfg.col_sma120]
        for col, color in zip(sma_cols, sma_colors):
            ax.plot(dates, df[col].values, color=color, 
                   linewidth=cfg.ma_linewidth, alpha=cfg.ma_alpha)
        
        # 绘制K线
        width = cfg.candle_width
        width2 = width / 2
        
        for i, (d, o, c, h, l, ma_val) in enumerate(zip(dates, opens, closes, highs, lows, trend_ma_vals)):
            # 颜色：基于 close >= MA100
            # 用户代码: source >= MA ? bull_color : bear_color
            color = cfg.col_candle_up if c >= ma_val else cfg.col_candle_dn
            
            # 影线
            ax.plot([d, d], [l, h], color=color, linewidth=1.0)
            
            # 实体
            lower = min(o, c)
            height = abs(c - o)
            if height == 0:
                height = (h - l) * 0.01 if h != l else 0.001
            
            rect = patches.Rectangle(
                (d - width2, lower), width, height,
                linewidth=0, edgecolor=None, facecolor=color
            )
            ax.add_patch(rect)
        
        # 注意：不绘制信号标记（三角形）
        # 原因：如果保留箭头，YOLO会退化成"寻找三角形"的模型
        # 只保留K线和均线，让模型学习形态本身
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 保存
        if output_path:
            plt.savefig(output_path, facecolor='white', dpi=cfg.dpi)
            plt.close(fig)
            return None, None
        
        return fig, ax
    
    def generate_yolo_label(
        self,
        df: pd.DataFrame,
        cluster_start_idx: Optional[int] = None,
        cluster_end_idx: Optional[int] = None,
        class_id: int = 0,
    ) -> str:
        """
        生成 YOLO 格式的标签
        
        Args:
            df: K线数据
            cluster_start_idx: 粘合区起始索引（在 df 内）
            cluster_end_idx: 粘合区结束索引（在 df 内）
            class_id: 类别ID (0=LONG, 1=SHORT)
            
        Returns:
            YOLO 格式标签字符串，如 "0 0.5 0.5 0.2 0.1"
        """
        if cluster_start_idx is None or cluster_end_idx is None:
            return ""
        
        n = len(df)
        ma_cols = ['SMA20', 'SMA60', 'SMA120', 'EMA20', 'EMA60', 'EMA120']
        
        # 坐标范围（与绘图一致）
        x_min, x_max = 0, n - 1
        y_min_data = min(df['low'].min(), df[ma_cols].min().min())
        y_max_data = max(df['high'].max(), df[ma_cols].max().max())
        y_pad = (y_max_data - y_min_data) * 0.05
        y_min_limit = y_min_data - y_pad
        y_max_limit = y_max_data + y_pad
        
        # 粘合区的价格范围
        cluster_df = df.iloc[cluster_start_idx:cluster_end_idx + 1]
        cluster_data = cluster_df[ma_cols]
        box_min_price = cluster_data.min().min()
        box_max_price = cluster_data.max().max()
        
        price_pad = (box_max_price - box_min_price) * 0.15
        box_min_price -= price_pad
        box_max_price += price_pad
        
        # 归一化
        def normalize_x(val):
            return (val - x_min) / (x_max - x_min)
        
        def normalize_y(val):
            norm = (val - y_min_limit) / (y_max_limit - y_min_limit)
            return 1.0 - norm  # 图像坐标Y轴反转
        
        nx1 = normalize_x(cluster_start_idx)
        nx2 = normalize_x(cluster_end_idx)
        ny_top = normalize_y(box_max_price)
        ny_bottom = normalize_y(box_min_price)
        
        w = abs(nx2 - nx1)
        h = abs(ny_bottom - ny_top)
        cx = (nx1 + nx2) / 2
        cy = (ny_top + ny_bottom) / 2
        
        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0.01, min(1, w))
        h = max(0.01, min(1, h))
        
        # Class 0 = ma_cluster
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def find_adhesion_region(df: pd.DataFrame, adhesion_threshold_pct: float = 0.5) -> Tuple[int, int]:
    """
    在数据中找到均线粘合区域
    
    Args:
        df: 包含均线列的 DataFrame
        adhesion_threshold_pct: 粘合阈值（价格的百分比）
        
    Returns:
        (start_idx, end_idx) 粘合区域的索引范围
    """
    ma_cols = ['SMA20', 'SMA60', 'SMA120']
    
    # 计算均线差值
    diff1 = (df['SMA20'] - df['SMA60']).abs()
    diff2 = (df['SMA60'] - df['SMA120']).abs()
    diff3 = (df['SMA20'] - df['SMA120']).abs()
    max_diff = pd.concat([diff1, diff2, diff3], axis=1).max(axis=1)
    
    threshold = df['close'] * adhesion_threshold_pct / 100.0
    is_adhesion = max_diff <= threshold
    
    # 从右边（最后一根K线）往左找粘合区
    n = len(df)
    end_idx = n - 1
    start_idx = end_idx
    
    # 往左扩展粘合区
    while start_idx > 0 and is_adhesion.iloc[start_idx]:
        start_idx -= 1
    
    # start_idx 现在指向第一个非粘合点，调整到粘合区起点
    if not is_adhesion.iloc[start_idx]:
        start_idx += 1
    
    # 确保最小宽度
    if end_idx - start_idx < 5:
        start_idx = max(0, end_idx - 30)
    
    return start_idx, end_idx


if __name__ == "__main__":
    # 测试
    import numpy as np
    
    print("Chart Generator - Test")
    
    np.random.seed(42)
    n = 60
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + np.abs(np.random.randn(n) * 0.3),
        'low': close - np.abs(np.random.randn(n) * 0.3),
        'close': close,
    })
    
    # 添加均线
    df['SMA20'] = df['close'].rolling(20).mean()
    df['SMA60'] = df['close'].rolling(20).mean()  # 为了测试，用更短周期
    df['SMA120'] = df['close'].rolling(20).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA60'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA120'] = df['close'].ewm(span=20, adjust=False).mean()
    
    df = df.dropna()
    
    generator = ChartGenerator()
    
    # 生成图表
    output_path = "/tmp/test_chart.png"
    generator.generate_chart(df, signal_type='LONG', output_path=output_path)
    print(f"Chart saved to: {output_path}")
    
    # 生成标签
    start, end = find_adhesion_region(df)
    label = generator.generate_yolo_label(df, start, end)
    print(f"YOLO Label: {label}")
