"""
Pine Script Signal Detector - 完整复现 TradingView 指标逻辑

复现用户的 Pine Script 指标 "20260114" 的完整信号逻辑，包括：
- 6条均线: SMA(20,60,120) + EMA(20,60,120)
- 均线粘合检测
- 交叉事件统计（滚动窗口）
- 粘合后首次突破检测 (方案A)
- 密集区高低点突破检测 (方案D)
- 六均线过滤 (方案B)
- 动能过滤
- 均线排列过滤
- K线力度过滤
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SignalConfig:
    """信号检测配置参数（对应 Pine Script 的 input 参数）"""
    # 均线设置
    ma_period_1: int = 20       # 短期均线周期
    ma_period_2: int = 60       # 中期均线周期
    ma_period_3: int = 120      # 长期均线周期
    
    # 密集检测
    density_window: int = 5     # 密集检测窗口
    cross_threshold: int = 4    # 交叉阈值
    bull_ratio: float = 60.0    # 看多比例阈值(%)
    adhesion_threshold: float = 0.5  # 均线粘合阈值(%)
    
    # 六均线过滤
    min_ma_confirm: int = 4     # 最少需满足均线数量
    use_strict_filter: bool = True  # 严格模式
    
    # 动能过滤
    use_osc_filter: bool = True
    osc_confirm_bars: int = 3   # 动能确认窗口（根K线）
    osc_threshold: float = 80.0 # 动能阈值(%)
    osc_ma_length: int = 50     # 动能计算MA长度
    
    # 均线排列过滤
    use_alignment_filter: bool = True
    
    # K线力度过滤
    use_candle_power: bool = True
    power_ratio: float = 75.0   # K线收盘力度阈值(%)


class PineSignalDetector:
    """Pine Script 信号检测器"""
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Args:
            df: 包含 open, high, low, close 列的 DataFrame
            
        Returns:
            添加了指标列的 DataFrame
        """
        df = df.copy()
        cfg = self.config
        
        # ========== 6条均线 ==========
        df['SMA20'] = df['close'].rolling(cfg.ma_period_1).mean()
        df['SMA60'] = df['close'].rolling(cfg.ma_period_2).mean()
        df['SMA100'] = df['close'].rolling(100).mean()  # 用于K线着色
        df['SMA120'] = df['close'].rolling(cfg.ma_period_3).mean()
        
        df['EMA20'] = df['close'].ewm(span=cfg.ma_period_1, adjust=False).mean()
        df['EMA60'] = df['close'].ewm(span=cfg.ma_period_2, adjust=False).mean()
        df['EMA120'] = df['close'].ewm(span=cfg.ma_period_3, adjust=False).mean()
        
        # ========== 交叉检测 ==========
        df['cross_up_1'] = self._crossover(df['SMA20'], df['SMA60'])
        df['cross_dn_1'] = self._crossunder(df['SMA20'], df['SMA60'])
        df['cross_up_2'] = self._crossover(df['SMA60'], df['SMA120'])
        df['cross_dn_2'] = self._crossunder(df['SMA60'], df['SMA120'])
        df['cross_up_3'] = self._crossover(df['SMA20'], df['SMA120'])
        df['cross_dn_3'] = self._crossunder(df['SMA20'], df['SMA120'])
        
        # ========== 交叉事件统计（滚动窗口）==========
        cross_event = (
            df['cross_up_1'] | df['cross_up_2'] | df['cross_up_3'] |
            df['cross_dn_1'] | df['cross_dn_2'] | df['cross_dn_3']
        ).astype(int)
        
        bull_event = (
            df['cross_up_1'] | df['cross_up_2'] | df['cross_up_3']
        ).astype(int)
        
        bear_event = (
            df['cross_dn_1'] | df['cross_dn_2'] | df['cross_dn_3']
        ).astype(int)
        
        df['total_in_window'] = cross_event.rolling(cfg.density_window).sum()
        df['bullish_cross'] = bull_event.rolling(cfg.density_window).sum()
        df['bearish_cross'] = bear_event.rolling(cfg.density_window).sum()
        
        # ========== 密集区域检测 ==========
        df['is_dense_area'] = df['total_in_window'] >= cfg.cross_threshold
        
        # ========== 均线粘合检测 ==========
        diff1 = (df['SMA20'] - df['SMA60']).abs()
        diff2 = (df['SMA60'] - df['SMA120']).abs()
        diff3 = (df['SMA20'] - df['SMA120']).abs()
        max_diff = pd.concat([diff1, diff2, diff3], axis=1).max(axis=1)
        
        # 粘合阈值：使用价格的百分比
        threshold_value = df['close'] * cfg.adhesion_threshold / 100.0
        df['is_adhesion'] = max_diff <= threshold_value
        
        # ========== 动能振荡器 ==========
        osc_src = df['close']
        osc_ma = osc_src.rolling(cfg.osc_ma_length).mean()
        osc_diff = osc_src - osc_ma
        osc_max = osc_diff.abs().rolling(cfg.osc_ma_length).max()
        df['osc'] = np.where(osc_max != 0, osc_diff / osc_max * 100, 0)
        
        # ========== 均线排列 ==========
        df['is_bullish_alignment'] = (df['SMA20'] > df['SMA60']) & (df['SMA60'] > df['SMA120'])
        df['is_bearish_alignment'] = (df['SMA20'] < df['SMA60']) & (df['SMA60'] < df['SMA120'])
        
        return df
    
    def calculate_stateful_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算需要状态跟踪的信号（方案A和方案D）
        必须在 calculate_indicators 之后调用
        
        这些信号需要逐行迭代，因为涉及状态变量
        """
        df = df.copy()
        cfg = self.config
        
        n = len(df)
        
        # 方案A: 粘合后首次突破
        was_adhesion_bull = np.zeros(n, dtype=bool)
        was_adhesion_bear = np.zeros(n, dtype=bool)
        adhesion_breakout_up = np.zeros(n, dtype=bool)
        adhesion_breakout_down = np.zeros(n, dtype=bool)
        
        # 方案D: 密集区高低点突破
        adhesion_high = np.full(n, np.nan)
        adhesion_low = np.full(n, np.nan)
        breakout_above_range = np.zeros(n, dtype=bool)
        breakout_below_range = np.zeros(n, dtype=bool)
        
        # 状态变量
        prev_was_bull = False
        prev_was_bear = False
        prev_adhesion_high = np.nan
        prev_adhesion_low = np.nan
        prev_is_adhesion = False
        
        is_adhesion = df['is_adhesion'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        sma20 = df['SMA20'].values
        
        for i in range(n):
            # 方案A: 记录粘合状态
            if is_adhesion[i]:
                prev_was_bull = True
                prev_was_bear = True
            
            was_adhesion_bull[i] = prev_was_bull
            was_adhesion_bear[i] = prev_was_bear
            
            # 粘合结束后的首次突破
            if prev_was_bull and not is_adhesion[i] and not np.isnan(sma20[i]) and close[i] > sma20[i]:
                adhesion_breakout_up[i] = True
                prev_was_bull = False
                
            if prev_was_bear and not is_adhesion[i] and not np.isnan(sma20[i]) and close[i] < sma20[i]:
                adhesion_breakout_down[i] = True
                prev_was_bear = False
            
            # 方案D: 记录粘合期间高低点
            if is_adhesion[i]:
                if np.isnan(prev_adhesion_high):
                    prev_adhesion_high = high[i]
                    prev_adhesion_low = low[i]
                else:
                    prev_adhesion_high = max(prev_adhesion_high, high[i])
                    prev_adhesion_low = min(prev_adhesion_low, low[i])
            else:
                # 粘合结束后重置
                if not prev_is_adhesion:
                    prev_adhesion_high = np.nan
                    prev_adhesion_low = np.nan
            
            adhesion_high[i] = prev_adhesion_high
            adhesion_low[i] = prev_adhesion_low
            
            # 突破密集区高低点
            if not np.isnan(prev_adhesion_high) and close[i] > prev_adhesion_high:
                breakout_above_range[i] = True
            if not np.isnan(prev_adhesion_low) and close[i] < prev_adhesion_low:
                breakout_below_range[i] = True
            
            prev_is_adhesion = is_adhesion[i]
        
        df['adhesion_breakout_up'] = adhesion_breakout_up
        df['adhesion_breakout_down'] = adhesion_breakout_down
        df['adhesion_high'] = adhesion_high
        df['adhesion_low'] = adhesion_low
        df['breakout_above_range'] = breakout_above_range
        df['breakout_below_range'] = breakout_below_range
        
        return df
    
    def check_signal(self, df: pd.DataFrame, idx: int) -> Tuple[bool, bool]:
        """
        检查指定索引位置是否满足最终信号条件
        
        Args:
            df: 已计算指标的 DataFrame
            idx: 要检查的行索引
            
        Returns:
            (is_long_signal, is_short_signal) 元组
        """
        cfg = self.config
        row = df.iloc[idx]
        
        # ========== 基础过滤（A+B+D方案组合）==========
        cross_up = row['cross_up_1'] or row.get('adhesion_breakout_up', False)
        cross_dn = row['cross_dn_1'] or row.get('adhesion_breakout_down', False)
        
        breakout_up = row.get('breakout_above_range', False) or row['cross_up_1']
        breakout_dn = row.get('breakout_below_range', False) or row['cross_dn_1']
        
        # 六均线过滤
        above_count = sum([
            self._is_candle_above(row, 'SMA20'),
            self._is_candle_above(row, 'SMA60'),
            self._is_candle_above(row, 'SMA120'),
            self._is_candle_above(row, 'EMA20'),
            self._is_candle_above(row, 'EMA60'),
            self._is_candle_above(row, 'EMA120'),
        ])
        
        below_count = sum([
            self._is_candle_below(row, 'SMA20'),
            self._is_candle_below(row, 'SMA60'),
            self._is_candle_below(row, 'SMA120'),
            self._is_candle_below(row, 'EMA20'),
            self._is_candle_below(row, 'EMA60'),
            self._is_candle_below(row, 'EMA120'),
        ])
        
        if cfg.use_strict_filter:
            is_above_enough = above_count >= 6
            is_below_enough = below_count >= 6
        else:
            is_above_enough = above_count >= cfg.min_ma_confirm
            is_below_enough = below_count >= cfg.min_ma_confirm
        
        filtered_cross_up = cross_up and is_above_enough and breakout_up
        filtered_cross_dn = cross_dn and is_below_enough and breakout_dn
        
        # ========== 动能过滤 ==========
        if cfg.use_osc_filter:
            osc_up_ok = self._has_osc_window(df, idx, cfg.osc_threshold, 'up')
            osc_dn_ok = self._has_osc_window(df, idx, cfg.osc_threshold, 'down')
        else:
            osc_up_ok = True
            osc_dn_ok = True
        
        # ========== 均线排列过滤 ==========
        if cfg.use_alignment_filter:
            alignment_long = row['is_bullish_alignment']
            alignment_short = row['is_bearish_alignment']
        else:
            alignment_long = True
            alignment_short = True
        
        # ========== K线力度过滤 ==========
        if cfg.use_candle_power:
            candle_range = row['high'] - row['low']
            if candle_range > 0:
                close_from_low = row['close'] - row['low']
                close_from_high = row['high'] - row['close']
                power_long = (close_from_low / candle_range * 100) >= cfg.power_ratio
                power_short = (close_from_high / candle_range * 100) >= cfg.power_ratio
            else:
                power_long = False
                power_short = False
        else:
            power_long = True
            power_short = True
        
        # ========== 最终信号 ==========
        final_long = filtered_cross_up and osc_up_ok and alignment_long and power_long
        final_short = filtered_cross_dn and osc_dn_ok and alignment_short and power_short
        
        return final_long, final_short
    
    def _crossover(self, series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """Pine Script ta.crossover: a > b AND a[1] <= b[1]"""
        return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))
    
    def _crossunder(self, series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """Pine Script ta.crossunder: a < b AND a[1] >= b[1]"""
        return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))
    
    def _is_candle_above(self, row: pd.Series, ma_col: str) -> bool:
        """检查K线是否在均线上方"""
        ma_val = row[ma_col]
        if pd.isna(ma_val):
            return False
        return row['open'] > ma_val and row['close'] > ma_val
    
    def _is_candle_below(self, row: pd.Series, ma_col: str) -> bool:
        """检查K线是否在均线下方"""
        ma_val = row[ma_col]
        if pd.isna(ma_val):
            return False
        return row['open'] < ma_val and row['close'] < ma_val
    
    def _has_osc_window(self, df: pd.DataFrame, idx: int, threshold: float, direction: str) -> bool:
        """检查最近 N 根K线内是否有动能突破阈值"""
        cfg = self.config
        start = max(0, idx - cfg.osc_confirm_bars)
        end = idx + 1
        
        osc_values = df['osc'].iloc[start:end]
        
        if direction == 'up':
            return (osc_values >= threshold).any()
        else:
            return (osc_values <= -threshold).any()


def detect_signals_in_window(df: pd.DataFrame, config: SignalConfig = None) -> Tuple[bool, bool]:
    """
    对整个窗口数据进行信号检测，返回最后一根K线的信号状态
    这是滑动窗口模式下的主入口函数
    
    Args:
        df: 窗口数据 DataFrame，包含足够的历史数据用于指标计算
        config: 信号检测配置
        
    Returns:
        (is_long_signal, is_short_signal) 元组，表示最后一根K线的信号
    """
    detector = PineSignalDetector(config)
    
    # 计算指标
    df = detector.calculate_indicators(df)
    df = detector.calculate_stateful_signals(df)
    
    # 检查最后一根K线
    last_idx = len(df) - 1
    
    # 确保有足够数据
    if df['SMA120'].isna().iloc[last_idx]:
        return False, False
    
    return detector.check_signal(df, last_idx)


if __name__ == "__main__":
    # 简单测试
    print("Pine Signal Detector - Test")
    
    # 创建测试数据
    import numpy as np
    np.random.seed(42)
    
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + np.abs(np.random.randn(n) * 0.3),
        'low': close - np.abs(np.random.randn(n) * 0.3),
        'close': close,
    })
    
    detector = PineSignalDetector()
    df = detector.calculate_indicators(df)
    df = detector.calculate_stateful_signals(df)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 检查信号
    for i in range(150, 200):
        is_long, is_short = detector.check_signal(df, i)
        if is_long or is_short:
            signal_type = "LONG" if is_long else "SHORT"
            print(f"Signal at index {i}: {signal_type}")
