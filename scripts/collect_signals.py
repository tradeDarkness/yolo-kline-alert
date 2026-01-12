import pandas as pd
import numpy as np
import mplfinance as mpf
import os

# --- Config ---
DATA_FILE = 'eth_3m.csv'
OUTPUT_DIR = 'data/collected_signals'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

WINDOW_SIZE = 120 # View window
OSC_LEN = 15
OSC_THRESHOLD = 0.17 # User input

# Colors
COLOR_SMA20 = '#000000'
COLOR_SMA60 = '#2962FF'
COLOR_SMA120 = '#8E24AA'

def calculate_wma(series, length):
    """Weighted Moving Average"""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, length):
    """Hull Moving Average"""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    
    raw_hma = 2 * wma_half - wma_full
    return calculate_wma(raw_hma, sqrt_length)

def calculate_indicators(df):
    """Port of PineScript Logic"""
    print("⏳ Calculating Indicators...")
    
    # 1. Basic MAs for Plotting (SMA + EMA)
    for span in [20, 60, 120]:
        df[f'SMA{span}'] = df['close'].rolling(span).mean()
        df[f'EMA{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # 2. Logic Source: hl2
    source = (df['high'] + df['low']) / 2
    
    # 3. Main Logic MA (SMA 60 as per default input)
    # length = input.int(60)
    # type_ma = SMA
    ma_logic = df['close'].rolling(60).mean() # Default source input.source(hl2) -> Wait, user input says source=hl2, but default in code source=input.source(hl2). 
    # Let's check PineScript again: MA = ma(source, length, type_ma). So MA is based on hl2.
    ma_main = source.rolling(60).mean()
    ma_20 = source.rolling(20).mean() # For crossover check
    
    # 4. Oscillator
    # diff = source - MA
    diff = source - ma_main
    
    # perc_r = ta.percentile_linear_interpolation(diff, 1000, 99)
    # Pandas quantile is approx similar
    perc_r = diff.rolling(1000).quantile(0.99)
    # Avoid division by zero
    perc_r = perc_r.replace(0, np.nan).fillna(method='ffill')
    
    # osc = ta.hma(ta.change(diff / perc_r, osc_len), 10)
    # change = val - val[1]
    ratio = diff / perc_r
    change_ratio = ratio.diff(OSC_LEN) # ta.change(src, len) -> src - src[len]. Wait, Pine `ta.change(src, length)` is `src - src[length]`. Code says `ta.change(..., osc_len)`.
    # osc_len = 15.
    
    osc = calculate_hma(change_ratio, 10)
    
    df['Osc'] = osc
    df['MA20_Src'] = ma_20
    df['MA60_Src'] = ma_main
    
    # 5. Signals
    # Up: crossover(ma20, ma60) and osc > 0.17 and osc > osc[1]
    # Dn: crossunder(ma20, ma60) and osc < -0.17 and osc < osc[1]
    
    # Crossover: 20 > 60 now, and 20 <= 60 prev
    crossover = (ma_20 > ma_main) & (ma_20.shift(1) <= ma_main.shift(1))
    crossunder = (ma_20 < ma_main) & (ma_20.shift(1) >= ma_main.shift(1))
    
    osc_up = (osc > OSC_THRESHOLD) & (osc > osc.shift(1))
    osc_dn = (osc < -OSC_THRESHOLD) & (osc < osc.shift(1))
    
    df['Signal_Up'] = crossover & osc_up
    df['Signal_Dn'] = crossunder & osc_dn
    
    return df

def save_chart(df, idx, signal_type):
    """Save Chart"""
    # Window
    start = idx - WINDOW_SIZE + 20 
    end = idx + 20 
    
    if start < 0 or end > len(df):
        return False
        
    sub_df = df.iloc[start:end].copy()
    if sub_df.empty:
        return False
    
    # Plotting
    plot_df = sub_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    
    mc = mpf.make_marketcolors(up='#0ecb81', down='#f6465d', edge='inherit', wick='inherit')
    s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, gridstyle='', facecolor='white')
    
    apds = [
        # SMA 
        mpf.make_addplot(plot_df['SMA20'], color=COLOR_SMA20, width=1.5),
        mpf.make_addplot(plot_df['SMA60'], color=COLOR_SMA60, width=1.5),
        mpf.make_addplot(plot_df['SMA120'], color=COLOR_SMA120, width=1.5),
        # EMA
        mpf.make_addplot(plot_df['EMA20'], color=COLOR_SMA20, width=0.8, alpha=0.4),
        mpf.make_addplot(plot_df['EMA60'], color=COLOR_SMA60, width=0.8, alpha=0.4),
        mpf.make_addplot(plot_df['EMA120'], color=COLOR_SMA120, width=0.8, alpha=0.4),
    ]
    
    # Add marker logic
    # Create Series matching sub_df index
    marker_series = pd.Series(np.nan, index=plot_df.index)
    
    # Target timestamp
    target_ts = df.index[idx]
    
    try:
        if target_ts in plot_df.index:
            if signal_type == 'UP':
                price = plot_df.loc[target_ts, 'Low'] * 0.998
                marker_color = 'green'
                marker_shape = '^'
            else:
                price = plot_df.loc[target_ts, 'High'] * 1.002
                marker_color = 'red'
                marker_shape = 'v'
            
            marker_series[target_ts] = price
            
            # Only add the plot if we have at least one valid value
            if not marker_series.isna().all():
                apds.append(mpf.make_addplot(marker_series, type='scatter', markersize=100, marker=marker_shape, color=marker_color))
        else:
             print(f"⚠️ Timestamp {target_ts} not in window?")

    except Exception as e:
        print(f"⚠️ Marker Error: {e}")
    
    ts_str = target_ts.strftime('%Y%m%d_%H%M')
    img_name = f"{signal_type}_{ts_str}.png"
    img_path = os.path.join(IMAGES_DIR, img_name)
    
    try:
        mpf.plot(plot_df, type='candle', addplot=apds, style=s,
                 savefig=dict(fname=img_path, dpi=100, bbox_inches='tight', pad_inches=0),
                 axisoff=True, tight_layout=True, volume=False)
        return True
    except Exception as e:
        print(f"Error plotting {ts_str}: {e}")
        return False

def main():
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return
        
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    # Find signals
    up_signals = np.where(df['Signal_Up'])[0]
    dn_signals = np.where(df['Signal_Dn'])[0]
    
    print(f"🔍 Found Signals -> UP: {len(up_signals)}, DOWN: {len(dn_signals)}")
    
    # Shuffle to get random samples from the whole year
    np.random.shuffle(up_signals)
    np.random.shuffle(dn_signals)
    
    count = 0
    # Process UP
    for idx in up_signals:
        if count >= 30: break
        if save_chart(df, idx, 'UP'):
            count += 1
            print(f"Saved UP {count}...")
            
    # Process DOWN
    count = 0
    for idx in dn_signals:
        if count >= 30: break
        if save_chart(df, idx, 'DOWN'):
            count += 1
            print(f"Saved DOWN {count}...")
            
    print(f"Done. Check {IMAGES_DIR}")

if __name__ == "__main__":
    main()
