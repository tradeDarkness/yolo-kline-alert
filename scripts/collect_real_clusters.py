import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- Config ---
DATA_FILE = 'eth_3m.csv'
BASE_DIR = 'data/ready_to_label'
UP_DIR = os.path.join(BASE_DIR, 'UP')
DN_DIR = os.path.join(BASE_DIR, 'DOWN')
NEG_DIR = os.path.join(BASE_DIR, 'NEGATIVE')

for d in [UP_DIR, DN_DIR, NEG_DIR]:
    os.makedirs(d, exist_ok=True)

# Threshold: ~0.6% Spread
SPREAD_THRESHOLD = 0.006 
WINDOW_SIZE = 120
STEP = 30 # Sampling step
TARGET_COUNT = 300 # Per class

# Colors (Adjusted for Light Mode/White BG)
# SMA (Warm)
COL_SMA20 = '#FBC02D'  # Darker Yellow
COL_SMA60 = '#FF9800'  # Orange
COL_SMA120 = '#D32F2F' # Red
# EMA (Cool) 
COL_EMA20 = '#0097A7'  # Darker Cyan/Teal
COL_EMA60 = '#1976D2'  # Blue
COL_EMA120 = '#7B1FA2' # Purple
# Candles
COL_CANDLE_UP = '#00897B' # Teal
COL_CANDLE_DN = '#FF9800' # Orange

def load_data(filepath):
    print(f"⏳ Loading {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def calculate_ma(df):
    for span in [20, 60, 120]:
        df[f'SMA{span}'] = df['close'].rolling(span).mean()
        df[f'EMA{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def find_clusters(df):
    print("⏳ Scanning for 6-MA Clusters...")
    cols = ['SMA20', 'SMA60', 'SMA120', 'EMA20', 'EMA60', 'EMA120']
    
    ma_max = df[cols].max(axis=1)
    ma_min = df[cols].min(axis=1)
    ma_mean = df[cols].mean(axis=1)
    spread = (ma_max - ma_min) / ma_mean
    
    candidates = np.where(spread < SPREAD_THRESHOLD)[0]
    print(f"🔍 Found {len(candidates)} raw points.")
    
    # Simple filtering for independence
    independent_candidates = []
    last_idx = -999
    
    # Use all data chronologically first to filter, then shuffle later
    for idx in candidates:
        if idx - last_idx < STEP: continue
        if idx < WINDOW_SIZE//2 or idx > len(df) - WINDOW_SIZE - 20: continue # Ensure room for future
        independent_candidates.append(idx)
        last_idx = idx
        
    print(f"✅ Independent samples: {len(independent_candidates)}")
    return independent_candidates

def find_negatives(df, clusters, count=300):
    print("⏳ Scanning for Negative Samples...")
    import random
    # Create a set of "forbidden" indices (around clusters)
    forbidden = set()
    for c in clusters:
        for i in range(c - WINDOW_SIZE, c + WINDOW_SIZE):
            forbidden.add(i)
            
    # Sample valid indices
    valid_candidates = []
    # Optimization: Scan with step to avoid checking every single index
    for i in range(WINDOW_SIZE, len(df) - WINDOW_SIZE, STEP):
        if i not in forbidden:
            valid_candidates.append(i)
            
    random.shuffle(valid_candidates)
    return valid_candidates[:count]

def check_outcome(df, idx):
    """Determine if price went UP or DOWN after the cluster."""
    # Look ahead 20 bars (1 hour)
    current_close = df.iloc[idx]['close']
    future_max = df.iloc[idx+1:idx+21]['high'].max()
    future_min = df.iloc[idx+1:idx+21]['low'].min()
    future_close = df.iloc[idx+20]['close']
    
    # Simple Rule: Return > 0.5% = UP, < -0.5% = DOWN
    ret = (future_close - current_close) / current_close
    
    if ret > 0.005: return 'UP'
    if ret < -0.005: return 'DOWN'
    return 'SIDEWAYS'

def save_chart(df, center_idx, sample_id, label):
    # Window settings
    half_window = WINDOW_SIZE // 2
    start_idx = center_idx - half_window + 20 
    end_idx = center_idx + half_window + 20
    
    if start_idx < 0 or end_idx > len(df): return False
    
    sub_df = df.iloc[start_idx:end_idx].copy()
    
    # ---------------------------------------------------------
    # 1. Calculate Bounding Box (Only for UP/DOWN)
    # ---------------------------------------------------------
    cols = ['SMA20', 'SMA60', 'SMA120', 'EMA20', 'EMA60', 'EMA120']
    
    label_line = ""
    
    if label != 'NEGATIVE':
        ma_max = sub_df[cols].max(axis=1)
        ma_min = sub_df[cols].min(axis=1)
        ma_mean = sub_df[cols].mean(axis=1)
        spread = (ma_max - ma_min) / ma_mean
        
        local_center = center_idx - start_idx
        knot_start = local_center
        knot_end = local_center
        
        while knot_start > 0 and spread.iloc[knot_start] < SPREAD_THRESHOLD:
            knot_start -= 1
        while knot_end < len(sub_df) - 1 and spread.iloc[knot_end] < SPREAD_THRESHOLD:
            knot_end += 1
            
        if knot_end - knot_start < 4:
            pad = 2
            knot_start = max(0, local_center - pad)
            knot_end = min(len(sub_df)-1, local_center + pad)
            
        knot_data = sub_df.iloc[knot_start:knot_end+1][cols]
        box_min_price = knot_data.min().min()
        box_max_price = knot_data.max().max()
        
        price_pad = (box_max_price - box_min_price) * 0.2
        box_min_price -= price_pad
        box_max_price += price_pad
    
    # ---------------------------------------------------------
    # 2. Plotting with STRICT Coordinates
    # ---------------------------------------------------------
    
    ema120 = sub_df['EMA120'].values
    closes = sub_df['close'].values
    opens = sub_df['open'].values
    highs = sub_df['high'].values
    lows = sub_df['low'].values
    dates = range(len(sub_df)) 
    
    # Limits
    x_min, x_max = 0, len(sub_df) - 1
    y_min_data = min(sub_df['low'].min(), sub_df[cols].min().min())
    y_max_data = max(sub_df['high'].max(), sub_df[cols].max().max())
    y_pad = (y_max_data - y_min_data) * 0.05
    y_min_limit = y_min_data - y_pad
    y_max_limit = y_max_data + y_pad
    
    # Create Figure with Exact Dimensions
    fig = plt.figure(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('white')
    
    # Add axes covering the WHOLE figure (0,0,1,1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')
    
    # Set Limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_limit, y_max_limit)
    
    # Plot MAs
    ax.plot(dates, sub_df['SMA20'], color=COL_SMA20, linewidth=1.0, alpha=0.9)
    ax.plot(dates, sub_df['SMA60'], color=COL_SMA60, linewidth=1.0, alpha=0.9)
    ax.plot(dates, sub_df['SMA120'], color=COL_SMA120, linewidth=1.0, alpha=0.9)
    
    ax.plot(dates, sub_df['EMA20'], color=COL_EMA20, linewidth=1.0, alpha=0.9)
    ax.plot(dates, sub_df['EMA60'], color=COL_EMA60, linewidth=1.0, alpha=0.9)
    ax.plot(dates, sub_df['EMA120'], color=COL_EMA120, linewidth=1.0, alpha=0.9)
    
    # Candles
    width = 0.6
    width2 = width / 2
    
    for i, (d, o, c, h, l, e120) in enumerate(zip(dates, opens, closes, highs, lows, ema120)):
        color = COL_CANDLE_UP if c >= e120 else COL_CANDLE_DN
        ax.plot([d, d], [l, h], color=color, linewidth=1.0)
        lower = min(o, c)
        height = abs(c - o)
        if height == 0: height = (h-l)*0.01
        rect = patches.Rectangle((d - width2, lower), width, height, linewidth=0, edgecolor=None, facecolor=color)
        ax.add_patch(rect)

    ax.axis('off')
    
    # ---------------------------------------------------------
    # 3. Label Generation
    # ---------------------------------------------------------
    if label != 'NEGATIVE':
        def normalize_x(val):
            return (val - x_min) / (x_max - x_min)
            
        def normalize_y(val):
            # Normalization with respect to plot limits
            norm = (val - y_min_limit) / (y_max_limit - y_min_limit)
            return 1.0 - norm # Invert for YOLO
        
        # X
        nx1 = normalize_x(knot_start)
        nx2 = normalize_x(knot_end)
        
        # Y
        ny_top = normalize_y(box_max_price) # Smaller value in image coords
        ny_bottom = normalize_y(box_min_price) # Larger value in image coords
        
        w = abs(nx2 - nx1)
        h = abs(ny_bottom - ny_top)
        cx = (nx1 + nx2) / 2
        cy = (ny_top + ny_bottom) / 2
        
        # Clamp
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        label_line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    
    # ---------------------------------------------------------
    # 4. Save
    # ---------------------------------------------------------
    ts_str = df.index[center_idx].strftime('%Y%m%d_%H%M')
    target_dir = UP_DIR if label == 'UP' else (DN_DIR if label == 'DOWN' else NEG_DIR)
    
    img_name = f"{label}_{ts_str}.png"
    txt_name = f"{label}_{ts_str}.txt"
    
    img_path = os.path.join(target_dir, img_name)
    txt_path = os.path.join(target_dir, txt_name)
    
    try:
        with open(txt_path, 'w') as f:
            if label_line:
                f.write(label_line + "\n")
            # Empty file for NEGATIVE
            
        # IMPORTANT: No bbox_inches='tight', no pad_inches
        # Just simple savefig. The axes fill the figure 100%.
        plt.savefig(img_path, facecolor='white', dpi=100)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error: {e}")
        plt.close(fig)
        return False

def main():
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found!")
        return

    df = load_data(DATA_FILE)
    df = calculate_ma(df)
    clusters = find_clusters(df)
    
    # Shuffle for randomness
    import random
    random.shuffle(clusters)
    
    # 1. Generate Positive Samples
    count_up = 0
    count_dn = 0
    
    print(f"🚀 Generating {TARGET_COUNT} UP / {TARGET_COUNT} DOWN samples...")
    
    for idx in clusters:
        if count_up >= TARGET_COUNT and count_dn >= TARGET_COUNT:
            break
            
        label = check_outcome(df, idx)
        
        if label == 'UP' and count_up < TARGET_COUNT:
            if save_chart(df, idx, count_up, 'UP'):
                count_up += 1
                if count_up % 50 == 0: print(f"Saved UP: {count_up}")
                
        elif label == 'DOWN' and count_dn < TARGET_COUNT:
            if save_chart(df, idx, count_dn, 'DOWN'):
                count_dn += 1
                if count_dn % 50 == 0: print(f"Saved DOWN: {count_dn}")
                
    # 2. Generate Negative Samples
    print(f"🚀 Generating {TARGET_COUNT} NEGATIVE samples...")
    negatives = find_negatives(df, clusters, TARGET_COUNT)
    count_neg = 0
    for idx in negatives:
        if save_chart(df, idx, count_neg, 'NEGATIVE'):
            count_neg += 1
            if count_neg % 50 == 0: print(f"Saved NEGATIVE: {count_neg}")
            
    print(f"Done! UP: {count_up}, DOWN: {count_dn}, NEG: {count_neg}")
    print(f"Check {BASE_DIR}")

if __name__ == "__main__":
    main()
