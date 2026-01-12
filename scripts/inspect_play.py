import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('playusdt_3m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Calculate MAs
# User mentioned "Double MA" (SMA+EMA) and "ChartPrime" (SMA40). 
# Let's calculate a mix to see what matches the image.
for span in [20, 40, 60, 120]:
    df[f'SMA{span}'] = df['close'].rolling(span).mean()
    df[f'EMA{span}'] = df['close'].ewm(span=span, adjust=False).mean()

# Target Time (UTC for 22:42 UTC+8)
target_time = '2026-01-12 14:42:00'

# Print context
try:
    idx = df.index.get_loc(target_time)
    start = max(0, idx - 2)
    end = min(len(df), idx + 3)
    
    subset = df.iloc[start:end]
    print(f"--- Data around {target_time} ---")
    print(subset[['close', 'SMA20', 'SMA40', 'SMA60', 'SMA120', 'EMA20', 'EMA60']])
    
    # Calculate spreads/crossovers at target
    row = df.loc[target_time]
    print("\n--- Specific Values at Target ---")
    print(f"Close: {row['close']}")
    print(f"SMA20: {row['SMA20']}")
    print(f"SMA60: {row['SMA60']}")
    print(f"SMA120: {row['SMA120']}")
    print(f"EMA20: {row['EMA20']}")
    print(f"EMA60: {row['EMA60']}")
    print(f"EMA120: {row['EMA120']}")
    
    # Check for tightness
    mas = [row['SMA20'], row['SMA60'], row['SMA120'], row['EMA20'], row['EMA60']]
    spread = (max(mas) - min(mas)) / np.mean(mas)
    print(f"Spread (Max-Min)/Mean: {spread:.5f} ({spread*100:.2f}%)")
    
except KeyError:
    print(f"Time {target_time} not found in data.")
    print(f"Range: {df.index[0]} to {df.index[-1]}")
