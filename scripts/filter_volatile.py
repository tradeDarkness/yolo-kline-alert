import os
import json
import time
from okx_utils import get_usdt_pairs, fetch_candles

# Config
DAYS_TO_CHECK = 15
VOLATILITY_THRESHOLD = 0.05 # 5%
OUTPUT_FILE = "volatility_candidates.json"

def filter_volatile_coins():
    print("🚀 Starting volatility scan...")
    
    # 1. Get all pairs
    all_pairs = get_usdt_pairs()
    print(f"📦 Found {len(all_pairs)} pairs to check.")
    
    volatile_coins = []
    
    for i, pair in enumerate(all_pairs):
        # Rate limit friendly (OKX limit is generous but let's be safe)
        if i % 10 == 0:
            print(f"Checking {i}/{len(all_pairs)}...")
        time.sleep(0.1) 
        
        # 2. Fetch last 15 days
        # We ask for a few more to be safe
        df = fetch_candles(pair, bar='1D', limit=DAYS_TO_CHECK + 2)
        
        if df is None or df.empty:
            continue
            
        # 3. Check volatility
        # We only care about the last 15 days
        # df is sorted oldest -> newest, so take last 15
        recent_df = df.tail(DAYS_TO_CHECK)
        
        is_volatile = False
        reason = ""
        
        for index, row in recent_df.iterrows():
            open_price = row['open']
            close_price = row['close']
            
            if open_price == 0: continue
            
            change_pct = (close_price - open_price) / open_price
            
            if abs(change_pct) > VOLATILITY_THRESHOLD:
                is_volatile = True
                date_str = row['datetime'].strftime('%Y-%m-%d')
                reason = f"Date: {date_str}, Change: {change_pct*100:.2f}%"
                break
        
        if is_volatile:
            print(f"🔥 Found volatile: {pair} ({reason})")
            volatile_coins.append(pair)
            
    # 4. Save results
    print(f"✅ Scan complete. Found {len(volatile_coins)} volatile coins.")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(volatile_coins, f, indent=2)
    print(f"💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_volatile_coins()
