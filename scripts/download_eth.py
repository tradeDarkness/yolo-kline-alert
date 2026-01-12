import requests
import pandas as pd
import time
import sys
from datetime import datetime, timedelta

def download_klines(symbol="ETHUSDT", interval="3m", days=365):
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1500
    
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    
    print(f"⬇️ 开始下载 {symbol} {interval} 数据 (最近 {days} 天)...")
    
    current_start = start_time
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit
        }
        
        try:
            resp = requests.get(base_url, params=params)
            data = resp.json()
            
            if not isinstance(data, list):
                print(f"❌ API Error: {data}")
                break
                
            if len(data) == 0:
                print("⚠️ No more data.")
                break
                
            all_data.extend(data)
            print(f"   已获取 {len(data)} 条 (Total: {len(all_data)})... Last: {datetime.fromtimestamp(data[-1][0]/1000)}")
            
            last_timestamp = data[-1][0]
            if last_timestamp >= end_time - (3 * 60 * 1000):
                break
                
            current_start = last_timestamp + 1
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            break
            
    if not all_data:
        return
        
    print(f"📊 正在处理 {len(all_data)} 条数据...")
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    filename = f"{symbol.lower()}_{interval}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ 保存成功: {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        download_klines(symbol=symbol, days=2)
    else:
        # Default fallback
        download_klines()
