from binance.client import Client
import pandas as pd
import time

client = Client()

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_5MINUTE

start_ts = "2023-01-01"

klines = []
start_time = start_ts
total = 0

while True:
    data = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_time,
        limit=1000
    )

    if not data:
        break

    klines.extend(data)
    total += len(data)
    print(f"已拉取 {total} 根 K 线")

    start_time = data[-1][0] + 1
    time.sleep(0.3)

df = pd.DataFrame(klines, columns=[
    "timestamp","open","high","low","close","volume",
    "close_time","quote_volume","trades",
    "taker_base","taker_quote","ignore"
])

df = df[["timestamp","open","high","low","close","volume"]]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df[["open","high","low","close","volume"]] = df[
    ["open","high","low","close","volume"]
].astype(float)

df.to_csv("btc_5m.csv", index=False)

print("✅ 下载完成，已生成 btc_5m.csv")
