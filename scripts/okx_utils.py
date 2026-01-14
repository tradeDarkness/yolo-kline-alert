"""
OKX API 工具模块
功能：
1. 获取所有 USDT 永续合约交易对列表
2. 获取指定交易对的历史 K 线数据 (支持自动翻页分页)

依赖：
- requests
- pandas
"""

import requests
import pandas as pd
import time
from datetime import datetime

# OKX API 基础 URL
BASE_URL = "https://www.okx.com"

def get_usdt_pairs():
    """
    获取 OKX 所有 USDT 结算的永续合约交易对 (SWAP)
    
    Returns:
        list: 交易对ID列表，例如 ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', ...]
    """
    url = f"{BASE_URL}/api/v5/public/instruments"
    params = {"instType": "SWAP"}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['code'] != '0':
            print(f"❌ 获取交易对失败: {data['msg']}")
            return []
            
        instruments = []
        for item in data['data']:
            # 只保留 USDT 结算的合约
            if item['instId'].endswith('-USDT-SWAP'):
                instruments.append(item['instId'])
        
        return instruments
        
    except Exception as e:
        print(f"❌ 获取交易对异常: {e}")
        return []

def fetch_candles(instId, bar='1D', limit=100):
    """
    获取指定交易对的历史 K 线数据
    支持自动分页获取超过 300 条的数据 (OKX 单次限制 100/300)
    
    Args:
        instId: 交易对 ID (如 'BTC-USDT-SWAP')
        bar: K 线周期 (如 '1D', '15m', '5m')
        limit: 需要获取的总记录数
        
    Returns:
        pd.DataFrame: 包含 timestamp, open, high, low, close, vol, datetime 等列的数据框
                      按时间正序排列（最早在前，最新在后）
    """
    all_data = []
    after = None  # 分页游标 (请求该时间戳之前的数据)
    
    # OKX 历史K线接口单次最大限制 100 (部分接口是300，保守起见分页处理)
    per_request = 100 
    remaining = limit
    
    while remaining > 0:
        url = f"{BASE_URL}/api/v5/market/history-candles"
        
        params = {
            "instId": instId,
            "bar": bar,
            "limit": min(per_request, remaining)
        }
        
        if after:
            params["after"] = after
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['code'] != '0':
                print(f"❌ API 错误: {data['msg']}")
                break
            
            if not data['data']:
                break
                
            all_data.extend(data['data'])
            remaining -= len(data['data'])
            
            # 获取当前页最早的一条数据的时间戳，作为下一页的 after 参数
            after = data['data'][-1][0]
            
            # API 频率限制保护
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ 获取 K 线异常 ({instId}): {e}")
            break
    
    if not all_data:
        return None
    
    # OKX 数据格式: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm']
    df = pd.DataFrame(all_data, columns=cols)
    
    # 数据类型转换
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['vol'] = pd.to_numeric(df['vol'])
    
    # 转换时间戳为 datetime 对象
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 排序：旧数据在前，新数据在后 (OKX 返回的是倒序)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # 测试代码
    print("正在获取交易对列表...")
    pairs = get_usdt_pairs()
    print(f"找到 {len(pairs)} 个 USDT-SWAP 交易对。")
    if pairs:
        sample = pairs[0]
        print(f"\n正在获取 {sample} 的 5m K线 (limit=500)...")
        df = fetch_candles(sample, bar='5m', limit=500)
        if df is not None:
            print(f"成功获取 {len(df)} 条数据")
            print(f"时间范围: {df['datetime'].iloc[0]} 到 {df['datetime'].iloc[-1]}")
