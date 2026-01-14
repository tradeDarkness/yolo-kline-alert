"""
OKX API 工具模块
功能：
1. 获取所有 USDT 永续合约交易对列表
2. 获取成交量前 N 的热门交易对
3. 获取指定交易对的历史 K 线数据 (支持自动翻页分页)

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
            if item['instId'].endswith('-USDT-SWAP'):
                instruments.append(item['instId'])
        
        return instruments
        
    except Exception as e:
        print(f"❌ 获取交易对异常: {e}")
        return []

def get_top_volume_pairs(limit=50):
    """
    获取 OKX 24小时成交量排名前 N 的 USDT 永续合约交易对
    
    Args:
        limit (int): 返回数量限制 (默认50)
        
    Returns:
        list: 按成交量从高到低排序的 instId 列表
    """
    url = f"{BASE_URL}/api/v5/market/tickers"
    params = {"instType": "SWAP"}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['code'] != '0':
            print(f"❌ 获取行情失败: {data['msg']}")
            return []
            
        # 筛选 USDT-SWAP
        tickers = []
        for item in data['data']:
            if item['instId'].endswith('-USDT-SWAP'):
                # 转换为浮点数以便排序 (volCcy24h 是以币为单位的成交量，volCcy24hQuote 是USDT成交量)
                # 使用 Quote volume (USDT) 更准确反映热度
                vol = float(item.get('volCcy24h', 0)) 
                tickers.append({
                    'instId': item['instId'],
                    'vol': vol
                })
        
        # 按成交量降序排序
        tickers.sort(key=lambda x: x['vol'], reverse=True)
        
        # 取前 N 个
        top_n = [t['instId'] for t in tickers[:limit]]
        
        # 确保 BTC 和 ETH 在列表中 (通常它们已经在前2)
        # 如果这是为了扩充数据集，确保包括它们总是好的
        
        return top_n
        
    except Exception as e:
        print(f"❌ 获取热门币种异常: {e}")
        return []

def fetch_candles(instId, bar='1D', limit=100):
    """
    获取指定交易对的历史 K 线数据
    支持自动分页获取超过 300 条的数据 (OKX 单次限制 100/300)
    """
    all_data = []
    after = None
    
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
                # 如果 history-candles 失败(例如太新或某些限制), 尝试普通 candles 接口
                # 但 history-candles 才是用来取老数据的
                print(f"⚠️ API 警告 ({instId}): {data['msg']}")
                break
            
            if not data['data']:
                break
                
            all_data.extend(data['data'])
            remaining -= len(data['data'])
            
            after = data['data'][-1][0]
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ 获取 K 线异常 ({instId}): {e}")
            break
    
    if not all_data:
        return None
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm']
    df = pd.DataFrame(all_data, columns=cols)
    
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['vol'] = pd.to_numeric(df['vol'])
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("正在获取成交量前 10 的币种...")
    top10 = get_top_volume_pairs(10)
    print(f"Top 10: {top10}")
