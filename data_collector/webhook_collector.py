import os, time, subprocess, json, mss, mss.tools
from flask import Flask, request
from queue import Queue
from threading import Thread
from datetime import datetime, timedelta

app = Flask(__name__)

# --- 针对你的目录结构配置绝对路径 ---
# 脚本在 data_collector/，所以根目录是它的上一级
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 对应你截图中的 datasets/raw_signals
SAVE_PATH = os.path.join(BASE_DIR, "datasets", "raw_signals") 
# 对应你截图中的根目录 tasks.json
JSON_PATH = os.path.join(BASE_DIR, "tasks.json")

os.makedirs(SAVE_PATH, exist_ok=True)

signal_queue = Queue()
DELAY_SECONDS = 30 

def update_tasks_json(pending_tasks):
    tasks_data = [
        {
            "symbol": t["symbol"], 
            "run_at": t["run_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "received_at": t["received_at"],
            "status": "Waiting"
        }
        for t in pending_tasks
    ]
    with open(JSON_PATH, "w") as f:
        json.dump(tasks_data, f, indent=4)
    print(f"📁 已同步至: {JSON_PATH}")

def switch_and_capture(symbol):
    try:
        applescript = f'''
        tell application "TradingView" to activate
        delay 0.5
        tell application "System Events"
            keystroke "{symbol}"
            delay 0.5
            key code 36
        end tell
        '''
        subprocess.run(["osascript", "-e", applescript])
        time.sleep(2.0)
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            filename = os.path.join(SAVE_PATH, f"{symbol}_{time.strftime('%H%M%S')}.png")
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=filename)
            print(f"✅ 截图存入: {filename}")
    except Exception as e:
        print(f"❌ 截图错误: {e}")

def worker():
    pending_tasks = []
    print(f"🤖 引擎启动 | 监听根目录: {JSON_PATH}")
    while True:
        while not signal_queue.empty():
            data = signal_queue.get()
            symbol = data.get('symbol', 'UNK')
            run_at = datetime.now() + timedelta(seconds=DELAY_SECONDS)
            received_at = datetime.now().strftime("%H:%M:%S")
            pending_tasks.append({"symbol": symbol, "run_at": run_at, "received_at": received_at})
            update_tasks_json(pending_tasks)

        now = datetime.now()
        original_len = len(pending_tasks)
        for task in pending_tasks[:]:
            if now >= task['run_at']:
                switch_and_capture(task['symbol'])
                pending_tasks.remove(task)
        
        if len(pending_tasks) != original_len:
            update_tasks_json(pending_tasks)
        time.sleep(1)

Thread(target=worker, daemon=True).start()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if data:
        signal_queue.put(data)
        return "OK", 200
    return "No Data", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)