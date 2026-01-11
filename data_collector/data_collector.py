# from flask import Flask, request
# from mss import mss
# import time
# import os

# app = Flask(__name__)

# # 创建保存数据集的文件夹
# SAVE_PATH = "datasets/raw_signals"
# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)

# @app.route('/webhook', methods=['POST'])
# def webhook():
#     data = request.json
#     if data:
#         print(f"🚩 收到信号: {data['symbol']} 价格: {data['price']}")
        
#         # 触发截图
#         capture_screen(data['symbol'])
#         return "Signal Received", 200
#     return "No Data", 400

# def capture_screen(symbol):
#     with mss() as sct:
#         # 设定截图文件名（币种+时间戳）
#         timestamp = int(time.time())
#         filename = f"{SAVE_PATH}/{symbol}_{timestamp}.png"
        
#         # 截取全屏（或者指定 TradingView 窗口区域）
#         # monitor = sct.monitors[1] # 全屏
#         # 如果你想只截取特定窗口，需要根据你的屏幕分辨率调整以下坐标
#         monitor = {"top": 100, "left": 100, "width": 1600, "height": 900}
        
#         sct_img = sct.grab(monitor)
#         import mss.tools
#         mss.tools.to_png(sct_img.rgb, sct_img.size, output=filename)
#         print(f"📸 截图已保存: {filename}")

# if __name__ == '__main__':
#     # 在本地 5000 端口启动
#     # 注意：如果 TradingView 在云端，你需要用 ngrok 将本地 5000 映射到公网
#     app.run(port=5000)