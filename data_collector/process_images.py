import os
from PIL import Image

RAW_DIR = "datasets/raw_signals"
CROP_DIR = "datasets/processed"

if not os.path.exists(CROP_DIR):
    os.makedirs(CROP_DIR)

# 这里的百分比坐标需要根据你的 TradingView 实际布局微调
# 假设我们要切掉左侧 5% 的工具栏，右侧 10% 的价格轴，顶部 10% 的菜单
def crop_chart(image_path, filename):
    with Image.open(image_path) as img:
        width, height = img.size
        
        # 裁剪参数：(左, 上, 右, 下)
        # 建议：左侧工具栏 (80像素), 顶部菜单 (100像素), 右侧价格 (120像素), 底部信息 (50像素)
        left = 80
        top = 100
        right = width - 120
        bottom = height - 80
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(os.path.join(CROP_DIR, filename))
        print(f"✨ 已处理: {filename}")

for file in os.listdir(RAW_DIR):
    if file.endswith(".png"):
        crop_chart(os.path.join(RAW_DIR, file), file)