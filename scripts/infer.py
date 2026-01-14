"""
YOLO 模型推理脚本
功能：
使用训练好的 YOLO 模型对新图像进行目标检测。

主要功能：
1. 加载模型权重
2. 设置置信度阈值
3. 对指定图像或目录进行推理
4. 显示或保存检测结果
"""

from ultralytics import YOLO
import os
import sys
from pathlib import Path

# =================配置=================
# 模型路径（训练完成后把最佳权重路径填在这里）
# 默认路径: runs/detect/kline_cluster_yolo11/weights/best.pt
MODEL_PATH = 'runs/detect/kline_cluster_yolo11/weights/best.pt'

# 测试图片路径（可以是单张图片，也可以是文件夹）
TEST_SOURCE = "data/pine_signals/images" 

# 置信度阈值 (0.0 - 1.0)
CONF_THRESHOLD = 0.25 
# =====================================

def infer():
    """执行推理"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("   请先运行 train_yolo.py 完成训练。")
        return

    print(f"🚀 加载模型: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    print(f"🔍 开始推理 (源: {TEST_SOURCE}, 置信度: {CONF_THRESHOLD})...")
    
    # 执行预测
    # save=True: 保存带标注的图片到 runs/detect/predict
    # conf: 置信度阈值
    results = model.predict(
        source=TEST_SOURCE, 
        save=True, 
        conf=CONF_THRESHOLD,
        project="runs/detect",
        name="inference_results",
        exist_ok=True
    )
    
    print(f"✅ 推理完成！")
    print(f"   结果已保存至: runs/detect/inference_results")
    
    # 打印一些统计信息
    count = 0
    for res in results:
        if len(res.boxes) > 0:
            count += 1
            
    print(f"   在 {len(results)} 张图片中，有 {count} 张检测到了目标。")

if __name__ == "__main__":
    # 如果命令行传入了参数，则使用命令行参数作为图片路径
    if len(sys.argv) > 1:
        TEST_SOURCE = sys.argv[1]
        
    infer()