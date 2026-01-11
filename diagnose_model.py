#!/usr/bin/env python3
"""
模型诊断脚本 - 检查模型训练效果和检测能力
"""
from ultralytics import YOLO
from pathlib import Path
import sys

def diagnose_model():
    # 找到最新的模型
    model_paths = list(Path('runs/detect').glob('train*/weights/best.pt'))
    if not model_paths:
        print("❌ 未找到训练好的模型")
        return
    
    latest_model = max(model_paths, key=lambda p: p.stat().st_mtime)
    print(f"📦 使用模型: {latest_model}\n")
    
    # 加载模型
    model = YOLO(str(latest_model))
    
    # 测试图片
    test_image = Path("data/images/Snipaste_2026-01-07_23-04-22.png")
    if not test_image.exists():
        # 找第一张图片
        images = list(Path("data/images").glob("*.png"))
        if images:
            test_image = images[0]
        else:
            print("❌ 未找到测试图片")
            return
    
    print(f"🔍 测试图片: {test_image}\n")
    
    # 用不同的置信度阈值测试
    conf_thresholds = [0.01, 0.1, 0.2, 0.3, 0.5]
    
    print("="*60)
    print("🧪 使用不同置信度阈值测试:")
    print("="*60)
    
    for conf in conf_thresholds:
        print(f"\n📊 置信度阈值: {conf}")
        results = model.predict(
            str(test_image),
            conf=conf,
            save=False,  # 不保存，只测试
            verbose=False
        )
        
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            print(f"   ✅ 检测到 {len(boxes)} 个目标:")
            for i, box in enumerate(boxes, 1):
                box_conf = float(box.conf[0])
                print(f"      {i}. 置信度: {box_conf:.2%}")
        else:
            print(f"   ❌ 未检测到目标")
    
    print("\n" + "="*60)
    print("💡 建议:")
    print("   1. 如果所有阈值都检测不到，可能是模型训练不充分")
    print("   2. 如果低阈值(0.01)能检测到，说明模型有效，只是阈值太高")
    print("   3. 可以尝试重新训练，增加训练轮数或数据量")
    print("="*60)

if __name__ == "__main__":
    diagnose_model()
