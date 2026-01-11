#!/usr/bin/env python3
"""
分析检测结果 - 找出最佳置信度阈值
"""
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def analyze_detections():
    """分析不同置信度阈值下的检测结果"""
    
    # 找到最新的模型
    model_paths = list(Path('runs/detect').glob('train*/weights/best.pt'))
    if not model_paths:
        print("❌ 未找到训练好的模型")
        return
    
    latest_model = max(model_paths, key=lambda p: p.stat().st_mtime)
    print(f"📦 使用模型: {latest_model}\n")
    
    model = YOLO(str(latest_model))
    
    # 测试图片
    test_image = Path("data/images/Snipaste_2026-01-07_23-04-22.png")
    if not test_image.exists():
        images = list(Path("data/images").glob("*.png"))
        if images:
            test_image = images[0]
        else:
            print("❌ 未找到测试图片")
            return
    
    print(f"🔍 分析图片: {test_image}\n")
    
    # 先用极低阈值获取所有检测结果
    results = model.predict(str(test_image), conf=0.01, verbose=False)
    boxes = results[0].boxes
    
    if boxes is None or len(boxes) == 0:
        print("❌ 未检测到任何目标")
        return
    
    # 获取所有置信度
    confidences = [float(box.conf[0]) for box in boxes]
    confidences.sort(reverse=True)
    
    print("="*60)
    print("📊 检测结果分析")
    print("="*60)
    print(f"总检测数: {len(confidences)}")
    print(f"最高置信度: {max(confidences):.2%}")
    print(f"最低置信度: {min(confidences):.2%}")
    print(f"平均置信度: {np.mean(confidences):.2%}")
    print(f"中位数置信度: {np.median(confidences):.2%}")
    
    print("\n" + "="*60)
    print("📈 置信度分布（Top 20）:")
    print("="*60)
    for i, conf in enumerate(confidences[:20], 1):
        print(f"  {i:2d}. {conf:.2%}")
    
    print("\n" + "="*60)
    print("🎯 不同阈值下的检测数量:")
    print("="*60)
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for thresh in thresholds:
        count = sum(1 for c in confidences if c >= thresh)
        print(f"  阈值 {thresh:.2f} ({thresh*100:.0f}%): {count:3d} 个检测")
    
    print("\n" + "="*60)
    print("💡 建议:")
    print("="*60)
    
    # 分析建议
    max_conf = max(confidences)
    if max_conf < 0.1:
        print("⚠️  最高置信度 < 10%，说明模型训练可能不充分")
        print("   建议:")
        print("   1. 增加训练轮数（epochs）")
        print("   2. 增加训练数据量")
        print("   3. 检查标注质量")
    elif max_conf < 0.3:
        print("⚠️  最高置信度 < 30%，模型效果一般")
        print("   建议:")
        print("   1. 可以尝试阈值 0.02-0.05 来过滤误报")
        print("   2. 考虑重新训练或增加数据")
    else:
        print("✅ 模型置信度较好")
        print("   建议阈值: 0.1-0.3")
    
    # 推荐阈值
    if len(confidences) > 0:
        # 找到能保留前10%检测的阈值
        top_10_percent = int(len(confidences) * 0.1)
        if top_10_percent > 0:
            recommended_thresh = confidences[min(top_10_percent, len(confidences)-1)]
            print(f"\n💡 推荐阈值: {recommended_thresh:.3f} (保留置信度最高的 {top_10_percent} 个检测)")
    
    print("="*60)

if __name__ == "__main__":
    analyze_detections()
