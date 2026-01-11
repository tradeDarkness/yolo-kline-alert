from ultralytics import YOLO
import os
from pathlib import Path

# 模型路径（使用最新的训练模型）
MODEL_PATH = 'runs/detect/train4/weights/best.pt'

# 测试图片路径（可以指定单张图片或目录）
IMAGE_PATH = "data/images"  # 可以改为单张图片路径，如 "data/images/Snipaste_2026-01-07_23-04-22.png"

# 置信度阈值
CONF_THRESHOLD = 0.25  # 可以调整，0.01 表示1%以上置信度就显示（用于测试）

def test_single_image(model, image_path, conf_threshold=0.3):
    """测试单张图片"""
    print(f"\n{'='*60}")
    print(f"🔍 检测图片: {image_path}")
    print(f"📊 置信度阈值: {conf_threshold}")
    print(f"{'='*60}")
    
    # 执行预测
    results = model.predict(
        image_path, 
        save=True, 
        conf=conf_threshold,
        save_txt=False,  # 不保存标注文件
        save_conf=True,  # 保存置信度
        line_width=2
    )
    
    # 解析结果
    result = results[0]
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        print(f"✅ 检测到 {len(boxes)} 个均线密集形态:")
        for i, box in enumerate(boxes, 1):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            print(f"   {i}. 置信度: {conf:.2%} | 位置: ({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")
        
        # 保存路径
        save_path = result.save_dir
        print(f"📸 结果已保存到: {save_path}")
    else:
        print("❌ 未检测到均线密集形态")
    
    return results

def test_multiple_images(model, image_dir, conf_threshold=0.3, max_images=5):
    """测试多张图片"""
    image_dir = Path(image_dir)
    image_extensions = ['.png', '.jpg', '.jpeg']
    
    # 获取所有图片
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
    
    if not image_files:
        print(f"❌ 在 {image_dir} 中未找到图片文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图片，测试前 {min(max_images, len(image_files))} 张")
    
    total_detections = 0
    for i, img_path in enumerate(image_files[:max_images], 1):
        print(f"\n[{i}/{min(max_images, len(image_files))}] ", end="")
        results = test_single_image(model, str(img_path), conf_threshold)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            total_detections += len(results[0].boxes)
    
    print(f"\n{'='*60}")
    print(f"📊 测试总结: 共检测到 {total_detections} 个形态")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 加载模型
    print(f"🚀 加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("💡 请先运行 train.py 训练模型")
        exit(1)
    
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功\n")
    
    # 判断是单张图片还是目录
    image_path = Path(IMAGE_PATH)
    if image_path.is_file():
        # 单张图片
        test_single_image(model, IMAGE_PATH, CONF_THRESHOLD)
    elif image_path.is_dir():
        # 目录，测试多张
        test_multiple_images(model, IMAGE_PATH, CONF_THRESHOLD, max_images=5)
    else:
        print(f"❌ 路径不存在: {IMAGE_PATH}")
        print("💡 请检查路径是否正确")