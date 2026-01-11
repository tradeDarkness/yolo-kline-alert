#!/usr/bin/env python3
"""
检查标注文件是否完整
"""
import os
from pathlib import Path

def check_labels():
    images_dir = Path("data/images")
    labels_dir = Path("data/labels")
    
    # 获取所有图片文件
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    print(f"📊 数据集检查报告")
    print("=" * 60)
    print(f"总图片数: {len(image_files)}")
    
    # 检查标注文件（优先检查 labels 目录，如果没有则在 images 目录）
    labeled_count = 0
    unlabeled_count = 0
    unlabeled_files = []
    
    for img_file in image_files:
        # 先检查 labels 目录
        txt_file_in_labels = labels_dir / f"{img_file.stem}.txt"
        # 再检查 images 目录（兼容两种组织方式）
        txt_file_in_images = img_file.with_suffix('.txt')
        
        if txt_file_in_labels.exists() and txt_file_in_labels.stat().st_size > 0:
            labeled_count += 1
        elif txt_file_in_images.exists() and txt_file_in_images.stat().st_size > 0:
            labeled_count += 1
        else:
            unlabeled_count += 1
            unlabeled_files.append(img_file.name)
    
    print(f"✅ 已标注: {labeled_count} 张")
    print(f"❌ 未标注: {unlabeled_count} 张")
    
    if unlabeled_files:
        print("\n⚠️  未标注的图片:")
        for i, filename in enumerate(unlabeled_files[:10], 1):
            print(f"   {i}. {filename}")
        if len(unlabeled_files) > 10:
            print(f"   ... 还有 {len(unlabeled_files) - 10} 张")
    
    print("\n" + "=" * 60)
    
    if unlabeled_count > 0:
        print("💡 解决方案:")
        print("   1. 使用 LabelImg 标注这些图片")
        print("   2. 打开 LabelImg，选择 YOLO 格式")
        print("   3. 打开目录: data/images/")
        print("   4. 保存目录: data/images/ (与图片同目录)")
        print("   5. 逐张标注均线密集形态")
        return False
    else:
        print("✅ 所有图片都已标注，可以开始训练！")
        return True

if __name__ == "__main__":
    check_labels()
