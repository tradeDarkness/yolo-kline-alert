#! /usr/bin/env python3
"""
检查标注文件是否完整 (UP/DOWN Structure)
"""
import os
from pathlib import Path

def check_dir(base_path, label_name):
    p = Path(base_path)
    if not p.exists():
        print(f"❌ 目录不存在: {base_path}")
        return 0, 0, []
        
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(p.glob(f"*{ext}")))
        
    labeled = 0
    unlabeled = 0
    missing = []
    
    for img in image_files:
        txt = img.with_suffix('.txt')
        if txt.exists() and txt.stat().st_size > 0:
            labeled += 1
        else:
            unlabeled += 1
            missing.append(img.name)
            
    print(f"\n📂 {label_name} ({base_path})")
    print(f"   总数: {len(image_files)}")
    print(f"   ✅ 已标注: {labeled}")
    print(f"   ❌ 未标注: {unlabeled}")
    
    if missing:
        print(f"   ⚠️  前5个未标注: {missing[:5]}")
        
    return labeled, unlabeled, missing

def check_labels():
    print(f"📊 数据集检查报告 (data/ready_to_label)")
    print("=" * 60)
    
    up_labeled, up_unlabeled, _ = check_dir("data/ready_to_label/UP", "UP Set")
    down_labeled, down_unlabeled, _ = check_dir("data/ready_to_label/DOWN", "DOWN Set")
    
    total_labeled = up_labeled + down_labeled
    total_unlabeled = up_unlabeled + down_unlabeled
    
    print("\n" + "=" * 60)
    print(f"📈 总计已标注: {total_labeled}")
    print(f"📉 总计未标注: {total_unlabeled}")
    
    if total_unlabeled > 0:
        print("\n💡 标注说明:")
        print("   1. 打开 LabelImg")
        print("   2. 'Open Dir' -> 选择 data/ready_to_label/UP (或 DOWN)")
        print("   3. 'Change Save Dir' -> 保持与图片同一目录")
        print("   4. 标注 'cluster' 或 'signal'")
        return False
    else:
        print("\n✅ 所有图片都已标注！")
        return True

if __name__ == "__main__":
    check_labels()
