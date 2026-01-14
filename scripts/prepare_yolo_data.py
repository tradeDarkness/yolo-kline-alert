"""
YOLO 数据集准备脚本
功能：
1. 从 data/pine_signals 读取生成的图像和标签
2. 随机划分为训练集 (train) 和验证集 (val)
3. 整理为 YOLOv8/v11 标准目录结构
4. 生成 dataset.yaml 配置文件
"""

import os
import shutil
import random
import yaml
from pathlib import Path

# =================配置=================
# 原始数据目录
SOURCE_IMG_DIR = 'data/pine_signals/images'
SOURCE_LBL_DIR = 'data/pine_signals/labels'

# 目标输出目录
DEST_DIR = 'data/yolo_dataset'

# 划分比例
TRAIN_RATIO = 0.8  # 80% 训练, 20% 验证

# 类别定义 (必须与 chart_generator.py 一致)
CLASS_NAMES = {
    0: 'LONG',
    1: 'SHORT'
}
# =====================================

def prepare_data():
    """执行数据准备流程"""
    print(f"🚀 开始准备 YOLO 数据集...")
    
    # 1. 清理并创建目标目录
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
        print(f"   已清理旧目录: {DEST_DIR}")
    
    # 创建 train/val 的 images/labels 目录
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_DIR, split, kind), exist_ok=True)
            
    # 2. 收集匹配的文件对
    if not os.path.exists(SOURCE_IMG_DIR) or not os.path.exists(SOURCE_LBL_DIR):
        print(f"❌ 源目录不存在！请先运行 detection 脚本生成数据。")
        return

    images = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith('.png')]
    valid_pairs = []
    
    print(f"   扫描源目录...")
    for img_file in images:
        base_name = os.path.splitext(img_file)[0]
        txt_file = base_name + '.txt'
        
        src_img_path = os.path.join(SOURCE_IMG_DIR, img_file)
        src_txt_path = os.path.join(SOURCE_LBL_DIR, txt_file)
        
        # 检查对应的标签文件是否存在
        if os.path.exists(src_txt_path):
            valid_pairs.append((src_img_path, src_txt_path))
        else:
            # 只有标签存在才算有效样本
            pass
            
    if not valid_pairs:
        print("❌ 未找到有效的 图像-标签 对！")
        return

    # 3. 随机划分
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * TRAIN_RATIO)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"📊 数据集统计:")
    print(f"   总样本: {len(valid_pairs)}")
    print(f"   训练集: {len(train_pairs)}")
    print(f"   验证集: {len(val_pairs)}")
    
    # 4. 复制文件
    def copy_file(pairs, split):
        print(f"   正在生成 {split} 集...")
        for img_src, txt_src in pairs:
            # 复制图片
            shutil.copy(img_src, os.path.join(DEST_DIR, split, 'images', os.path.basename(img_src)))
            # 复制标签
            shutil.copy(txt_src, os.path.join(DEST_DIR, split, 'labels', os.path.basename(txt_src)))
            
    copy_file(train_pairs, 'train')
    copy_file(val_pairs, 'val')
    
    # 5. 生成 dataset.yaml
    yaml_content = {
        'path': os.path.abspath(DEST_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': CLASS_NAMES
    }
    
    yaml_path = os.path.join(DEST_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"✅ 数据集准备完成！")
    print(f"   配置文件: {yaml_path}")
    print(f"   训练命令提示: yolo train data={yaml_path} ...")

if __name__ == "__main__":
    prepare_data()
