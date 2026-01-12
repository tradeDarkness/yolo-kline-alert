import os
import shutil
import random
from pathlib import Path
import yaml

# Config
SOURCE_DIRS = ['data/ready_to_label/UP', 'data/ready_to_label/DOWN', 'data/ready_to_label/NEGATIVE']
DEST_DIR = 'data/yolo_dataset'
TRAIN_RATIO = 0.8

def prepare_data():
    # 1. Clean/Create Destination
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_DIR, split, kind), exist_ok=True)
            
    # 2. Collect all pairs
    all_pairs = [] # (img_path, txt_path)
    
    for s_dir in SOURCE_DIRS:
        if not os.path.exists(s_dir):
            print(f"Skipping missing dir: {s_dir}")
            continue
            
        files = os.listdir(s_dir)
        images = [f for f in files if f.endswith('.png')]
        
        for img in images:
            base_name = os.path.splitext(img)[0]
            txt_name = base_name + '.txt'
            
            img_path = os.path.join(s_dir, img)
            txt_path = os.path.join(s_dir, txt_name)
            
            if os.path.exists(txt_path):
                all_pairs.append((img_path, txt_path))
    
    # 3. Shuffle and Split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * TRAIN_RATIO)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    print(f"📊 Total Samples: {len(all_pairs)}")
    print(f"   Train: {len(train_pairs)}")
    print(f"   Val:   {len(val_pairs)}")
    
    # 4. Copy Files
    def copy_set(pairs, split):
        print(f"🚀 Preparing {split} set...")
        for img_src, txt_src in pairs:
            # Copy Image
            shutil.copy(img_src, os.path.join(DEST_DIR, split, 'images', os.path.basename(img_src)))
            # Copy Label
            shutil.copy(txt_src, os.path.join(DEST_DIR, split, 'labels', os.path.basename(txt_src)))
            
    copy_set(train_pairs, 'train')
    copy_set(val_pairs, 'val')
    
    # 5. Create data.yaml
    yaml_content = {
        'path': os.path.abspath(DEST_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'ma_cluster'
        }
    }
    
    yaml_path = os.path.join(DEST_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"✅ Data preparation complete!")
    print(f"   Config saved to: {yaml_path}")

if __name__ == "__main__":
    prepare_data()
