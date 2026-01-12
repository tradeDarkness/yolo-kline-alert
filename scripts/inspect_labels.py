import cv2
import os
import glob
import matplotlib.pyplot as plt

# Config
IMG_DIR = "data/yolo_dataset/train/images"
LABEL_DIR = "data/yolo_dataset/train/labels"
OUTPUT_DIR = "runs/debug_labels"

def inspect():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get some samples
    img_files = glob.glob(os.path.join(IMG_DIR, "*.png"))[:10]
    
    print(f"🕵️ Inspecting {len(img_files)} labels...")
    
    for img_path in img_files:
        basename = os.path.basename(img_path)
        txt_name = os.path.splitext(basename)[0] + ".txt"
        txt_path = os.path.join(LABEL_DIR, txt_name)
        
        if not os.path.exists(txt_path):
            print(f"⚠️ Label missing for {basename}")
            continue
            
        # Read Image
        img = cv2.imread(img_path)
        h_img, w_img, _ = img.shape
        
        # Read Label
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            cls, cx, cy, w, h = parts
            
            # Convert YOLO to Pixel
            x1 = int((cx - w/2) * w_img)
            y1 = int((cy - h/2) * h_img)
            x2 = int((cx + w/2) * w_img)
            y2 = int((cy + h/2) * h_img)
            
            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "Cluster", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # Save debug image
        out_path = os.path.join(OUTPUT_DIR, "debug_" + basename)
        cv2.imwrite(out_path, img)
        print(f"   Saved: {out_path}")

if __name__ == "__main__":
    inspect()
