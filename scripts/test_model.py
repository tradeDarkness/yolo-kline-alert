from ultralytics import YOLO
import os
import glob
import random
import matplotlib.pyplot as plt
import cv2

# Config
MODEL_PATH = "runs/detect/kline_cluster_yolo11/weights/best.pt"
TEST_IMAGES_DIR = "data/yolo_dataset/val/images"
OUTPUT_DIR = "runs/detect/inference_test"

def test():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # 1. Load Model
    print(f"🚀 Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 2. Get random test images
    all_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
    if not all_images:
        print("❌ No images found in val set.")
        return
        
    test_samples = random.sample(all_images, min(5, len(all_images)))
    
    # 3. Predict and Show
    print(f"📸 Running inference on {len(test_samples)} images...")
    
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = model.predict(test_samples, save=True, project="runs/detect", name="inference_test", exist_ok=True)
    
    print(f"✅ Inference complete! Check results in: {OUTPUT_DIR}")
    
    # Optional: Display one input vs output path logic (Ultralytics saves to runs/detect/inference_test/image.png)
    for img_path in test_samples:
        fname = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, fname)
        print(f"   Saved: {save_path}")

if __name__ == "__main__":
    test()
