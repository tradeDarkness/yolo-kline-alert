from ultralytics import YOLO
import os

def train():
    # 1. Load Model
    # YOLO11 is the latest SOTA model from Ultralytics
    print("🚀 Loading YOLO11 Nano model...")
    model = YOLO("yolo11n.pt") 
    
    # 2. Config Path
    data_yaml = os.path.abspath("data/yolo_dataset/data.yaml")
    
    # 3. Train
    print(f"🔥 Starting Training on {data_yaml} (Device: MPS/M4)...")
    results = model.train(
        data=data_yaml,
        epochs=50,          # Adjust epochs as needed
        imgsz=640,          # Image size matches generation (approx)
        batch=16,
        project="runs/detect",
        name="kline_cluster_yolo11",
        exist_ok=True,      # Overwrite existing run
        device='mps'        # Use Apple Silicon (M4) acceleration
    )
    
    print("✅ Training Complete!")
    print(f"   Best model saved at: runs/detect/kline_cluster_v1/weights/best.pt")

if __name__ == "__main__":
    train()
