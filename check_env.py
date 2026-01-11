# check_env.py
import sys
import torch
from ultralytics import YOLO
import cv2

def check():
    print(f"✅ Python Version: {sys.version.split()[0]}")
    
    # 检查 Torch 和 MPS (Mac 加速)
    print(f"✅ PyTorch Version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("🚀 macOS MPS (Metal) acceleration is AVAILABLE! (Great for YOLO)")
    else:
        print("⚠️ macOS MPS not detected. Will use CPU (Slower but works).")

    # 检查 OpenCV
    print(f"✅ OpenCV Version: {cv2.__version__}")

    # 检查 YOLO
    try:
        # 这里会自动下载 yolo11n.pt 模型，可能会稍微花点时间
        model = YOLO('yolo11n.pt') 
        print("✅ YOLO11 imported and model loaded successfully.")
    except Exception as e:
        print(f"❌ YOLO Error: {e}")

if __name__ == "__main__":
    check()