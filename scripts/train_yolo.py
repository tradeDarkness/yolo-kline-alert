"""
YOLO 模型训练脚本
功能：
1. 加载预训练的 YOLOv11 模型 (yolo11n.pt)
2. 读取数据集配置文件 (dataset.yaml)
3. 执行模型训练 (Fine-tuning)
4. 保存最佳权重

使用前请确保：
1. 已运行 prepare_yolo_data.py 生成了 data/yolo_dataset
2. 已安装 ultralytics 库
"""

from ultralytics import YOLO
import os

def train():
    """执行训练流程"""
    # 1. 加载模型
    # yolo11n.pt 是 Nano 版本，速度最快，适合实时检测
    print("🚀 加载 YOLO11 Nano 模型...")
    model = YOLO("yolo11n.pt") 
    
    # 2. 配置文件路径
    # 必须指向 prepare_yolo_data.py 生成的 dataset.yaml
    yaml_path = os.path.abspath("data/yolo_dataset/dataset.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ 未找到配置文件: {yaml_path}")
        print("   请先运行: python scripts/prepare_yolo_data.py")
        return
    
    # 3. 开始训练
    print(f"🔥 开始训练 (配置文件: {yaml_path})...")
    # 参数说明：
    # epochs: 训练轮数 (建议 50-100)
    # imgsz: 输入图像大小 (需与生成图像时保持一致或接近)
    # batch: 批次大小 (根据显存调整)
    # device: 'mps' (Mac M系列芯片), 'cuda' (NVIDIA GPU), 'cpu'
    # 参考: 金融图表训练最佳实践
    # 严禁使用 flipud, degrees, mosaic 等破坏 K 线结构和时间序列的增强
    results = model.train(
        data=yaml_path,
        epochs=50,          
        imgsz=640,          
        batch=16,
        project="runs/detect",  # 训练结果保存目录
        name="kline_cluster_yolo11", # 实验名称
        exist_ok=True,      # 是否覆盖同名实验目录
        device='mps',       # Mac M系列芯片加速 (如果不适用请改为 'cpu')
        
        # =========================================
        # 数据增强覆盖 (针对 K 线图优化)
        # =========================================
        degrees=0.0,      # 禁止旋转 (保持时间轴水平)
        translate=0.1,    # 允许轻微平移
        scale=0.5,        # 允许缩放 (模拟不同视野)
        shear=0.0,        # 禁止剪切
        perspective=0.0,  # 禁止透视
        flipud=0.0,       # 禁止垂直翻转 (多空反转是灾难)
        fliplr=0.0,       # 禁止水平翻转 (时间不可逆)
        mosaic=0.0,       # 禁止马赛克 (破坏时间连续性)
        mixup=0.0,        # 禁止混合
        # =========================================
    )
    
    print("✅ 训练完成！")
    print(f"   最佳模型权重已保存至: runs/detect/kline_cluster_yolo11/weights/best.pt")

if __name__ == "__main__":
    train()
