from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO('yolo11n.pt') 

    # 开始训练
    # data: 指向你的 data.yaml
    # epochs: 训练轮数，50轮通常够演示用
    # imgsz: 图片大小 640
    model.train(data='data.yaml', epochs=50, imgsz=640, device='mps') # Mac M1/M2 用 mps，否则用 cpu