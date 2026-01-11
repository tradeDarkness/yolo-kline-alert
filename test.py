from ultralytics import YOLO
import cv2
import os

# 1. 加载模型
model = YOLO('runs/detect/train_yolo11_synthetic2/weights/best.pt')

# 2. 图片路径
source_img = '/Users/zhangzc/2026/yolo-kline-alert/test.png' 

# 3. 推理 - 使用低阈值先检测，看看是否有任何结果
CONF_THRESHOLD = 0.013 # 主要置信度阈值
LOW_CONF_THRESHOLD = 0.01  # 低阈值，用于探索性检测

print("="*60)
print(f"🔍 检测图片: {source_img}")
print("="*60)

# 先用低阈值检测，看看是否有任何检测
results_low = model.predict(source=source_img, imgsz=640, conf=LOW_CONF_THRESHOLD, verbose=False)
result_low = results_low[0]
boxes_low = result_low.boxes

# 再用主要阈值检测
results = model.predict(source=source_img, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
result = results[0]
boxes = result.boxes

# 4. 解析检测结果
print(f"\n📊 检测结果 (置信度阈值: {CONF_THRESHOLD})")
print("-"*60)

if boxes is not None and len(boxes) > 0:
    print(f"✅ 检测到 {len(boxes)} 个均线密集形态 (阈值 >= {CONF_THRESHOLD*100:.0f}%):\n")
    
    # 按置信度排序
    confidences = [(float(box.conf[0]), int(box.cls[0]), box.xyxy[0].tolist()) for box in boxes]
    confidences.sort(reverse=True, key=lambda x: x[0])
    
    for i, (conf, cls, xyxy) in enumerate(confidences, 1):
        print(f"  {i}. 置信度: {conf:.2%} | 位置: ({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")
    
    # 统计信息
    conf_list = [c[0] for c in confidences]
    print(f"\n📈 统计信息:")
    print(f"   最高置信度: {max(conf_list):.2%}")
    print(f"   最低置信度: {min(conf_list):.2%}")
    print(f"   平均置信度: {sum(conf_list)/len(conf_list):.2%}")
else:
    print(f"❌ 未检测到均线密集形态 (阈值 >= {CONF_THRESHOLD*100:.0f}%)")
    
    # 如果主要阈值没检测到，显示低阈值的结果
    if boxes_low is not None and len(boxes_low) > 0:
        print(f"\n🔍 使用低阈值 ({LOW_CONF_THRESHOLD*100:.0f}%) 检测到 {len(boxes_low)} 个候选目标:")
        
        # 按置信度排序，只显示前10个
        confidences_low = [(float(box.conf[0]), box.xyxy[0].tolist()) for box in boxes_low]
        confidences_low.sort(reverse=True, key=lambda x: x[0])
        
        print(f"   (显示前10个最高置信度的检测):\n")
        for i, (conf, xyxy) in enumerate(confidences_low[:10], 1):
            print(f"  {i}. 置信度: {conf:.2%} | 位置: ({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")
        
        if len(confidences_low) > 10:
            print(f"   ... 还有 {len(confidences_low) - 10} 个检测结果")
        
        conf_list_low = [c[0] for c in confidences_low]
        print(f"\n💡 建议:")
        print(f"   最高置信度: {max(conf_list_low):.2%}")
        if max(conf_list_low) < CONF_THRESHOLD:
            print(f"   可以尝试将阈值降低到 {max(conf_list_low)*0.8:.3f} 来查看检测结果")
    else:
        print(f"\n⚠️  即使使用极低阈值 ({LOW_CONF_THRESHOLD*100:.0f}%) 也未检测到任何目标")
        print(f"   可能原因:")
        print(f"   1. 图片中确实没有均线密集形态")
        print(f"   2. 模型训练不充分，需要更多训练数据")
        print(f"   3. 图片格式或尺寸不匹配")

print("="*60)

# 5. 保存检测结果图片（带标注框）
save_dir = "runs/detect/test_results"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "test_result.jpg")

# 使用低阈值的结果来绘制（显示更多检测）
res_img = result_low.plot() if boxes_low is not None and len(boxes_low) > 0 else result.plot()
cv2.imwrite(save_path, res_img)
print(f"\n📸 检测结果已保存到: {save_path}")
print(f"   可以打开查看检测框是否准确")

# 6. 分析检测质量
if boxes is not None and len(boxes) > 0:
    conf_list = [float(box.conf[0]) for box in boxes]
    max_conf = max(conf_list)
    
    print(f"\n⚠️  检测质量分析:")
    if max_conf < 0.05:
        print(f"   ❌ 置信度极低（最高仅 {max_conf:.2%}），检测可能不准确")
        print(f"   💡 建议:")
        print(f"      1. 检查检测结果图片，看检测框位置是否正确")
        print(f"      2. 如果检测框位置不对，可能需要重新训练模型")
        print(f"      3. 增加训练数据量或训练轮数")
    elif max_conf < 0.2:
        print(f"   ⚠️  置信度较低（最高 {max_conf:.2%}），检测可能不够准确")
        print(f"   💡 建议提高模型训练质量")
    else:
        print(f"   ✅ 置信度较好（最高 {max_conf:.2%}）")

# 7. 显示图片
window_name = "YOLO11 Test - Press any key to Close"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, res_img)

print("\n图片已弹出。请点击图片窗口，然后按键盘上的【任意键】退出。")

# 等待按键输入，0 表示无限等待
cv2.waitKey(0)

# 彻底销毁所有窗口
cv2.destroyAllWindows()
for i in range(5): # Mac 系统下有时需要多次调用才能彻底关闭
    cv2.waitKey(1)
    
print("脚本运行结束，已正常退出。")