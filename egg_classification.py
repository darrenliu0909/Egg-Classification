import torch
from ultralytics import YOLO
import cv2
import os
import random
import shutil

# 🟢 类别映射
class_mapping = {
    "臭蛋": 0,
    "反蛋": 1,
    "空位蛋": 2,
    "无精蛋": 3,
    "有精蛋": 4,
}

# 🟢 数据集拆分（支持子文件夹）
def split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    all_images = []
    for category in os.listdir(image_dir):
        category_path = os.path.join(image_dir, category)
        if os.path.isdir(category_path):
            images = [os.path.join(category, img) for img in os.listdir(category_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            all_images.extend(images)

    random.shuffle(all_images)

    train_size = int(len(all_images) * train_ratio)
    val_size = int(len(all_images) * val_ratio)
    
    splits = {
        'train': all_images[:train_size],
        'val': all_images[train_size:train_size+val_size],
        'test': all_images[train_size+val_size:]
    }

    for split, split_images in splits.items():
        for image in split_images:
            src_img_path = os.path.join(image_dir, image)
            dest_img_path = os.path.join(output_dir, 'images', split, os.path.basename(image))
            shutil.copy(src_img_path, dest_img_path)

            # ✅ 生成标签（如果没有标注，创建空的 .txt）
            label_file = os.path.splitext(os.path.basename(image))[0] + '.txt'
            label_path = os.path.join(output_dir, 'labels', split, label_file)
            class_name = os.path.dirname(image)  # 获取类别名称
            class_id = class_mapping.get(class_name, -1)
            if class_id != -1:
                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # 示例标注（需替换成真实的）

    print("✅ 数据集拆分完成！")

# 🟢 训练 YOLOv8
def train_yolov8(model_type='yolov8s', data_yaml='data.yaml', epochs=50, batch_size=8, lr=0.001, weight_decay=0.0005):
    model = YOLO(model_type + '.pt')  # 载入预训练模型
    model.train(
        data=data_yaml, 
        epochs=epochs, 
        batch=batch_size, 
        imgsz=640, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr0=lr, 
        optimizer='SGD'
    )
    print("🎯 训练完成！")

# 🟢 评估模型
def evaluate_model(model_path, data_yaml):
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    print("📊 评估完成！", results)

# 🟢 实时推理（摄像头/视频）
def real_time_inference(model_path, video_source=0):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = f"{model.names[class_id]}: {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Egg Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 🟢 运行脚本
if __name__ == "__main__":
    image_dir = "dataset/images"
    labels_dir = "dataset/labels"
    output_dir = "dataset_split"
    data_yaml = "data.yaml"

    # 1️⃣ 拆分数据集
    split_dataset(image_dir, output_dir)

    # 2️⃣ 训练 YOLOv8
    train_yolov8(model_type='yolov8s', data_yaml=data_yaml, epochs=50, batch_size=8, lr=0.001, weight_decay=0.0005)

    # 3️⃣ 评估模型
    evaluate_model(model_path='runs/train/exp/weights/best.pt', data_yaml=data_yaml)

    # 4️⃣ 进行实时推理（按需开启）
    # real_time_inference(model_path='runs/train/exp/weights/best.pt', video_source=0)
