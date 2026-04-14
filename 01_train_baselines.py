import os
from ultralytics import YOLO

# ==========================================
# 1. 路径配置 (请确保这些路径与您电脑上的完全一致)
# ==========================================
DATASET_YAML = "/media/dl-d/Data/sty/tea/Tea leaf disease Dataset/Tea Leaf Diseases/data.yaml"

# 定义模型字典：键为模型名称(用于区分输出文件夹)，值为模型权重绝对路径
MODEL_PATHS = {
    "YOLOv5su": "/media/dl-d/Data/sty/tea/models/yolov5su.pt",
    "YOLOv8m":  "/media/dl-d/Data/sty/tea/models/yolov8m.pt",
    "YOLOv9c":  "/media/dl-d/Data/sty/tea/models/yolov9c.pt",
    "YOLOv10m": "/media/dl-d/Data/sty/tea/models/yolov10m.pt"
}

# 设置总输出根目录
OUTPUT_DIR = "output"

def train_baselines():
    # 确保输出总目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 开始批量训练基线模型...")
    print(f"📁 结果将统一保存在: {os.path.abspath(OUTPUT_DIR)} 目录下\n")

    # 遍历字典，依次训练每个模型
    for model_name, model_path in MODEL_PATHS.items():
        print("=" * 60)
        print(f"🔥 当前正在训练: {model_name}")
        print(f"📦 正在加载权重: {model_path}")
        print("=" * 60)
        
        try:
            # 1. 加载模型
            model = YOLO(model_path)
            
            # 2. 开始训练
            # 注意：project 设置为 OUTPUT_DIR，name 设置为 model_name
            # 最终该模型的权重和图表会保存在 output/YOLOv5su/ 等对应目录下
            results = model.train(
                data=DATASET_YAML,   
                epochs=50,          
                imgsz=640,           
                batch=64,            
                device=0,            
                project=OUTPUT_DIR,  
                name=model_name,     
                exist_ok=True,       
                workers=0,          
                amp=False,         
                plots=True           
            )
            
            print(f"\n✅ {model_name} 训练完成！")
            print(f"📂 权重及评估结果已保存至: {os.path.join(OUTPUT_DIR, model_name)}\n")
            
        except Exception as e:
            print(f"❌ 训练 {model_name} 时发生错误: {e}")
            print("将跳过该模型，继续训练下一个...\n")

if __name__ == '__main__':
    train_baselines()