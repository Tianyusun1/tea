import os
import csv
import math
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# ==========================================
# 1. 定义专属 ECA 模块 (读取 Tea_YOLO 必备)
# ==========================================
class ECA(nn.Module):
    """Efficient Channel Attention 模块"""
    def __init__(self, k_size=5, *args, **kwargs): 
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# 动态注入，让 YOLO 认识您的改进模块
tasks.ECA = ECA 

# ==========================================
# 2. 路径与模型精细化配置
# ==========================================
DATASET_YAML = "/media/dl-d/Data/sty/tea/Tea leaf disease Dataset/Tea Leaf Diseases/data.yaml"

# 统一存放评估结果（CSV表格和图表）的文件夹
EVAL_OUTPUT_DIR = "eval_results"

# 🌟 改进点：使用字典直接映射每个模型的 best.pt 绝对路径
# 请根据您电脑上的实际位置，检查并修改下面的路径！
MODELS_DICT = {
    "YOLOv5su": "/home/sty/pyfile/tea/detect/output/YOLOv5su/weights/best.pt", 
    "YOLOv8m":  "/home/sty/pyfile/tea/detect/output/YOLOv8m/weights/best.pt",  
    "YOLOv9c":  "/home/sty/pyfile/tea/detect/output/YOLOv9c/weights/best.pt",  
    "YOLOv10m": "/home/sty/pyfile/tea/detect/output/YOLOv10m/weights/best.pt",          
    "Tea_YOLO": "/home/sty/pyfile/tea/detect/output/Tea_YOLO/weights/best.pt"           # 改进模型路径
}

def evaluate_models():
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    print("📊 开始在 Test 测试集上评估所有模型...")
    
    results_summary = []
    
    for model_name, weight_path in MODELS_DICT.items():
        print("\n" + "=" * 60)
        
        # 严格检查路径是否存在
        if not os.path.exists(weight_path):
            print(f"⚠️ 找不到 {model_name} 的权重: {weight_path}")
            print("请检查 MODELS_DICT 中的路径是否填写正确。跳过...")
            continue
            
        print(f"🔍 正在评估: {model_name}")
        print(f"📦 加载权重: {weight_path}")
        print("=" * 60)
        
        try:
            model = YOLO(weight_path)
            
            # 运行评估
            metrics = model.val(
                data=DATASET_YAML, 
                split='test',  
                device=0,      
                plots=True,    
                project=EVAL_OUTPUT_DIR,   # 评估生成的图表统一保存在 eval_results 目录下
                name=f"{model_name}_eval", 
                exist_ok=True
            )
            
            map50 = metrics.box.map50           
            map50_95 = metrics.box.map          
            inference_speed = metrics.speed['inference'] 
            
            print(f"\n✅ {model_name} 评估完成！")
            print(f"🎯 mAP@0.5: {map50:.4f}")
            print(f"⚡ 推理速度: {inference_speed:.2f} ms/img")
            
            results_summary.append({
                "Model Name": model_name,
                "mAP@0.5": round(map50, 4),
                "mAP@0.5:0.95": round(map50_95, 4),
                "Inference Speed (ms)": round(inference_speed, 2)
            })
            
        except Exception as e:
            print(f"❌ 评估 {model_name} 时发生错误: {e}")

    # ==========================================
    # 3. 导出对比 CSV
    # ==========================================
    if results_summary:
        csv_file = os.path.join(EVAL_OUTPUT_DIR, "model_comparison_results.csv")
        keys = results_summary[0].keys()
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results_summary)
            
        print("\n" + "🎉" * 20)
        print(f"✅ 评估大满贯！")
        print(f"📂 所有论文图表和 CSV 均已安全存入: {os.path.abspath(EVAL_OUTPUT_DIR)}")
        print("🎉" * 20)

if __name__ == '__main__':
    evaluate_models()