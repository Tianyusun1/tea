import os
import math
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks # 导入 YOLO 的底层任务模块

# ==========================================
# 1. 定义您专属的 ECA 注意力机制 (修复适配版)
# ==========================================
class ECA(nn.Module):
    """Efficient Channel Attention 模块 (绕过 YOLO 解析器限制版)"""
    # 移除强制的 c1, c2 参数，使用固定的 kernel_size=5
    # 这能完美绕过 YOLO 解析器报错，并且在深层网络中表现优异
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

# ==========================================
# 2. 动态注入 (Monkey Patching) ✨ 最优雅的解法
# ==========================================
# 将我们自己写的 ECA 类强行挂载到 YOLO 的命名空间中
# 这样底层解析 yolov8_tea.yaml 时，就能顺利认识并加载 ECA 模块了！
tasks.ECA = ECA 

# ==========================================
# 3. 配置并启动训练
# ==========================================
DATASET_YAML = "/media/dl-d/Data/sty/tea/Tea leaf disease Dataset/Tea Leaf Diseases/data.yaml"
YAML_CFG = "yolov8_tea.yaml"  
PRETRAINED_WEIGHTS = "/media/dl-d/Data/sty/tea/models/yolov8m.pt" 
OUTPUT_DIR = "output"

def train_custom_model():
    print("🚀 正在组装并构建专属模型 Tea-YOLO...")
    
    # 按照 YAML 构建架构，并加载 v8m 的部分基础权重来加速收敛
    model = YOLO(YAML_CFG).load(PRETRAINED_WEIGHTS)
    
    # 打印模型结构，您能清晰看到深度的 DWConv 和 ECA 被成功挂载！
    model.info()

    print("\n🔥 开始训练 Tea-YOLO...")
    results = model.train(
        data=DATASET_YAML,
        epochs=50,           
        imgsz=640,           
        batch=32,             
        device=0,            
        project=OUTPUT_DIR,  
        name="Tea_YOLO",      # 结果将保存在 output/Tea_YOLO
        workers=0,            # 避免多线程加载报错
        amp=False,            # 避免混合精度报错
        plots=True           
    )
    print("\n🎉 Tea-YOLO 训练完成！快去看看结果吧！")

if __name__ == '__main__':
    train_custom_model()