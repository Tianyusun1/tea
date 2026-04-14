import os
# 🌟 强行修改 Gradio 的临时文件存放路径，完美绕过实验室服务器的权限报错
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_tmp")

import gradio as gr
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 设置为无界面后台绘制，防止服务器多线程报错
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义专属 ECA 模块 (加载 Tea_YOLO 必备)
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

tasks.ECA = ECA 

# ==========================================
# 2. 配置路径 (包含模型权重 和 训练日志CSV)
# ==========================================
MODEL_DICT = {
    "YOLOv5su": "/home/sty/pyfile/tea/detect/output/YOLOv5su/weights/best.pt",
    "YOLOv8m":  "/home/sty/pyfile/tea/detect/output/YOLOv8m/weights/best.pt",
    "YOLOv9c":  "/home/sty/pyfile/tea/detect/output/YOLOv9c/weights/best.pt",
    "YOLOv10m": "/home/sty/pyfile/tea/detect/output/YOLOv10m/weights/best.pt",
    "Tea-YOLO": "/home/sty/pyfile/tea/detect/output/Tea_YOLO/weights/best.pt" 
}

CSV_DICT = {
    "YOLOv5su": "/home/sty/pyfile/tea/detect/output/YOLOv5su/results.csv",
    "YOLOv8m":  "/home/sty/pyfile/tea/detect/output/YOLOv8m/results.csv",
    "YOLOv9c":  "/home/sty/pyfile/tea/detect/output/YOLOv9c/results.csv",
    "YOLOv10m": "/home/sty/pyfile/tea/detect/output/YOLOv10m/results.csv",
    "Tea-YOLO": "/home/sty/pyfile/tea/detect/output/Tea_YOLO/results.csv" 
}

# 🌟 指标映射也换成纯英文，显得更学术
METRIC_MAP = {
    "mAP@0.5": "metrics/mAP50(B)",
    "mAP@0.5:0.95": "metrics/mAP50-95(B)",
    "Train Box Loss": "train/box_loss",
    "Val Box Loss": "val/box_loss"
}

loaded_models = {}
def get_model(model_name):
    if model_name not in loaded_models:
        path = MODEL_DICT[model_name]
        if os.path.exists(path):
            loaded_models[model_name] = YOLO(path)
        else:
            return None
    return loaded_models[model_name]

# ==========================================
# 3. 核心功能函数
# ==========================================
def compare_models(image, model_a_name, model_b_name, conf_threshold):
    if image is None:
        return None, None, "请先上传图片"
    
    results_out = []
    reports = []

    for name in [model_a_name, model_b_name]:
        model = get_model(name)
        if model is None:
            results_out.append(None)
            reports.append(f"❌ 找不到 {name} 的权重文件")
            continue
        
        res = model.predict(source=image, conf=conf_threshold, device=0)[0]
        plot_img = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
        results_out.append(plot_img)
        
        count = len(res.boxes)
        speed_ms = res.speed['inference']
        reports.append(f"【{name}】识别到 {count} 个病害目标 ⚡(耗时: {speed_ms:.1f} ms)")

    final_report = f"📊 对比分析：\n{reports[0]}\n{reports[1]}"
    return results_out[0], results_out[1], final_report

def plot_training_curves(selected_models, selected_metric):
    if not selected_models:
        fig, ax = plt.subplots(figsize=(10, 6))
        # 🌟 提示语换成英文
        ax.text(0.5, 0.5, "Please select models to compare", ha='center', va='center', fontsize=15)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    target_col_raw = METRIC_MAP[selected_metric]
    
    has_data = False
    for model_name in selected_models:
        csv_path = CSV_DICT.get(model_name)
        
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip() 
                
                if target_col_raw in df.columns and 'epoch' in df.columns:
                    # 画线
                    ax.plot(df['epoch'], df[target_col_raw], label=model_name, linewidth=2.5, alpha=0.8)
                    has_data = True
            except Exception as e:
                print(f"Failed to read CSV for {model_name}: {e}")

    if has_data:
        # 🌟 坐标轴和标题全部换成高大上的英文
        ax.set_title(f"Training Curve Comparison - {selected_metric}", fontsize=16, pad=15)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(selected_metric, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)
        if "Loss" not in selected_metric:
            ax.set_ylim(bottom=0)
    else:
        # 🌟 错误提示换成英文
        ax.text(0.5, 0.5, f"Data not found for {selected_metric}\nPlease check results.csv paths.", 
                ha='center', va='center', fontsize=12, color='red')

    plt.tight_layout()
    return fig

# ==========================================
# 4. UI 界面设计 
# ==========================================
with gr.Blocks(title="茶叶病害多模型对比系统") as demo:
    gr.Markdown("# 🍃 Tea-YOLO：茶叶病害智能检测与分析系统")
    
    with gr.Tabs():
        with gr.Tab("📷 实时病害检测对比"):
            gr.Markdown("上传茶叶图片，直观对比基线模型与 **Tea-YOLO** 在微小病斑识别与抗背景干扰上的表现差异。")
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(label="上传待测图片", type="numpy")
                    conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="置信度阈值")
                    compare_btn = gr.Button("🚀 启动智能检测", variant="primary")
                    status_msg = gr.Textbox(label="检测结论汇总", interactive=False, lines=3)
                    
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column():
                            model_a_sel = gr.Dropdown(list(MODEL_DICT.keys()), value=list(MODEL_DICT.keys())[-2], label="选择左侧模型")
                            out_a = gr.Image(label="左侧模型识别结果")
                        with gr.Column():
                            model_b_sel = gr.Dropdown(list(MODEL_DICT.keys()), value=list(MODEL_DICT.keys())[-1], label="选择右侧模型")
                            out_b = gr.Image(label="Tea-YOLO 识别结果")

            compare_btn.click(
                fn=compare_models,
                inputs=[input_img, model_a_sel, model_b_sel, conf_slider],
                outputs=[out_a, out_b, status_msg]
            )

        with gr.Tab("📈 模型训练数据面板"):
            gr.Markdown("实时读取各模型的训练日志（`results.csv`），动态生成**学术级英文对比曲线**。")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 图表设置")
                    models_checkbox = gr.CheckboxGroup(
                        choices=list(MODEL_DICT.keys()), 
                        value=list(MODEL_DICT.keys())[-2:], 
                        label="1. 选择要对比的模型"
                    )
                    metric_dropdown = gr.Dropdown(
                        choices=list(METRIC_MAP.keys()), 
                        value="mAP@0.5", 
                        label="2. 选择对比指标"
                    )
                    plot_btn = gr.Button("📊 生成对比曲线图", variant="primary")
                
                with gr.Column(scale=3):
                    plot_output = gr.Plot(label="训练趋势图 (Training Curves)")

            plot_btn.click(
                fn=plot_training_curves,
                inputs=[models_checkbox, metric_dropdown],
                outputs=[plot_output]
            )
            
            metric_dropdown.change(
                fn=plot_training_curves,
                inputs=[models_checkbox, metric_dropdown],
                outputs=[plot_output]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)