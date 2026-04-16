import os
import numpy as np
# 🌟 强行修改 Gradio 的临时文件存放路径，完美绕过实验室服务器的权限报错
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_tmp")

import gradio as gr
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO, FastSAM
import ultralytics.nn.tasks as tasks
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 设置为无界面后台绘制，防止服务器多线程报错
import matplotlib.pyplot as plt

# ==========================================
# 0. 加载 Fast SAM 模型 (用于叶片分割)
# ==========================================
print("📦 正在加载本地 FastSAM 实例分割模型...")
FAST_SAM_PATH = "/home/sty/pyfile/tea/models/FastSAM-s.pt"
if os.path.exists(FAST_SAM_PATH):
    SAM_MODEL = FastSAM(FAST_SAM_PATH)
else:
    print(f"⚠️ 找不到 FastSAM 模型: {FAST_SAM_PATH}，将自动下载...")
    SAM_MODEL = FastSAM('FastSAM-s.pt')

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

# 动态注入，确保 YOLO 解析器能识别自定义模块
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
# 3. 核心功能：孤立叶片并标注病灶
# ==========================================
def process_isolated_leaf(img_bgr, boxes):
    """
    使用 FastSAM 将叶片单独分割出来（背景涂黑），并在上面标注病害范围
    """
    if len(boxes) == 0:
        return np.zeros_like(img_bgr), ""

    # 1. 获取 YOLO 提供的所有框作为 Prompt
    # 强制转换为纯 Python int 列表，避免 Float32 类型报错
    bboxes = []
    for box in boxes:
        coords = box.xyxy[0].cpu().numpy()
        bboxes.append([int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])])

    sam_results = SAM_MODEL.predict(img_bgr, bboxes=bboxes, verbose=False)[0]

    if sam_results.masks is None:
        return img_bgr, " ⚠️ 分割失败"

    # 2. 合并所有叶片的 Mask
    master_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
    for mask_data in sam_results.masks.data:
        mask = mask_data.cpu().numpy()
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
        master_mask = cv2.bitwise_or(master_mask, (mask > 0).astype(np.uint8) * 255)

    # 3. 抠出孤立叶片 (背景变黑)
    leaf_only = cv2.bitwise_and(img_bgr, img_bgr, mask=master_mask)

    # 4. 色彩异常检测 (在干净的叶片上找病斑)
    hsv = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    disease_mask = cv2.bitwise_and(master_mask, cv2.bitwise_not(green_mask))
    
    kernel = np.ones((5, 5), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 在孤立叶片上画出病斑精确轮廓
    iso_info = ""
    if contours:
        valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for idx, c in enumerate(valid_contours):
            if cv2.contourArea(c) < 50: 
                continue
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(leaf_only, [c], -1, (0, 0, 255), 2) # 红色病斑边缘
                cv2.circle(leaf_only, (cx, cy), 5, (255, 0, 0), -1)  # 蓝色病斑中心
                iso_info += f"\n      - 细粒度病灶{idx+1}: 中心({cx}, {cy})"

    return leaf_only, iso_info

# ==========================================
# 4. 业务逻辑函数
# ==========================================
def compare_models(image, model_a_name, model_b_name, conf_threshold):
    if image is None:
        return None, None, None, None, "请先上传图片", "请先上传图片"
    
    img_bgr_orig = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results_out_yolo = []
    results_out_iso = []
    reports = []
    
    # 🔥 新增：用于专门收集诊断病种类的列表
    disease_classes_summary = []

    for name in [model_a_name, model_b_name]:
        model = get_model(name)
        if model is None:
            results_out_yolo.append(None)
            results_out_iso.append(None)
            reports.append(f"❌ 找不到 {name} 的权重文件")
            disease_classes_summary.append(f"【{name}】: 模型加载失败")
            continue
        
        # --- 阶段 1：原本的 YOLO 宏观识别 ---
        res = model.predict(source=image, conf=conf_threshold, device=0)[0]
        plot_img = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
        results_out_yolo.append(plot_img)
        
        count = len(res.boxes)
        speed_ms = res.speed['inference']
        
        location_info = ""
        if count > 0:
            # 🔥 提取具体的病症类别，使用 set 去重
            unique_labels = list(set([res.names[int(box.cls[0])] for box in res.boxes]))
            disease_classes_summary.append(f"【{name}】: 识别为 {', '.join(unique_labels)}")
            
            location_info = "\n   📍 YOLO 叶片级识别："
            for i, box in enumerate(res.boxes):
                cls_id = int(box.cls[0])
                label = res.names[cls_id]
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                location_info += f"\n    {i+1}. [{label}] 置信度:{conf:.2f} | 框:[{x1},{y1}]至[{x2},{y2}]"
            
            # --- 阶段 2：单独分割叶片并标出具体病害 ---
            iso_img_bgr, iso_info = process_isolated_leaf(img_bgr_orig.copy(), res.boxes)
            results_out_iso.append(cv2.cvtColor(iso_img_bgr, cv2.COLOR_BGR2RGB))
            if iso_info:
                location_info += f"\n   ✂️ 孤立叶片微观定位:{iso_info}"
        else:
            disease_classes_summary.append(f"【{name}】: 健康 / 未检测到病害")
            location_info = "\n   ✅ 未发现明显病害区域。"
            results_out_iso.append(np.zeros_like(image)) # 全黑图

        reports.append(f"【{name}】检测到 {count} 个目标 (耗时: {speed_ms:.1f} ms){location_info}")

    # 组合文本输出
    final_disease_class_text = "\n".join(disease_classes_summary)
    final_report = f"📊 深度对比分析结论：\n\n{reports[0]}\n\n{reports[1]}"
    
    # 返回 4 张图 (A宏观, B宏观, A微观, B微观) 和 2 个文本报告 (病症类别, 详细坐标)
    return results_out_yolo[0], results_out_yolo[1], results_out_iso[0], results_out_iso[1], final_disease_class_text, final_report

def plot_training_curves(selected_models, selected_metric):
    if not selected_models:
        fig, ax = plt.subplots(figsize=(10, 6))
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
                    ax.plot(df['epoch'], df[target_col_raw], label=model_name, linewidth=2.5, alpha=0.8)
                    has_data = True
            except Exception as e:
                print(f"Failed to read CSV for {model_name}: {e}")

    if has_data:
        ax.set_title(f"Training Curve Comparison - {selected_metric}", fontsize=16, pad=15)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(selected_metric, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)
        if "Loss" not in selected_metric:
            ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, f"Data not found for {selected_metric}", ha='center', va='center', fontsize=12, color='red')
    plt.tight_layout()
    return fig

# ==========================================
# 5. UI 界面设计 (分上下两层展示)
# ==========================================
with gr.Blocks(title="茶叶病害多模型对比系统") as demo:
    gr.Markdown("# 🍃 Tea-YOLO：茶叶病害智能检测与分割分析系统")
    
    with gr.Tabs():
        with gr.Tab("📷 实时病害检测对比"):
            gr.Markdown("上传茶叶图片，系统将**保留原本叶片标记功能**，并在**下方单独输出提取后的叶片与精准病斑**。")
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(label="上传待测图片", type="numpy")
                    conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="置信度阈值")
                    compare_btn = gr.Button("🚀 启动智能检测与分割", variant="primary")
                    
                    # 🔥 新增输出：专门展示具体病症类别的文本框，放在坐标详情上方
                    disease_class_msg = gr.Textbox(label="🌿 最终诊断病症类别", interactive=False, lines=2)
                    
                    status_msg = gr.Textbox(label="检测结论与坐标详情", interactive=False, lines=12)
                    
                with gr.Column(scale=2):
                    # 上半部分：原本的功能 (原图+框)
                    gr.Markdown("### 1️⃣ 宏观定位 (保持原有功能)")
                    with gr.Row():
                        with gr.Column():
                            model_a_sel = gr.Dropdown(list(MODEL_DICT.keys()), value=list(MODEL_DICT.keys())[-2], label="选择左侧模型")
                            out_a_yolo = gr.Image(label="左侧模型 - 叶片标记")
                        with gr.Column():
                            model_b_sel = gr.Dropdown(list(MODEL_DICT.keys()), value=list(MODEL_DICT.keys())[-1], label="选择右侧模型")
                            out_b_yolo = gr.Image(label="Tea-YOLO - 叶片标记")
                    
                    # 下半部分：单独提取的叶子 + 病害精细标注
                    gr.Markdown("### 2️⃣ 孤立叶片与微观病灶提取")
                    with gr.Row():
                        with gr.Column():
                            out_a_iso = gr.Image(label="左侧模型 - 单独叶片及病灶")
                        with gr.Column():
                            out_b_iso = gr.Image(label="Tea-YOLO - 单独叶片及病灶")

            compare_btn.click(
                fn=compare_models,
                inputs=[input_img, model_a_sel, model_b_sel, conf_slider],
                # 🔥 对应 4 个图像输出组件和 2 个文本输出 (注意添加了 disease_class_msg)
                outputs=[out_a_yolo, out_b_yolo, out_a_iso, out_b_iso, disease_class_msg, status_msg]
            )

        # with gr.Tab("📈 模型训练数据面板"):
        #     gr.Markdown("动态生成学术级训练对比曲线。")
        #     with gr.Row():
        #         with gr.Column(scale=1):
        #             gr.Markdown("### ⚙️ 图表设置")
        #             models_checkbox = gr.CheckboxGroup(
        #                 choices=list(MODEL_DICT.keys()), 
        #                 value=list(MODEL_DICT.keys())[-2:], 
        #                 label="1. 选择要对比的模型"
        #             )
        #             metric_dropdown = gr.Dropdown(
        #                 choices=list(METRIC_MAP.keys()), 
        #                 value="mAP@0.5", 
        #                 label="2. 选择对比指标"
        #             )
        #             plot_btn = gr.Button("📊 生成对比曲线图", variant="primary")
                
        #         with gr.Column(scale=3):
        #             plot_output = gr.Plot(label="训练趋势图 (Training Curves)")

        #     plot_btn.click(
        #         fn=plot_training_curves,
        #         inputs=[models_checkbox, metric_dropdown],
        #         outputs=[plot_output]
        #     )
        #     metric_dropdown.change(
        #         fn=plot_training_curves,
        #         inputs=[models_checkbox, metric_dropdown],
        #         outputs=[plot_output]
        #     )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)