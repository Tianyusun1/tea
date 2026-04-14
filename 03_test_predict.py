import os
import glob
from ultralytics import YOLO

# ==========================================
# 1. 路径配置
# ==========================================
# 选一个您刚才训练好的模型的 best.pt 权重路径 (这里以 YOLOv8m 为例)
# 等您跑完自定义的 Tea-YOLO，只需把这里换成 Tea-YOLO 的 best.pt 即可
WEIGHT_PATH = "output/YOLOv8m/weights/best.pt" 

# 测试图片的目录 (直接用您测试集里的 images 文件夹)
TEST_IMAGES_DIR = "/media/dl-d/Data/sty/tea/Tea leaf disease Dataset/Tea Leaf Diseases/test/images"

# 预测结果保存的目录
PREDICT_OUTPUT_DIR = "output/predictions"

def predict_images():
    print(f"👁️ 开始运行推理测试...")
    
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 找不到权重文件: {WEIGHT_PATH}")
        print("请确保您已经成功运行了训练脚本。")
        return

    # 加载训练好的模型
    print(f"📦 加载权重: {WEIGHT_PATH}")
    model = YOLO(WEIGHT_PATH)

    # 随机选几张测试集里的图片来做演示 (这里选前 5 张，您可以自己改数量或指定单张图片绝对路径)
    # 比如您想测单张： image_paths = ["/media/.../test_image_1.jpg"]
    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))[:5] 
    
    if not image_paths:
        print(f"❌ 在 {TEST_IMAGES_DIR} 中没有找到 .jpg 图片。")
        return

    print(f"🖼️ 找到 {len(image_paths)} 张测试图片，开始推理...")

    # 运行预测
    # save=True 会自动把画好框的图片保存下来
    # conf=0.25 意思是只显示置信度大于 25% 的检测框，避免画出太多无关紧要的框
    results = model.predict(
        source=image_paths, 
        save=True,           # 保存结果图片
        project=PREDICT_OUTPUT_DIR, # 保存的主目录
        name="demo_results", # 保存的子目录
        conf=0.25,           # 置信度阈值 (论文里通常写 0.25 或 0.5)
        iou=0.45,            # NMS 交并比阈值
        device=0             # 使用 4090 显卡
    )

    print("\n" + "=" * 60)
    print("✅ 推理完成！")
    print(f"📂 带有检测框的直观效果图已保存至: {os.path.join(PREDICT_OUTPUT_DIR, 'demo_results')}")
    print("💡 您可以去该目录下挑选效果最好的图片，作为【图 4-x 实际检测效果展示】插入到论文中。")
    print("=" * 60)

if __name__ == '__main__':
    predict_images()