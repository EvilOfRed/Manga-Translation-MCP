import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# 1. 加载模型和处理器
image_processor = RTDetrImageProcessor.from_pretrained("comic-text-and-bubble-detector")
model = RTDetrForObjectDetection.from_pretrained("comic-text-and-bubble-detector")

# 2. 加载本地图片
image_path = r"C:\Comic\dataset2\0007.webp"
image = Image.open(image_path).convert("RGB")

# 3. 推理
inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 4. 解析结果（设置阈值0.5，过滤低置信度）
results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=0.5   # 只保留置信度>0.5的结果，您可根据需要调整
)

# 5. 绘制可视化
draw_image = image.copy()   # 不修改原图
draw = ImageDraw.Draw(draw_image)

# 尝试加载字体（如果系统没有，则使用默认字体）
try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = ImageFont.load_default()

# 颜色映射（可为不同类别分配不同颜色）
color_map = {
    "bubble": "blue",
    "text_bubble": "green",
    "text_free": "orange"
}

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score = score.item()
        label = model.config.id2label[label_id.item()]
        box = [round(i, 2) for i in box.tolist()]  # 浮点坐标
        x1, y1, x2, y2 = map(int, box)  # 转换为整数

        # 绘制矩形框
        color = color_map.get(label, "red")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签（在框的左上角）
        text = f"{label} {score:.2f}"
        # 计算文本位置（如果框太靠上，则绘制在框内下方）
        text_y = y1 - 20 if y1 - 20 > 0 else y1 + 5
        draw.text((x1, text_y), text, fill=color, font=font)

# 6. 保存或显示结果
output_path = r"C:\Comic\output_visualized.jpg"  # 您可自定义路径
draw_image.save(output_path)
print(f"可视化结果已保存至：{output_path}")