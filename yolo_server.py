import mcp
from mcp.server import MCPServer
from pathlib import Path
import os
import json
from typing import Annotated,List,Dict,Union
from pydantic import Field
import threading
import base64
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import ImageDraw,ImageFont,Image
import glob
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torch
import gc
import sys

mcp = MCPServer("MAN_MCP")

BASE_DIR = Path(__file__).resolve().parent
ocr_model = None
text_recognition_model=None
ocr_model_path = BASE_DIR/"GLM-OCR"
text_recognition_model_path = BASE_DIR/"comic-text-and-bubble-detector"



def loadTextRecognitionModel():
    global text_recognition_model
    if text_recognition_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RTDetrForObjectDetection.from_pretrained(text_recognition_model_path).to(device)  # type: ignore[arg-type]
        processor = RTDetrImageProcessor.from_pretrained(text_recognition_model_path)
        text_recognition_model = (model, processor)

def load_ocr_model():
    global ocr_model
    if ocr_model is None:
        ocr_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=ocr_model_path,
            torch_dtype="auto",
            device_map="auto",
        )

def release_model():
    """释放文字检测模型，清理显存和内存"""
    global text_recognition_model
    if text_recognition_model is not None:
        del text_recognition_model
        text_recognition_model = None
    global ocr_model
    if ocr_model is not None:
        del ocr_model
        ocr_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Text detection model released.")

def detect_text_regions(image_path: str) -> List[Dict]:
    """返回检测到的文字区域列表，每个元素包含 label='text' 和 box (xyxy)"""
    global text_recognition_model
    if text_recognition_model is None:
        loadTextRecognitionModel()
    assert text_recognition_model is not None
    model, processor = text_recognition_model

    # 加载图片
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    # 将输入也移动到与模型相同的设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 解析结果，设置置信度阈值（可调整）
    target_sizes = torch.tensor([image.size[::-1]]).to(model.device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5 # type: ignore[arg-type]
    )

    #存疑
    detections = []
    id2label = model.config.id2label
    if id2label is None:
        raise RuntimeError("模型配置缺少 id2label，无法解析类别名称。")
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            label = id2label[label_id.item()]
            # 只保留文字区域：对话框内文字 或 自由文字
            if label not in ["text_bubble", "text_free"]:
                continue
            bbox = box.tolist()  # [x1, y1, x2, y2]
            detections.append({"label": label, "box": bbox})
    return detections


def useOcrModel(image_input: Union[str, Image.Image]):
    global ocr_model
    if ocr_model is None:
        load_ocr_model()
    assert ocr_model is not None 
    processor = AutoProcessor.from_pretrained(ocr_model_path)
    model = ocr_model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": image_input
                },
                {
                    "type": "text",
                    "text": "Text Recognition:"
                }
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    generated_ids = model.generate(**inputs, max_new_tokens=8192) # type: ignore[attr-defined]
    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return output_text


def process_image_with_ocr(image_path: str,save_vis_path: str|None = None) -> List[Dict]:
    """
    对一张图片执行检测 + 裁剪 + OCR，并可选择保存可视化结果。
    
    参数:
        image_path: 输入图片路径
        save_vis_path: 如果提供（如 "output/result.jpg"），则保存带标注的图片
    返回:
        包含 label, box, text 的列表
    """
    # 1. 打开原图（用于裁剪和绘制）
    original_img = Image.open(image_path).convert("RGB")
    # 为了绘图，复制一份（避免修改原图用于后续裁剪，实际上裁剪用 original_img，绘制用 draw_img）
    draw_img = original_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # 尝试加载默认字体（如果系统没有，PIL 会报错，则使用默认无字体）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # 2. 获取 YOLO 检测结果
    detections = detect_text_regions(image_path)
    
    # 类别颜色映射
    color_map = {
        "frame": "blue",
        "text": "green",
        "text_free": "orange",
    }
    default_color = "red"

    results = []
    for det in detections:
        box = det["box"]  # [x1, y1, x2, y2]
        label = det["label"]
        color = color_map.get(label, default_color)
        
        # 3. 裁剪子图送 OCR
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_img.width, x2), min(original_img.height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
            
        cropped = original_img.crop((x1, y1, x2, y2))
        try:
            ocr_text = useOcrModel(cropped)
        except Exception as e:
            ocr_text = f"[Error]"
        
        # 4. 在 draw_img 上绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 5. 在框上方绘制标签和 OCR 文本（限制显示长度，防止溢出）
        display_text = f"{label}: {ocr_text[:20]}"  # 截断过长文本
        # 计算文本位置（框的左上角往上偏移）
        text_position = (x1, y1 - 25 if y1 - 25 > 0 else y1 + 5)
        # 为了文字清晰，先画个白色背景框
        bbox = draw.textbbox(text_position, display_text, font=font)
        draw.rectangle(bbox, fill="black")
        draw.text(text_position, display_text, fill="white", font=font)

        results.append({
            "label": label,
            "coords": box,
            "text": ocr_text
        })

    # 6. 如果指定了保存路径，保存带标注的图片
    if save_vis_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
        draw_img.save(save_vis_path)
        print(f"Visualization saved to {save_vis_path}")

    return results



def get_sorted_image_paths(
    directory: Path,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
) -> List[str]:
    """
    扫描目录，获取所有指定扩展名的图片路径，并按文件名自然排序（不区分大小写）。
    """
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    # 去重并按文件名排序（自然顺序）
    paths = sorted(set(paths), key=lambda x: os.path.basename(x))
    return paths



def load_ts_json(json_path):
    """读取 ts.json，返回按页码组织的列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def draw_text_in_box(
    draw,
    box,
    label,
    text,
    textLayout,
    font_path,
    max_font_size=60,
    min_font_size=10,
    padding=-4,
    fill_color=(0, 0, 0, 255),
):
    x1, y1, x2, y2 = map(int, box)
    box_width, box_height = x2 - x1, y2 - y1
    chars = list(text)

    if box_width <= 0 or box_height <= 0 or not chars:
        return

    x1 = x1 - padding
    y1 = y1 - padding
    x2 = x2 + padding
    y2 = y2 + padding

    if label == "text_free":
        region = draw._image.crop((x1, y1, x2, y2))
        overlay = Image.new('RGBA', region.size, (255, 255, 255, 200))
        blended = Image.alpha_composite(region, overlay)
        draw._image.paste(blended, (x1, y1))
    else:
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(255, 255, 255, 255),
        )

    def load_font(size):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            return ImageFont.load_default()

    def measure_font(font):
        left, top, right, bottom = draw.textbbox((0, 0), "中", font=font)
        return right - left + 2, bottom - top + 2

    total_chars = len(chars)

    if textLayout == "horizontal":
        # ---------- 水平排版：从左到右，从上到下 ----------
        for font_size in range(max_font_size, min_font_size - 1, -2):
            font = load_font(font_size)
            char_width, char_height = measure_font(font)
            chars_per_line = max(1, box_width // char_width)
            lines = (total_chars + chars_per_line - 1) // chars_per_line
            if lines * char_height <= box_height:
                break
        else:
            font = load_font(min_font_size)
            char_width, char_height = measure_font(font)
            chars_per_line = max(1, box_width // char_width)
            lines = (total_chars + chars_per_line - 1) // chars_per_line

        total_height = lines * char_height
        start_y = y1 + (box_height - total_height) // 2

        for line_idx in range(lines):
            start = line_idx * chars_per_line
            end = min(start + chars_per_line, total_chars)
            line_chars = chars[start:end]
            line_width = len(line_chars) * char_width
            start_x = x1 + (box_width - line_width) // 2
            y = start_y + line_idx * char_height
            for i, char in enumerate(line_chars):
                x = start_x + i * char_width
                draw.text((x, y), char, font=font, fill=fill_color)
    else:
        # ---------- 垂直排版：从右到左，每列从上到下（原有逻辑） ----------
        for font_size in range(max_font_size, min_font_size - 1, -2):
            font = load_font(font_size)
            char_width, char_height = measure_font(font)
            rows = max(1, box_height // char_height)
            columns = (total_chars + rows - 1) // rows
            if columns * char_width <= box_width:
                break
        else:
            font = load_font(min_font_size)
            char_width, char_height = measure_font(font)
            rows = max(1, box_height // char_height)
            columns = (total_chars + rows - 1) // rows

        total_width = columns * char_width
        start_x = x1 + (box_width - total_width) // 2
        start_y = y1 + (box_height - rows * char_height) // 2

        for column in range(columns):
            x = start_x + (columns - 1 - column) * char_width
            start = column * rows
            end = min(start + rows, total_chars)
            for row, char in enumerate(chars[start:end]):
                y = start_y + row * char_height
                draw.text((x, y), char, font=font, fill=fill_color)


def process_page(image_path, context_list, output_path, font_path):
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    for item in context_list:
        translation = item.get("translation", "")
        if not translation:
            continue
        boxes = item.get("boxes", [])
        textLayout=item.get("layout","horizontal")
        if not boxes:
            continue
        for box in boxes:
            coords = box.get("coords")
            lable = box.get("label")
            draw_text_in_box(draw, coords, lable, translation,textLayout,font_path)
    img.convert("RGB").save(output_path)




@mcp.tool(name="extract_boxes_process", description="从原始图片中提取气泡和自由文本，输出矩形框信息。除非用户指定否则不可调用。")
def extract_boxes_process(
    input_dir: Annotated[Path, Field(description="图片所在目录。")],
    output_json:Annotated[Path, Field(description="结果保存的文件，json格式。")],
    save_vis_dir: Annotated[Path, Field(default=None,description="保存中间产生的可视化结果的目录。")],
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
):
    # 1. 获取所有图片文件并按文件名排序（确保顺序稳定）
    image_paths = get_sorted_image_paths(input_dir, extensions)
    if not image_paths:
        print(f"在目录 {input_dir} 中未找到任何图片")
        return []

    all_results = []
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"处理第 {idx} 张图片: {img_path}")

        # 如果指定了可视化保存目录，则构造保存路径
        vis_path = None
        if save_vis_dir:
            os.makedirs(save_vis_dir, exist_ok=True)
            base = os.path.basename(img_path)
            vis_path = os.path.join(save_vis_dir, f"vis_{idx}_{base}")

        # 调用 OCR 处理函数（会返回检测框及识别文本）
        detections = process_image_with_ocr(img_path, save_vis_path=vis_path)

        # 保存该页结果
        all_results.append({
            "page": idx,
            "boxes": detections
        })

    # 2. 保存为 JSON 文件
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    release_model()
    print(f"结果已保存至 {output_json}")




@mcp.tool(name="apply_comic_lettering", description="应用使用嵌字信息。除非用户指定否则不可调用。")
def apply_comic_lettering(
    json_path:Annotated[Path, Field(description="存有嵌字信息的json")], 
    image_dir:Annotated[Path, Field(description="图片所在目录")],
    output_dir:Annotated[Path, Field(description="结果输出目录")], 
    font_path:Annotated[Path, Field(default=r"C:\Windows\Fonts\simhei.ttf",description="字体文件")]
    ):
    data = load_ts_json(json_path)
    os.makedirs(output_dir, exist_ok=True)
    image_paths = get_sorted_image_paths(image_dir)
    for image_path, page_info in zip(image_paths, data):
        page_context = page_info.get("context", [])
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, base_name)
        process_page(image_path, page_context, output_path, font_path)
        print(f"Processed {base_name} -> {output_path}")


if __name__ == "__main__": 
    try:
        mcp.run()
    except KeyboardInterrupt:
        pass
    finally:
        # 退出阶段：MCP 通信已经结束，FD 1 不再需要保留
        # 把它永久指向 stderr，挡住 qwen_tts / torch 退出清理的污染
        os.dup2(2, 1)
        sys.stdout = sys.stderr