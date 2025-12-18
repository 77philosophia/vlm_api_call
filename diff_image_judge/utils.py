from pathlib import Path
from PIL import Image
import os
import json # 导入 json 库
from PIL import Image, ImageDraw, ImageFont # 导入 PIL 库用于图像处理
import re
import base64

#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_file_url(local_path):
    """确保本地文件路径以 'file://' 格式返回"""
    local_path = Path(local_path).resolve() 
    return f"file://{local_path}"

# --- 新增：获取图片尺寸的辅助函数 ---
def get_image_size(image_path):
    """
    根据图片路径获取图片的尺寸 (Width x Height)。
    """
    if not image_path:
        return "请上传图片"
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return f"{width} x {height}"
    except Exception as e:
        return f"无法读取图片尺寸: {e}"
    

def parse_vlm_response(vlm_response):
    """
    解析VLM响应，提取JSON数据
    """
    if not vlm_response:
        return None
    
    # 尝试直接解析JSON
    try:
        return json.loads(vlm_response)
    except json.JSONDecodeError:
        pass
    
    # 如果直接解析失败，尝试从文本中提取JSON
    try:
        # 使用正则表达式匹配JSON格式
        json_match = re.search(r'\{.*\}', vlm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
    except:
        pass
    
    return None

def draw_bbox_on_image(vlm_response, original_image_path, output_image_path=None):
    """
    解析 VLM 返回的 BBOX 信息，并在原始图片上绘制 BBOX。
    
    参数:
        vlm_response: VLM返回的响应文本
        original_image_path: 原始图片路径
        output_image_path: 输出图片路径，如果为None则不保存
    
    返回:
        tuple: (Image对象, 状态信息)
    """
    if not vlm_response:
        return None, "错误：VLM 响应为空，无法绘制 BBOX。"
    
    if not original_image_path or not os.path.exists(original_image_path):
        return None, "错误：原始图片路径无效，无法绘制 BBOX。"

    try:
        # 解析VLM响应
        response_data = parse_vlm_response(vlm_response)
        print(f"解析出的VLM响应数据: {response_data}")
        if not response_data:
            return None, "错误：无法从VLM响应中解析出有效的JSON数据。"

        # 加载原始图片
        img = Image.open(original_image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        bbox_drawn_count = 0
        
        # 提取检测结果
        detections = []
        if isinstance(response_data, dict):
            if "detections" in response_data:
                detections = response_data["detections"]
            elif "boxes" in response_data:
                detections = response_data["boxes"]
            elif "predictions" in response_data:
                detections = response_data["predictions"]
        elif isinstance(response_data, list):
            detections = response_data
        
        if not detections:
            return img, "警告：VLM 响应中未找到检测信息，将返回原图。"
        
        print(f"检测到的对象数量: {len(detections)}")
        print(f"检测详情: {detections}")

        # 设置字体
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("Arial.ttf", 16)
        except IOError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except IOError:
                try:
                    font = ImageFont.load_default()
                    print("警告：使用默认字体，可能显示效果不佳")
                except:
                    font = None

        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
        
        for i, detection in enumerate(detections):
            color = colors[i % len(colors)]
            
            # 处理不同的bbox键名
            bbox = None
            if "bbox" in detection and len(detection["bbox"]) == 4:
                bbox = detection["bbox"]
            elif "box" in detection and len(detection["box"]) == 4:
                bbox = detection["box"]
            elif "coordinates" in detection and len(detection["coordinates"]) == 4:
                bbox = detection["coordinates"]
            
            if bbox is None:
                continue
                
            # 确保坐标是数值类型
            try:
                x1, y1, x2, y2 = map(float, bbox)
            except (ValueError, TypeError):
                continue
            
            # 验证坐标范围
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            if x1 >= x2 or y1 >= y2:
                continue

            # 获取标签和置信度
            label = detection.get("label", "object")
            confidence = detection.get("confidence", detection.get("score", 1.0))
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 准备标签文本
            text = f"{label}"
            if confidence is not None:
                try:
                    conf_value = float(confidence)
                    text += f" {conf_value:.2f}"
                except (ValueError, TypeError):
                    pass
            
            # 绘制标签背景和文本
            if font:
                try:
                    # 获取文本尺寸
                    bbox_text = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                    
                    # 绘制文本背景
                    draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
                    # 绘制文本
                    draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)
                except:
                    # 如果字体渲染失败，使用简单文本
                    draw.text((x1, y1 - 15), text, fill=color)
            else:
                draw.text((x1, y1 - 15), text, fill=color)
            
            bbox_drawn_count += 1

        if output_image_path:
            img.save(output_image_path)
            print(f"检测结果已保存到: {output_image_path}")

        if bbox_drawn_count == 0:
            return img, "警告：未解析出有效的边界框，将返回原图。"
        else:
            return img, f"成功绘制了 {bbox_drawn_count} 个边界框。"

    except Exception as e:
        return None, f"绘制边界框时发生错误：{str(e)}"