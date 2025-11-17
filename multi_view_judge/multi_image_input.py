import gradio as gr
import dashscope
import os
from dotenv import load_dotenv

# 加载环境变量 (确保你的 .env 文件中有 DASHSCOPE_API_KEY)
load_dotenv() 

# --- 配置 ---
# 使用 Qwen3-VL 模型
QWEN_MODEL_NAME = 'qwen3-vl-plus-2025-09-23' 
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

# 若使用新加坡地域的模型，取消下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# 默认的分析指令
DEFAULT_PROMPT = "请结合四张图片，详细描述目标物体的当前状态和所有可见的细节。如果物体有运动，请说明运动趋势。"

def format_image_for_dashscope(image_path: str) -> str:
    """将本地文件路径格式化为 DashScope API 要求的 file:// 格式"""
    # 确保路径是绝对路径
    absolute_path = os.path.abspath(image_path)
    return f"file://{absolute_path}"

def multi_camera_analysis_four_views(
    left_image_file: str, 
    right_image_file: str, 
    pano_image_file: str, # 新增
    bottom_image_file: str, # 新增
    prompt_text: str
) -> str:
    """
    接收四张图片的文件路径和文本，调用 DashScope Qwen-VLM API 进行分析。
    """
    
    # 1. 检查 API Key
    if not DASHSCOPE_API_KEY:
        return "错误：未找到 'DASHSCOPE_API_KEY' 环境变量。请在 .env 文件中配置。"

    # 2. 检查输入文件
    if not all([left_image_file, right_image_file, pano_image_file, bottom_image_file]):
        return "错误：请上传所有四个相机（左侧、右侧、全景、底部）的图片。"

    # 3. 格式化图片路径为 DashScope 要求的 'file://' 格式
    try:
        image_paths = [
            format_image_for_dashscope(left_image_file),
            format_image_for_dashscope(right_image_file),
            format_image_for_dashscope(pano_image_file),
            format_image_for_dashscope(bottom_image_file),
        ]
    except Exception as e:
        return f"文件路径处理错误：{e}"

    # 4. 构造 DashScope API 的 messages 结构
    # 使用交错的方式描述图片，以提供上下文
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_paths[0]},
                {"text": "第一张是左视图。"},
                {"image": image_paths[1]},
                {"text": "第二张是右视图。"},
                {"image": image_paths[2]},
                {"text": "第三张是全景图。"},
                {"image": image_paths[3]},
                {"text": "第四张是底部视图。"},
                {"text": f"请结合这四张图片提供的信息，回答以下问题：{prompt_text}"}
            ]
        }
    ]

    print(f"正在调用 DashScope API，模型：{QWEN_MODEL_NAME}...")

    # 5. 调用 DashScope API
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL_NAME,
            messages=messages
        )
        
        # 6. 处理并返回结果
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            error_msg = f"DashScope API 调用失败。状态码: {response.status_code}\n"
            error_msg += f"错误信息: {response.code} - {response.message}"
            return error_msg
            
    except Exception as e:
        return f"API 调用或网络错误：{e}"

# --- Gradio 界面搭建 ---
with gr.Blocks(title="DashScope Qwen-VLM 四相机目标状态判断") as demo:
    gr.Markdown("# 👁️‍🗨️ DashScope Qwen-VLM 四相机目标状态判断应用")
    gr.Markdown("请上传来自**左侧、右侧、全景和底部相机**的四张图片，并提供一个**分析指令**。")

    # 第一行：左右视图
    with gr.Row():
        image_input_left = gr.Image(type="filepath", label="1. 左侧相机图像 (Left)", sources=['upload'], width=300)
        image_input_right = gr.Image(type="filepath", label="2. 右侧相机图像 (Right)", sources=['upload'], width=300)

    # 第二行：全景和底部视图
    with gr.Row():
        image_input_pano = gr.Image(type="filepath", label="3. 全景相机图像 (Panoramic)", sources=['upload'], width=300)
        image_input_bottom = gr.Image(type="filepath", label="4. 底部相机图像 (Bottom)", sources=['upload'], width=300)


    # 文字输入框，设置默认值
    prompt_input = gr.Textbox(
        label="分析指令/问题", 
        value=DEFAULT_PROMPT, # 设置默认的 PROMPT
        lines=3
    )

    # 按钮
    submit_button = gr.Button("🚀 调用 Qwen-VLM 进行状态判断")

    # 输出框
    output_text = gr.Textbox(label="模型分析结果", lines=10)

    # 绑定事件
    submit_button.click(
        fn=multi_camera_analysis_four_views,
        inputs=[image_input_left, image_input_right, image_input_pano, image_input_bottom, prompt_input],
        outputs=output_text
    )
    
    gr.Markdown(f"--- \n使用的模型：`{QWEN_MODEL_NAME}` | 提示：请确保你的 `.env` 文件中配置了有效的 `DASHSCOPE_API_KEY`。")

# 运行 Gradio 应用
if __name__ == "__main__":
    if not DASHSCOPE_API_KEY:
        print("\n!!! 警告：DASHSCOPE_API_KEY 未设置。应用将启动，但无法调用 API。!!!\n")
        
    demo.launch(inbrowser=True)