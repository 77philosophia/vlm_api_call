import gradio as gr
import os
import time
# 导入你的 QwenRequester 类和相关的 dashscope 库
import dashscope
from dashscope import MultiModalConversation
from utils import get_file_url, get_image_size, draw_bbox_on_image
from history_manager import HistoryManager
from qwen_requester import QwenRequester
from qwen3_vl_2d import plot_bounding_boxes

from dotenv import load_dotenv
load_dotenv() # 这会加载 .env 文件中的变量到 os.environ
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# --- 初始化历史管理器 ---
history_manager = HistoryManager()

# --- Gradio 界面函数 ---

# ❗️ 恢复 system_prompt 参数
def gradio_qwen_call(api_key, input_image_path, question, system_prompt):
    """Gradio 接口函数，用于连接 UI 输入和 QwenRequester 逻辑。"""
    
    if not api_key:
        return "错误：请输入 Qwen API Key。", "Token 信息：API Key 缺失"
    
    if not input_image_path:
        return "错误：请上传图像。", "Token 信息：图像缺失"

    # 1. 初始化 Requester
    try:
        requester = QwenRequester(api_key=api_key)
    except Exception as e:
        return f"错误：初始化 QwenRequester 失败。\n{e}", "Token 信息：初始化失败"

    # 2. 调用请求函数 (传入 system_prompt)
    response_text, token_info = requester.request_qwen(
        question=question, 
        image_path=input_image_path,
        system_prompt=system_prompt # 传递 UI 输入的 system_prompt
    )
    
    # 3. 保存到历史记录
    # history_manager.add_record(input_image_path, question, system_prompt, response_text, token_info)
    
    return response_text, token_info, input_image_path, question, system_prompt

# main.py
def save_history_record(original_image_path, question, system_prompt, model_response, token_info, saved_annotated_image_path):
    """
    负责将所有信息（包括保存的图片路径）保存到历史记录。
    """
    # 假设 history_manager.add_record 已经更新，可以接受保存的图片路径
    
    # ⚠️ 你需要确保你的 history_manager.add_record 函数可以接受这个额外的参数
    history_manager.add_record(
        original_image_path, 
        question, 
        system_prompt, 
        model_response, 
        token_info, 
        saved_annotated_image_path # 传入保存路径
    )
    
    # 返回一个状态信息给用户
    return f"历史记录已保存。标注图路径: {saved_annotated_image_path}"

# --- Gradio 界面定义 (恢复 System Prompt 输入框，默认值为空) ---

with gr.Blocks(title="Qwen-VL 机器人技能决策") as demo:
    gr.Markdown("## 🤖 Qwen-VL 机器人技能决策系统")
    gr.Markdown("基于达摩院 Qwen-VL Plus 模型，根据实时图像判断机器人下一步应执行的技能。")

    with gr.Tab("主界面"):
        with gr.Row():
            # 左侧配置区域
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="Qwen API Key (sk-...)", 
                    type="password", 
                    placeholder="在此输入您的达摩院 API Key",
                    interactive=True,
                    value=DASHSCOPE_API_KEY
                )
                
                image_input = gr.Image(
                    type="filepath", 
                    label="机器人工作区实时图像 (RGBD)", 
                    height=250
                )
                
                # 新增：用于显示图片尺寸的 Textbox
                image_size_output = gr.Textbox(
                    label="图像尺寸 (Width x Height)", 
                    lines=1,
                    interactive=False
                )
                
                question_input = gr.Textbox(
                    label="VLM 提问/指令",
                    # 移除 f 前缀
                    value="""请检测图片中的黑色线缆。

                要求以JSON格式返回检测结果，格式如下:
                [
                        {"bbox_2d": [165, 260, 624, 649], "label": "松鼠桂鱼"},
                        {"bbox_2d": [743, 208, 989, 451], "label": "白灼生菜"},
                        {"bbox_2d": [484, 519, 999, 958], "label": "蒜蓉开背虾"},
                        {"bbox_2d": [40, 772, 451, 1000], "label": "青椒肉丝"},
                        {"bbox_2d": [447, 18, 721, 143], "label": "海鲜煲"},
                        {"bbox_2d": [306, 35, 406, 117], "label": "米饭"}
                ]""",
                    lines=2
                )
                
                # ❗️ 恢复 System Prompt 输入框，并设置 value 为空
                system_prompt_input = gr.Textbox(
                    label="自定义 System Prompt (默认为空)",
                    value="", # 默认值为空
                    lines=10,
                    placeholder="在此输入 VLM 的角色设定、约束和详细指令（如技能库）。如果留空，将只发送问题和图片。",
                    show_copy_button=True,
                    autoscroll=True
                )

                submit_btn = gr.Button("🚀 执行技能决策 (调用 Qwen-VL)", variant="primary")

            # 右侧输出区域
            with gr.Column(scale=2):
                output_result = gr.Textbox(
                    label="VLM 决策结果 (技能编号)", 
                    lines=5, 
                    show_copy_button=True, 
                    autoscroll=True
                )
                
                token_output = gr.Textbox(
                    label="Token & 耗时统计", 
                    lines=5,
                    autoscroll=True
                )

                # 新增：用于显示带 BBOX 的图片
                annotated_image_output = gr.Image(
                    label="检测结果 (带 BBOX)",
                    interactive=False, # 不允许用户交互修改
                    height=300
                )

                # 新增：用于显示 BBOX 绘制状态
                bbox_status_output = gr.Textbox(
                    label="BBOX 绘制状态",
                    lines=1,
                    interactive=False
                )

    # ❗️ 移除 System Prompt 配置 Tab

    # 新增历史记录页面
    with gr.Tab("历史记录"):
        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("🔄 刷新历史记录", variant="secondary")
                clear_btn = gr.Button("🗑️ 清空历史记录", variant="stop")
            with gr.Column():
                history_status = gr.Textbox(
                    label="操作状态",
                    interactive=False
                )
        
        history_output = gr.HTML(
            label="调用历史记录",
            value=history_manager.load_history_records
        )

    # --- 按钮点击事件绑定 ---
    submit_btn.click(
        fn=gradio_qwen_call,
        # ❗️ 恢复 system_prompt_input
        inputs=[api_key_input, image_input, question_input, system_prompt_input],
        outputs=[output_result, token_output, gr.State(value=None), gr.State(value=None), gr.State(value=None)]
    ).then(
        fn=plot_bounding_boxes,
        inputs=[image_input, output_result],
        outputs=[annotated_image_output]
    )
    
    # 新增：图片上传/改变时，更新尺寸信息
    image_input.change(
        fn=get_image_size,
        inputs=[image_input],
        outputs=[image_size_output]
    )

    # 历史记录页面按钮事件
    refresh_btn.click(
        fn=history_manager.load_history_records,
        outputs=history_output
    )
    
    clear_btn.click(
        fn=history_manager.clear_history,
        outputs=[history_status, history_output]
    )
    
    # --- 示例 ---
    # 注意：请确保这些示例图片路径在你运行 Gradio 的环境中是存在的！
    example_path = "qwen_pictures/left_side/000000.png"
    if os.path.exists(example_path):
        gr.Examples(
            examples=[
                # ❗️ 示例中 System Prompt 也应为空
                ["YOUR_API_KEY", example_path, "我接下来应该调用哪个技能?", ""],
            ],
            # ❗️ 恢复 system_prompt_input
            inputs=[api_key_input, image_input, question_input, system_prompt_input],
            outputs=[output_result, token_output],
            label="示例 (请先替换 YOUR_API_KEY)"
        )


if __name__ == '__main__':
    DEFAULT_PORT = 7870
    print(f"Gradio App 正在启动，请在浏览器中访问 http://127.0.0.1:{DEFAULT_PORT}")
    demo.launch(server_port=DEFAULT_PORT)
    # demo.launch(server_port=DEFAULT_PORT,share=True)    稳定后生成固定链接