import os
import dashscope

# 若使用新加坡地域的模型，请取消下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

QWEN_MODEL_NAME = 'qwen3-vl-plus-2025-09-23'
from dotenv import load_dotenv
load_dotenv() # 这会加载 .env 文件中的变量到 os.environ

# 将xxx/eagle.png替换为你本地图像的绝对路径
left_image_path_1 = "../picture/left_side/000000.png"
left_image_path_1 = f"file://{left_image_path_1}"

right_image_path_1 = "../picture/right_side/000000.png"
right_image_path_1 = f"file://{right_image_path_1}"

messages = [
    {
        "role": "user",
        "content": [
            {"image": left_image_path_1},
            {"image": right_image_path_1},
            {"text": "第一张图是左视图，第二张图是右视图。这些图描绘了什么内容?"}
        ]
    }
]

response = dashscope.MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model=QWEN_MODEL_NAME, # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages=messages
)

print(response.output.choices[0].message.content[0]["text"])