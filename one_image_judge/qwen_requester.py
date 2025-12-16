import dashscope
from dashscope import MultiModalConversation
from utils import get_file_url, encode_image
import time

QWEN_MODEL_NAME = 'qwen3-vl-plus-2025-09-23'

class QwenRequester:
    # ... (这部分代码保持不变)
    def __init__(self, api_key, model_name=QWEN_MODEL_NAME):
        # 仅设置 API Key
        dashscope.api_key = api_key 
        self.api_key = api_key
        self.model_name = model_name

    def create_request_messages_base64(self, question, image_path, system_prompt):
        """
        构造 DashScope SDK 所需的消息列表。
        """
        # file_url = get_file_url(image_path)

        base64_image = encode_image(image_path)
        # ❗️ 逻辑修改：只有当 system_prompt 不为空时，才将其拼接到 question 后面
        if system_prompt and system_prompt.strip():
            full_question = question + "\n\n" + system_prompt
        else:
            full_question = question

        print("28 debug-------------------")
        # print(file_url)
        
        messages = [
            {
                'role': 'user',
                'content': [
                    {"type": "image_url",
                    'image_url': {"url": f"data:image/png;base64,{base64_image}"}},        # 图片文件 URL (使用 file:// 协议)
                    {'type':"text",'text': full_question}     # 文本问题 + (可选的) System Prompt
                ]
            }
        ]
        return messages

        
    def create_request_messages(self, question, image_path, system_prompt):
        """
        构造 DashScope SDK 所需的消息列表。
        """
        file_url = get_file_url(image_path)
        
        # ❗️ 逻辑修改：只有当 system_prompt 不为空时，才将其拼接到 question 后面
        if system_prompt and system_prompt.strip():
            full_question = question + "\n\n" + system_prompt
        else:
            full_question = question

        print("28 debug-------------------")
        print(file_url)
        
        messages = [
            {
                'role': 'user',
                'content': [
                    {'image': file_url},        # 图片文件 URL (使用 file:// 协议)
                    {'text': full_question}     # 文本问题 + (可选的) System Prompt
                ]
            }
        ]
        return messages

    def request_qwen(self, question, image_path, system_prompt):
        """
        接收 system_prompt 参数。
        """
        start_time = time.time()
        
        # 1. 构造消息 (传入 system_prompt)
        messages = self.create_request_messages(question, image_path, system_prompt)
        # base64_image = get_file_url(image_path, return_base64=True)

        # 2. 调用 SDK
        response = MultiModalConversation.call(
            model=self.model_name,
            messages=messages
        )
        
        # 3. 检查并提取结果
        if response.status_code != 200:
            error_message = f"DashScope API 调用失败。Code: {response.code}，Message: {response.message}"
            print(error_message)
            return error_message, f"Status: Failed (Code {response.code})"
            
        try:
            response_text = response["output"]["choices"][0]["message"].content[0]["text"]
        except (KeyError, IndexError):
            error_message = "Error: Failed to parse response content from DashScope."
            print(error_message)
            return error_message, "Status: Failed to parse response"
        
        # 4. 打印并构造 Token 统计信息
        usage = response.get('usage', {})
        input_img_token_num = usage.get('image_tokens', 0)
        input_txt_token_num = usage.get('input_tokens_details', {}).get('text_tokens', 0)
        output_txt_token_num = usage.get('output_tokens_details', {}).get('text_tokens', 0)
        total_token_num = usage.get('total_tokens', 0)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        token_info = (
            f"--- Token 和时间统计 ---\n"
            f"总耗时: {execution_time:.2f} 秒\n"
            f"输入图像的 Token 数: {input_img_token_num}\n"
            f"输入文本的 Token 数: {input_txt_token_num}\n"
            f"输出文本的 Token 数: {output_txt_token_num}\n"
            f"总 Token 数: {total_token_num}"
        )
        
        return response_text, token_info
