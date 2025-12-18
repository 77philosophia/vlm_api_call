import os
import json
from datetime import datetime

# --- 历史记录管理类 ---
class HistoryManager:
    # ... (这部分代码保持不变)
    def __init__(self, history_file="call_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self):
        """加载历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(self):
        """保存历史记录到文件"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史记录失败: {e}")
    
    def add_record(self, image_path, question, system_prompt, response, token_info):
        """添加新的调用记录"""
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path,
            "question": question,
            # 记录 system_prompt 预览，即使它为空
            "system_prompt_preview": system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt,
            "response": response,
            "token_info": token_info
        }
        self.history.insert(0, record)  # 新的记录放在最前面
        # 只保留最近50条记录
        if len(self.history) > 50:
            self.history = self.history[:50]
        self.save_history()
    
    def get_history(self):
        """获取所有历史记录"""
        return self.history
    
    def load_history_records_for_gradio(self):
        history = self.get_history()
        if not history:
            return [], []
        
        data_for_df = []
        image_paths = []

        for record in history:
            data_for_df.append([
                record['timestamp'],
                record['question'],
                record['system_prompt_preview'],
                record['response'],
                record['token_info']
            ])

            if os.path.exists(record['image_path']):
                image_paths.append(record['image_path'])
            else:
                image_paths.append(None)  # 图像文件不存在时使用 None
        return data_for_df, image_paths
    
    def load_history_records(self):
        # ... (这部分代码保持不变)
        """加载历史记录用于显示"""
        history = self.get_history()
        
        if not history:
            return "暂无历史记录"
        
        history_html = """
        <style>
            .history-record {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                background: #f9f9f9;
            }
            .history-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            }
            .history-timestamp {
                color: #666;
                font-size: 0.9em;
            }
            .history-content {
                display: grid;
                grid-template-columns: 200px 1fr;
                gap: 15px;
            }
            .history-image img {
                max-width: 100%;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            .history-text {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .history-question {
                font-weight: bold;
                color: #333;
            }
            .history-response {
                background: white;
                padding: 10px;
                border-radius: 4px;
                border-left: 4px solid #4CAF50;
            }
            .history-token-info {
                font-size: 0.8em;
                color: #666;
                background: #f0f0f0;
                padding: 5px;
                border-radius: 3px;
            }
        </style>
        """
        
        for record in history:
            # 检查图片文件是否存在
            image_html = ""
            image_filename = os.path.basename(record['image_path'])
            if os.path.exists(record['image_path']):
                image_html = f'<img src="file/IMAGES/{image_filename}" alt="输入图像">'
            else:
                image_html = f'<div style="color: #999; text-align: center;">图像文件不存在<br>{record["image_path"]}</div>'
            
            history_html += f"""
            <div class="history-record">
                <div class="history-header">
                    <strong>调用记录</strong>
                    <span class="history-timestamp">{record['timestamp']}</span>
                </div>
                <div class="history-content">
                    <div class="history-image">
                        {image_html}
                    </div>
                    <div class="history-text">
                        <div class="history-question">📝 问题: {record['question']}</div>
                        <div class="history-response">🤖 决策结果: {record['response']}</div>
                        <div class="history-token-info">{record['token_info'].replace(chr(10), '<br>')}</div>
                    </div>
                </div>
            </div>
            """
        
        return history_html
    
    def clear_history(self):
        """清空历史记录"""
        self.history = []
        self.save_history()
        return "历史记录已清空", self.load_history_records()
