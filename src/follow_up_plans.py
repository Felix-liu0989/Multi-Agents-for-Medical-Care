import json
import re
from openai import OpenAI
from typing import Dict, List, Union
from utils import jsonl2json

class FollowUpGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"  # DeepSeek API endpoint
        )

    def generate_follow_up_plan(self, patient_data: Dict) -> Dict:
        """生成个性化随访计划"""
        prompt = self._construct_prompt(patient_data)

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一名专业的医疗助理"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                top_p=0.95,
                max_tokens=4096,
                stream=False
            )

            generated_text = response.choices[0].message.content
            return self._extract_follow_up_content(generated_text, prompt)

        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return {"error": str(e)}

    def _construct_prompt(self, patient: Dict) -> str:
        """构建增强的提示词模板"""
        patient_info = patient.get('patient_info', {})
        disease_info = patient.get('disease_info', '未知')
        main_issues = ', '.join(patient.get('main_issues', [])) or '无'
        key_observations = ', '.join(patient.get('key_observations', [])) or '无'
        treatments = ', '.join(patient.get('possible_treatments', [])) or '无'
        notes = patient.get('notes_info', '无')

        return f"""根据患者的完整病历信息生成结构化的随访计划。
请严格按照以下JSON格式输出，包含以下字段：
{{
  "复诊时间安排": "根据疾病类型和恢复周期建议的具体复诊时间",
  "必要检查项目": ["检查1", "检查2", ...],
  "生活方式调整建议": ["建议1", "建议2", ...],
  "随访计划时间表": [
    {{"时间点": "术后1周", "事项": "伤口检查"}},
    {{"时间点": "术后1月", "事项": "功能评估"}}
  ],
  "短信提醒模板": "友好简洁的短信内容，包含患者姓名和关键提醒"
}}

生成要求：
1. 所有字段都必须填写具体内容，不能为空
2. 内容要基于患者具体病情定制化
3. 时间安排要符合医疗常规
4. 检查项目要有明确的医学依据

患者信息摘要：
├─ 基本信息：{patient_info.get('name', '未知')}，{patient_info.get('age', '未知')}岁，{patient_info.get('gender', '未知')}
├─ 诊断：{disease_info}
├─ 主要问题：{main_issues}
├─ 关键观察指标：{key_observations}
├─ 治疗方案：{treatments}
└─ 出院医嘱：{notes}

请生成上述患者的结构化随访计划："""

    def _extract_follow_up_content(self, response: str, prompt: str) -> Dict:
        """从API响应中提取结构化内容"""
        generated_part = response.strip()

        def preprocess_json(json_str: str) -> str:
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'\/\/.*?$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
            return json_str

        try:
            json_match = re.search(r'\{[\s\S]*\}', generated_part)
            if json_match:
                processed_json = preprocess_json(json_match.group(0))
                return json.loads(processed_json)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            print(f"原始内容: {generated_part}")

        return {
            "复诊时间安排": self._extract_section(generated_part, "复诊时间安排"),
            "必要检查项目": self._extract_list(generated_part, "必要检查项目"),
            "生活方式调整建议": self._extract_list(generated_part, "生活方式调整建议"),
            "随访计划时间表": self._extract_time_table(generated_part),
            "短信提醒模板": self._extract_section(generated_part, "短信提醒模板")
        }

    def _extract_section(self, text: str, keyword: str) -> str:
        """提取特定部分的内容"""
        start_idx = text.find(keyword)
        if start_idx == -1:
            return "未生成相关内容"

        end_idx = text.find('\n', start_idx + len(keyword))
        if end_idx == -1:
            end_idx = len(text)

        content = text[start_idx:end_idx].replace(keyword, '').strip(' :\n')
        return content.split('\n')[0] if '\n' in content else content

    def _extract_list(self, text: str, keyword: str) -> List[str]:
        """提取列表类型内容"""
        section = self._extract_section(text, keyword)
        if section.startswith('[') and section.endswith(']'):
            try:
                return json.loads(section)
            except:
                pass

        items = []
        lines = section.split('\n')
        for line in lines:
            if line.strip().startswith('-') or line.strip().startswith('•'):
                items.append(line.strip()[1:].strip())

        return items if items else [section]

    def _extract_time_table(self, text: str) -> List[Dict[str, str]]:
        """提取时间表内容"""
        start_idx = text.find("随访计划时间表")
        if start_idx == -1:
            return []

        array_match = re.search(r'\[[\s\S]*\]', text[start_idx:])
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except:
                pass

        time_table = []
        current_section = text[start_idx:]
        lines = current_section.split('\n')

        for line in lines[1:]:
            if not line.strip():
                continue

            if '时间点' in line and '事项' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    time_point = parts[0].replace('时间点', '').strip()
                    event = parts[1].replace('事项', '').strip()
                    time_table.append({
                        "时间点": time_point,
                        "事项": event
                    })

        return time_table if time_table else [{"错误": "未能解析时间表"}]


def process_patients(input_path: str, output_path: str, api_key: str) -> Dict:
    """处理所有患者数据"""
    generator = FollowUpGenerator(api_key)
    json_input_path = "/root/autodl-fs/healthRAG/outputs/doctor_advice_test_06_03.json"
    jsonl2json.jsonl2json(input_path,json_input_path)
    try:
        with open(json_input_path, 'r', encoding='utf-8') as f:
            patients = json.load(f)
    except Exception as e:
        print(f"输入文件读取失败: {str(e)}")
        return {}

    results = {}
    for patient in patients:
        patient_id = patient.get('id', 'unknown')
        try:
            print(f"正在处理患者: {patient_id}")
            follow_up_plan = generator.generate_follow_up_plan(patient)
            results[patient_id] = follow_up_plan
            print(f"患者 {patient_id} 的随访计划生成完成")
        except Exception as e:
            print(f"患者 {patient_id} 处理失败: {str(e)}")
            results[patient_id] = {
                "error": str(e),
                "patient_info": patient.get('patient_info', {})
            }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至 {output_path}")
    except Exception as e:
        print(f"结果保存失败: {str(e)}")

    return results


if __name__ == "__main__":
    # 配置参数
    API_KEY = "api-key"  # 替换为你的DeepSeek API密钥
    INPUT_JSON = "doctor_advice_test.json"
    OUTPUT_JSON = "follow_up_plans-test-1.json"

    # 运行处理流程
    print("开始生成随访计划...")
    plans = process_patients(INPUT_JSON, OUTPUT_JSON, API_KEY)
    print(f"处理完成! 成功生成 {len([v for v in plans.values() if 'error' not in v])}/{len(plans)} 个随访计划")