from collections import defaultdict
from argparse import ArgumentParser
from runForUI import run

from framework import visualize_patient_data as draw
from framework import visualize_patient_data_for_debug as drawDebug
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from follow_up_plans import process_patients

class frontTools:
    def __init__(self, mode = 'release'):
        self.mode = mode
        self.patients = defaultdict(list)
        self.summaryText = ""
        self.followup = ""
        self.calender = None
        self.pipelinePic = None
        self.parser = ArgumentParser()
        self.output_path = '../outputs/doctor_advice_test_06_03.jsonl'
        self.defaultFollowupPath = '../outputs/follow_up_plans.json'
        self.outputFollowupPath = '../outputs/follow_up_plans.json'
        self.pipelinePath = "../flowcharts"
        self.videoPath = ''
        self.API_KEY = "api-key"
        self.flag = 0
        # self.prepareParser()

    def setAPIKey(self, key):
        self.API_KEY = key

    def prepareData(self, data):
        if self.mode == 'release':
            transData = data
        else:
            with open('../datatsets/full_cases.json', 'r', encoding='utf-8') as file:
                transData = json.load(file)
        for record in transData:
            pid = record.get("id", "未知")
            self.patients[pid].append(record)
    
    def summary(self, data):
        parsers = parser()
        run(parsers, self.API_KEY,data, self.output_path, mode = self.mode)
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if self.mode == 'debug':
                        if record.get("id") != data.get("id"):
                            #print(record.get("id"), data.get("id"))
                            continue
                    self.summaryText = self.summaryText + frontTools.generatePatientNote(record)
                    # self.generateVideo(record)
                    
                except json.JSONDecodeError:
                    print("跳过无效的 JSON 行")

    def advice(self, inData):
        path = self.outputFollowupPath
        if self.mode == 'release':
            process_patients(self.output_path, self.outputFollowupPath, self.API_KEY)
        else:
            path = self.defaultFollowupPath
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for group_id, group_info in data.items():
                if self.mode == 'debug':
                    if int(group_id) != int(inData.get("id")):
                        #print(int(group_id), int(inData.get("id")))
                        continue
                self.followup = self.followup + frontTools.generateDescription(group_id, group_info)
                self.calender = frontTools.generateCalender(group_id, group_info)

    def pipeline(self, data):
        if self.mode == 'release':
            draw(self.output_path, self.pipelinePath)
        else:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get("id") != data.get("id"):
                            continue
                        drawDebug(record, self.pipelinePath)
                    except json.JSONDecodeError:
                        print("跳过无效的 JSON 行")
        self.pipelinePath = f"{self.pipelinePath}/patient_{data.get('id')}_flowchart.png"

    
    @staticmethod
    def generatePatientNote(json_line: str) -> str:
        data = json_line
        parts = []

        def add_section(title, content):
            if content:
                parts.append(f"**{title}：** {'；'.join(content) if isinstance(content, list) else content}。")

        add_section("患者住院期间的主要诊断", data.get("main_issues"))
        add_section("医生观察重点", data.get("key_observations"))
        add_section("当前暂无特别异常指标，指标方面表现为", data.get("normal_indicators"))
        add_section("治疗建议包括", data.get("possible_treatments"))
        add_section("请注意", data.get("important_note"))
        add_section("建议每日进行以下监测", data.get("monitoring"))
        add_section("生活方式方面建议", data.get("lifestyle_changes"))
        add_section("若出现以下情况需及时就医", data.get("warning_signs"))

        markdown_text = "\n\n".join(parts)

        return f'{markdown_text}'
    
    def generateVideo(self, data):
        patient_info = data.get("patient_info")

        if patient_info:
            # 获取 name，如果不存在则返回 None
            patient_name = patient_info.get("name")
            if patient_name is None or patient_name == '':
                name = '某'
            else:
                name = patient_name[:1]  # 取名字的第一个字符

            # 获取 gender，如果不存在则返回 None
            patient_gender = patient_info.get("gender")
            if patient_gender is None or patient_gender == '':
                gender = '男'
            else:
                gender = patient_gender
        else:
            # 如果 patient_info 不存在，设置默认值
            name = '某'
            gender = '男'
        monitoring = ''
        lifestyle_changes = ''
        warning_signs = ''
        for item in iter(data.get("monitoring")):
            monitoring += item
        for item in iter(data.get("lifestyle_changes")):
            lifestyle_changes += item
        for item in iter(data.get("warning_signs")):
            warning_signs += item
        content = monitoring + '。' + '在生活方式上，' + lifestyle_changes + '。' + '特别注意：若出现' + warning_signs + '尽快就医。'
        if self.mode == 'release':
            from ..video_generater import HealthVideoGenerator
            vd = HealthVideoGenerator(name, gender, content)
            vd.run()
            self.videoPath = vd.video_path
        else:
            print(name,gender,content)
            self.videoPath = '../test.mp4'

    @staticmethod
    def generateDescription(group_id, group_info):
        lines = []

        if "复诊时间安排" in group_info:
            lines.append(f"**复诊时间安排**：{group_info['复诊时间安排']}")

        if "必要检查项目" in group_info:
            checks = "、".join(group_info["必要检查项目"])
            lines.append(f"**必要检查项目**：{checks}。")

        if "生活方式调整建议" in group_info:
            lifestyle = "；".join(group_info["生活方式调整建议"])
            lines.append(f"**生活方式调整建议**：{lifestyle}。")

        return "\n\n".join(lines)
    
    @staticmethod
    def generateCalender(group_id, group_info):
        calendar = dict()
        if "随访计划时间表" in group_info:
            for item in group_info["随访计划时间表"]:
                time = item.get("时间点", "")
                matter = item.get("事项", "")
                calendar.update({time: matter})
        return calendar

class parser:
    def __init__(self):
        self.llm_name = 'deepseek-chat'
        self.emb_model_name_or_path = '/root/autodl-fs/bge-m3'
        self.rerank_model_name_or_path = '/root/autodl-fs/bge-reranker'
        self.corpus_path = '../datatsets/sampled_medical_dialogue.json'
        self.query_path = '../datatsets/full_cases.json'
        self.faiss_index_path = '/root/autodl-fs/healthRAG/faiss_index_bge-m3-06-03'
        self.retrieval_methods = ['bm25', 'vector']
    
# ,llm_name,emb_model_path,rerank_model_path,corpus_path,query_path,faiss_index_path,retrieval_mathod


def genText(patient_records):
    return "主要病症是xxx", "用药方法是xxx", "康复注意事项是xxx"

def genPipeline(patient_records):
    list = []
    return list

def genScript(patient_records):
    return "标题：术后注意事项"+'\n'+"1. 开场画面：病人在病房内休息。"+'\n'+"2. 医生出场，提醒保持切口清洁、定期换药。"+'\n'+"3. 动画展示：出院后 7 天内避免剧烈运动。"+'\n'+"4. 结尾：护士微笑提醒：“如有不适，请及时就诊。"+'\n'+"（本动画由医院宣教系统自动生成）"

def genVideo(patient_records):
    return "test.mp4"

def genAdvice(patient_records):
    dayList = [7, 14, 21]
    adviceList = ["复查血常规、血压监测", "门诊随访：术后康复评估", "复查B超、心电图"]
    return dayList, adviceList

def genCommand(patient_records):
    return "医嘱"

def generate_education_content(patient_records, gen_func, gen_script, gen_video):
    """
    模拟：根据患者记录生成宣教内容、图片与动画脚本。
    """
    name = None

    for entry in patient_records:
        name = name or entry.get("姓名", "某患者")

    disease_text, medicine_text, recover_text = gen_func(patient_records) 
    script = gen_script(patient_records)
    video_path = gen_video(patient_records)

    return {
        "text": f"""
### 住院记录总结：

尊敬的{name}，您在住院期间主要诊断包括：
{disease_text}

以下是您住院期间部分护理观察及注意事项：
{medicine_text},
{recover_text},
请您在出院后注意休息，保持良好饮食与睡眠习惯。
""",

        "script": f"""
【动画脚本片段】
{script}
"""
    }, video_path

from datetime import datetime, timedelta

def generate_followup_plan(patient_records, gen_advice, gen_command):
    """
    模拟：根据患者记录生成随访计划与日历。
    """
    dayList, adviceList = gen_advice(patient_records)
    command = gen_command(patient_records)
    start_date = datetime.today()
    calendar = dict()
    for i in range(len(dayList)):
        calendar.update({(start_date + timedelta(days=dayList[i])).strftime("%Y-%m-%d"): adviceList[i]})
    text = f"""
### 出院医嘱
您将在出院后：
{command}
请根据随访计划准时就诊。
"""

    return {
        "text": text,
        "calendar": calendar
    }
