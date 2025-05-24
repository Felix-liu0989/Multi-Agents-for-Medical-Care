from openai import OpenAI
import pprint
import os
import dotenv
import time

from .Base import BaseModel

dotenv.load_dotenv()
api_key = "sk-3b9c860b83844a789ea639881a569acc"
# "sk-1242a45170f84b089e04143c15822233"
# "sk-3b9c860b83844a789ea639881a569acc" ds
base_url = "https://api.deepseek.com/v1"
# "https://dashscope.aliyuncs.com/compatible-mode/v1" # qwen

# "https://api.deepseek.com/v1" ds
from openai import OpenAI

with open("./datatsets/prompt.txt","r",encoding="utf-8") as f:
    prompt = f.read()
    
with open("./datatsets/prompt_for_retrieval.txt","r",encoding="utf-8") as f:
    prompt_for_retrieval = f.read()

def build_simple_template():
    prompt_template = prompt + "\n" + "病例信息：\n" + "{}\n"
    return prompt_template

def build_retrieval_template():
    prompt_template = "以下是外部文档：\n" + "{}\n" + "病例信息：\n" + "{}\n"
    return prompt_template


class Deepseek(BaseModel):
    def __init__(self,temperature=0):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.simple_template = build_simple_template()
        self.retrieval_template = build_retrieval_template()

    def prompt(self,context,query):
        try:
            response = self.client.chat.completions.create(
                model = "deepseek-chat",
                messages=[
                    {"role":"system","content":prompt_for_retrieval},
                    {"role": "user", "content":self.retrieval_template.format(context,query)}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(2)
        
        return ""