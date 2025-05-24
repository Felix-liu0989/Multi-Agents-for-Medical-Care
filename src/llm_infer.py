from openai import OpenAI
import time
import os, json
from tqdm import tqdm
import traceback

with open("../datatsets/prompt.txt","r",encoding="utf-8") as f:
    prompt_for_simple_prediction = f.read()
    
with open("../datatsets/prompt_for_retrieval.txt","r",encoding="utf-8") as f:
    prompt_for_retrieval_prediction = f.read()

def build_simple_template():
    prompt_template = "患者症状：\n" + "{}\n"
    return prompt_template

def build_retrieval_template():
    prompt_template = "## 以下是外部文档，记录了其他患者相似疾病的问诊记录和医嘱,请参考这些信息, 对患者症状给出医嘱：\n" \
                    "请注意外部文档中的问诊记录与患者症状可能存在不一致性，仅作为参考，请根据患者症状给出医嘱。\n" \
                     "{}\n" \
                     "## 患者症状：\n" \
                     "{}\n"
    return prompt_template




API_KEYS = {
    "openai": '',
    'deepseek': "",
}
# os.getenv('DEEPSEEK_API_KEY')

API_URLS = {
    'openai': '',
    'deepseek':'https://api.deepseek.com/v1'
}





class LLMPredictor(object):
    def __init__(self, model_name,temperature=0.5, **kwargs):
        self.max_token = 4096
        self.simple_template = build_simple_template()
        self.retrieval_template = build_retrieval_template()
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        if 'deepseek' in model_name:
            api_key, api_url = API_KEYS['deepseek'], API_URLS['deepseek']
        else:
            raise ValueError(f"Model {model_name} not supported")
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        
    def llm_infer(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def with_retrieval_predict(self, context, query):
        docs = []
        for i, doc in enumerate(context):
            doc_str = f"相关问诊记录 ({i+1})：\n---" \
                      f"{doc['instruction']}\n" \
                      f"相关医嘱 ({i+1})：\n---" \
                      f"{doc['output']}\n"
            docs.append(doc_str)
            doc_strs = '\n'.join(docs)
            
        prompt= self.get_prompt(doc_strs, query)
        # print(f'prompt:{prompt}')
        messages = [
            {"role": "system", "content": prompt_for_retrieval_prediction},
            {"role": "user", "content": prompt} 
        ]
        response = self.llm_infer(messages)
        return response
    

    def get_prompt(self, context, query):

        prompt = self.retrieval_template.format(context, query)

        return prompt

    def simple_predict(self, query):
        prompt = self.simple_template.format(query)
        messages = [
            {"role": "system", "content": prompt_for_simple_prediction},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_infer(messages)
        return response

    def my_llm_infer(self, prompt, device='cuda'):

        raise NotImplementedError