import re
import tqdm
import pandas as pd
import json

class Reader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._read_data()
        
    def _read_data(self):
        with open(self.file_path,"r",encoding='utf-8') as file:
            data = json.load(file)
            
        return data
            
    def get_tasks(self):
        """
        提取任务：每条数据的指令（instruction）和期望输出（output）
        """ 
        
        tasks = []
        for example in self.data:
            task = {
                "department": example['department'],
                "instruction": example['title']+"\n"+str(example["ask"]),
                #"input": example['input'],
                "output": example['answer']
            }
            tasks.append(task)  
            
        return tasks
if __name__ == "__main__":
    reader = Reader("/root/autodl-fs/healthRAG/datatsets/sampled_medical_dialogue.json")
    tasks = reader.get_tasks()
    print(tasks[0])