import jsonlines
import json

def jsonl2json(input_path,out_path):


    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    data = [json.loads(line) for line in lines]

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"已将 JSONL 文件转换为标准 JSON 文件，保存在 '{output_path}'。")