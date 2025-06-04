# 模块（1）HealthRAG

HealthRAG 是一个面向医疗对话场景的 RAG智能医嘱生成系统。项目集成了检索、重排序和大语言模型，能够根据病例自动生成专业医嘱。

## 目录结构

```
healthRAG/
├── src/                # 核心源码（run.py 主程序等）
├── datatsets/          # 数据集（如 sampled_medical_dialogue.json, full_cases.json）
├── outputs/            # 生成结果输出目录
├── faiss_index_bge-m3-06-03/ # Faiss 索引文件
├── utils/              # 工具函数
├── flowcharts/         # 流程图等文档
├── debug.json          # 调试用数据
├── test.mp4            # 测试视频
└── .git/               # 版本控制
```

## 主要功能

- 支持基于向量（emb）或 BM25 的检索
- 支持检索结果重排序（rerank）
- 支持大模型生成医嘱
- 支持断点续跑，已处理的 case 不会重复处理

## 依赖环境

```python
torch==2.2.1
transformers==4.43.0
langchain==0.3.25
jieba==0.42.1
rank-bm25==0.2.2
faiss-gpu==1.7.2
graphviz==0.20.3
openai==1.76.2
streamlit==1.35.0
pillow==11.1.0
tqdm==4.67.1
numpy==1.26.4
```

## 快速开始

1. 安装依赖

```bash
conda create -n healthcare_agents python=3.9
conda activate healthcare_agents
pip install -r requirements.txt
```

2. 运行主程序

```bash
cd src
python run.py \
  --llm_model_name deepseek-chat \
  --emb_model_name_or_path /root/autodl-fs/bge-m3 \
  --rerank_model_name_or_path /root/autodl-fs/bge-reranker-v2-m3 \
  --retrieval_methods emb \
  --corpus_path ../datatsets/sampled_medical_dialogue.json \
  --query_path ../datatsets/full_cases.json \
  --faiss_index_path /root/faiss_index_bge-m3-0523
```

参数说明：

- `--llm_model_name`：大模型名称，默认 deepseek-chat
- `--emb_model_name_or_path`：embedding 模型路径
- `--rerank_model_name_or_path`：rerank 模型路径
- `--retrieval_methods`：检索方法，支持稠密检索和稀疏检索
- `--corpus_path`：病例问答库路径
- `--query_path`：病例测试集路径
- `--faiss_index_path`：faiss 索引路径

3. 输出结果

- 结果保存在 `outputs/doctor_advice_test.jsonl`，每行为一个 JSON，包含生成的医嘱和相关检索文档。

## 问题解决

- 针对输入病例数据非结构化的问题，我们调用大模型对病例进行结构化输出，分为个人信息和疾病总结两类，以下是一个示例：

  ## 示例输入

  ```json
  {
    "姓名": "刘**",
    "性别": "女",
    "年龄": "72岁",
    "入院诊断": "1、左侧椎动脉重度狭窄（支架植入术后） 2、缺血性脑血管病 3、多发脑动脉狭窄（轻-重度） 4、高血压病2级 极高危",
    "主诉": "反复头晕1周。"
  }
  ```

  ## 示例输出

  ```json
  {
    "patient_info": {
      "name": "刘**",
      "age": "72岁",
      "gender": "女"
    },
    "disease_summary": {
      "primary_diagnosis": ["左侧椎动脉重度狭窄", "缺血性脑血管病"],
      "secondary_diagnosis": ["多发脑动脉狭窄", "高血压病2级"],
      "risk_factors": ["极高危"],
      "notes": "支架植入术后状态，主诉反复头晕"
    }
  }
  ```

- 将结构化之后的病例内容进行query改写，使用自然语言问题作为查询，以下是一个示例

  ## 示例输入

  ```json
  {
    "patient_info": {
      "name": "刘**",
      "age": "72岁",
      "gender": "女"
    },
    "disease_summary": {
      "primary_diagnosis": ["左侧椎动脉重度狭窄", "缺血性脑血管病"],
      "secondary_diagnosis": ["多发脑动脉狭窄", "高血压病2级"],
      "risk_factors": ["极高危"],
      "notes": "支架植入术后状态，主诉反复头晕"
    }
  }
  ```

  ## 示例输出

  ​	72岁女性患者同时存在左侧椎动脉重度狭窄和高血压病2级，在生活方式和用药方面有哪些特别建议？药物选择上需要注意什么？

- 检索Agent：为了使得生成的医嘱更加准确，消解可能出现的医学知识幻觉，我们构建了一个10000条的病例问答数据集，分为内科，外科，心血管，男科，妇科等多个科室，使得query查询可以匹配到更精准的专家问诊信息，辅助模型生成准确的医嘱建议。

- 医嘱生成Agent：将病例信息、自然语言问题查询以及检索到的外部文档进行整合，模型生成医嘱建议
