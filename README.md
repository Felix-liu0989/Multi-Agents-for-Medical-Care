---

# 模块（1）MediCal-Care RAG

MediCal-Care RAG 是一个面向医疗场景的 RAG系统，用于自动生成高质量的医疗建议或医嘱。

## 目录结构

```
MediCal-Care-RAG/
├── src/                # 核心代码
│   ├── run.py          # 主运行脚本，批量生成医嘱
│   ├── retriever.py    # 检索模块，支持BM25和向量检索
│   ├── reranker.py     # 检索结果重排序模块
│   ├── llm_infer.py    # LLM 推理与生成
│   ├── read_dataset.py # 数据集读取工具
│   └── run.sh          # 运行脚本示例
├── utils/              # 工具脚本
│   └── jsonl2json.py   # JSONL转JSON工具
├── outputs/            # 生成结果输出目录
│   └── doctor_advice_test.jsonl # 生成的医嘱结果
├── datatsets/          # 数据集
│   ├── sampled_medical_dialogue.json # 检索语料
│   ├── full_cases.json             # 测试集
│   ├── prompt.txt/prompt_for_retrieval.txt # 提示词
└── README.md           # 项目说明文档
```

## 环境依赖

- Python 3.9+
- torch
- langchain
- rank_bm25
- tqdm
- transformers
- faiss

## 快速开始

1. **准备数据集**  
   - 检索语料：`datatsets/sampled_medical_dialogue.json`
   - 测试集：`datatsets/full_cases.json`

2. **运行主程序**
   ```bash
   cd src
   python run.py \
     --llm_model_name model_name \
     --emb_model_name_or_path your_local_embedding_model_path\
     --rerank_model_name_or_path your_local_reranker_model_path \
     --retrieval_methods emb \
     --corpus_path ../datatsets/sampled_medical_dialogue.json \
     --query_path ../datatsets/full_cases.json \
     --faiss_index_path your_local_faiss_index_path
   ```

3. **输出结果**  
   生成的医嘱会保存在 `outputs/doctor_advice_test.jsonl` 文件中，每行为一个 JSON 对象，包含原始测试用例、生成的医嘱、检索到的外部文档等信息。

## 核心流程

1. **数据加载**：读取原始病例数据和检索语料。
2. **医疗助手Agent**：考虑到原始病历数据每个格式都不统一。首先会对原始病历做处理，生成结构化的病例信息，候诊科室，诊疗建议等；
3. **Query改写Agent**：由于每个病例信息都是碎片化的，对每个结构化的病例信息进行query改写，更加符合问诊实际情况；
4. **检索Agent**：支持 BM25 和向量检索（可选），返回相关文档。
5. **重排序**：对检索结果用重排序模型排序，筛选高分文档。
6. **医嘱生成Agent**：将高分文档和原始 query 输入 LLM，生成最终医嘱。
7. **保存**：每生成一个结果立即写入输出文件，支持断点续跑。

## 主要模块说明

- `retriever.py`：实现 BM25 和向量检索，支持 FAISS 加速。
- `reranker.py`：对检索结果进行重排序，提升相关性。
- `llm_infer.py`：封装大语言模型推理接口。
- `run.py`：主流程脚本，串联各模块，批量处理测试集。



---
# 模块（2）Health Video Generator

Health Video Generator 是一个面向医疗宣教场景的视频生成系统，结合大语言模型和语音合成服务，自动生成带图像动画、语音和字幕的个性化健康教育视频。

## 目录结构

```
Health-Video-Generator/
├── video_generator.py     # 核心类与运行入口
├── fonts/                        # 中文字体（Noto Sans）
│   ├── NotoSansSC-Regular.ttf
│   └── NotoSansSC-ExtraBold.ttf
├── assets/                       # 视频所用背景图与图像元素
│   ├── background/
│   └── images/
├── audio/                        # 临时生成的语音片段
├── output/                       # 生成的视频与脚本输出
│   ├── final_video.mp4
│   └── scene_script.json
├── .env                          # 环境变量（API 密钥配置）
```

## 环境依赖

- Python 3.8+
- openai
- moviepy
- python-dotenv
- tencentcloud-sdk-python

## 快速开始

1. **配置 API 密钥**

在项目根目录创建 `.env` 文件：

```
OPENAI_API_KEY=sk-xxxx
TENCENT_SECRET_ID=AKIDxxxx
TENCENT_SECRET_KEY=xxxx
```

2. **准备素材**
- 图片素材放入 `assets/background/` 和 `assets/images/`
- 中文字体放入 `fonts/`

3. **运行脚本生成视频**

```bash
python health_video_generator.py
```

4. **查看输出**

- `output/final_video.mp4`：生成的视频
- `output/scene_script.json`：生成的脚本场景

## 核心流程

1. 通过 LLM 生成宣教脚本场景（JSON）
2. 使用腾讯云 TTS 合成语音
3. 自动加载图片素材并生成视频动画
4. 合成字幕与语音
5. 输出 MP4 视频与 JSON 脚本

## 主要模块说明

- `HealthVideoGenerator`：主类，封装所有功能
- `generate_script()`：生成脚本 JSON 场景
- `synthesize_tts()`：语音合成
- `create_scene_clip()`：构建每个场景的图像 + 音频合成片段
- `run()`：执行整体流程，保存结果





