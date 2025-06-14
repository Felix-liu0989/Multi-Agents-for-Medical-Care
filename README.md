# 项目整体运行介绍
```bash
conda create -n healthcare_agents python=3.9
conda activate healthcare_agents
pip install -r requirements.txt
```


- 在项目根目录创建 `.env` 文件：

```bash
touch .env
vim .env
```
- 写入：
```bash
OPENAI_API_KEY=sk-xxxx
TENCENT_SECRET_ID=AKIDxxxx
TENCENT_SECRET_KEY=xxxx
```

```bash
cd healthRAG/src
streamlit run webUI.py 
```

注意: 本项目处理非结构化病例数据的代码并未加入到整个工作流中，但是提供了处理好的病例数据，在datatsets/full_cases.json中，为输入数据，输出则包括医嘱建议，随访计划，流程图，宣教视频等

# 常见问题解决办法
- 如果在视频生成部分出现错误，还需要安装imagemagick，以保证moviepy的正常运行
```bash
wget https://www.imagemagick.org/download/ImageMagick.tar.gz
tar -xzvf ImageMagick.tar.gz
cd ImageMagick-7.1.1-47
chmod +x configure
./configure
make && sudo make install
sudo ldconfig /usr/local/lib
```

- 如果在处理字体和 PDF/PS（PostScript）文件时，缺少 Ghostscript (gs) 和 Freetype 支持。以下是解决方案：

1. 安装 Ghostscript (gs)
ImageMagick 依赖 Ghostscript 处理 PDF/PS 文件。报错 sh: 1: gs: not found 表明系统未安装它。


```bash
sudo apt-get install ghostscript
```
验证是否安装成功：
```bash
gs --version
```


2. 确保 Freetype 支持（处理字体）
报错 delegate library support not built-in 'fonts/NotoSansSC-ExtraBold.ttf' (Freetype) 表明：

ImageMagick 编译时未启用 Freetype，或

系统缺少 Freetype 库。

解决方法：
(1) 安装 Freetype 开发库：

bash
sudo apt-get install libfreetype6-dev
(2) 重新编译 ImageMagick（确保启用 Freetype）：

```bash
cd ~/autodl-fs/ImageMagick-7.1.1-47  # 进入源码目录
make clean                           # 清理旧编译
./configure --with-freetype=yes      # 明确启用 Freetype
make && sudo make install            # 重新编译安装
sudo ldconfig                        # 更新库缓存
```

```bash
magick -list delegate | grep ps      # 检查 Ghostscript 是否被识别
magick -list configure | grep freetype  # 检查 Freetype 支持
```


本项目同时支持各个模块分别运行

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

# 模块（2）Health Video Generator

Health Video Generator 是一个面向医疗宣教场景的视频生成系统，结合大语言模型和语音合成服务，自动生成带图像动画、语音和字幕的个性化健康教育视频。

## 目录结构

```
src/
├── health_video_generator.py     # 核心类与运行入口
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
├── requirements.txt             # Python依赖列表
└── README.md                    # 项目说明文档
```

## 环境依赖

- Python 3.8+
- openai
- moviepy==1.03
- python-dotenv
- tencentcloud-sdk-python
- 下载imagemagick
- 将下载的文件上传到服务器中，或者直接在服务器里面通过wget命令下载
- 
- 解压 tar -xzvf tiff-4.0.8.tar.gz
- 进入目录 cd tiff-4.0.8
- 执行 ./configure
- 执行 make && make install

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
