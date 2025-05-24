#!/bin/bash


echo "program start..."
CUDA_VISIBLE_DEVICES=0 python run.py \
      --llm_model_name deepseek-chat \
      --emb_model_name_or_path  /root/autodl-fs/bge-m3 \
      --rerank_model_name_or_path /root/autodl-fs/bge-reranker \
      --retrieval_methods bm25,vector \
      --corpus_path ../datatsets/sampled_medical_dialogue.json \
      --query_path ../datatsets/full_cases.json \
      --faiss_index_path /root/faiss_index_bge-m3-0523