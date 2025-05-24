# reranker.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from langchain.schema import Document

class Reranker:
    """
    Reranker 类，用于基于查询对检索到的文档进行重新排序，
    以提高相关性。
    """
    def __init__(self, rerank_model_name_or_path: str, device: str = 'cuda'):
        """
        初始化 Reranker，加载指定的模型和分词器。

        Args:
            rerank_model_name_or_path (str): 预训练 rerank 模型的名称或路径。
            device (str, optional): 使用的设备，默认为 'cuda'。
        """
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name_or_path)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name_or_path)\
            .to(device).eval()
        self.device = device
        print('成功加载 rerank 模型......')

    def rerank(self, docs: List[Document], query: str, k: int = 5) -> List[str]:
        
        """
        对检索到的文档进行重新排序，返回前 k 个相关性最高的文档。

        Args:
            docs (List[Document]): 检索到的文档列表。
            query (str): 查询字符串。
            k (int, optional): 返回的文档数量，默认为 5。

        Returns:
            List[str]: 前 k 个重新排序后的文档内容。
        """
        
        # 创建 (query, doc) 对的列表
        #print(len(docs))
        #print(f"docs:{docs}")
        
        
        input_pairs = []
        for retrieval_data in docs:
            doc = retrieval_data["output"]
            
            pair = [query,doc]
            input_pairs.append(pair)
        

        # 分词和编码
        inputs = self.rerank_tokenizer(
            input_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            # 获取模型输出
            outputs = self.rerank_model(**inputs,return_dict=True)
            scores = outputs.logits.view(-1,).float()
            scores = scores.tolist()
            # outputs.logits.squeeze().cpu()

            # 根据模型输出调整分数提取方式
            # if logits.dim() > 1:
            #     # 对于多分类模型，相关性得分在第 1 类
            #     scores = logits[:, 1]
            # else:
            #     # 对于二分类或单输出模型，直接使用 logits 作为得分
            #     scores = logits
                
            # model(**inputs, return_dict=True).logits.view(-1, ).float()
            # outputs.logits.squeeze().cpu()
            print(scores)

        # 将文档与得分配对
        doc_scores = [
            {
                "instruction": doc["instruction"],
                "output": doc["output"],
                "score": score
            }
            for doc,score in zip(docs,scores)
        ]
        

        # 按得分降序排序
        sorted_docs = sorted(doc_scores, key=lambda x: x["score"], reverse=True)

        # 选择前 k 个文档
        top_k_docs = sorted_docs[:k]
        
        
        return top_k_docs
    
if __name__ == "__main__":
    from retriever import HealthRetriever
    from read_dataset import Reader
    reader = Reader("/root/autodl-fs/healthRAG/datatsets/sampled_medical_dialogue.json")
    tasks = reader.get_tasks()
    retriever = HealthRetriever(
        emb_model_name_or_path="/root/autodl-fs/bge-m3",
        tasks=tasks,
        device='cuda',
        language='zh',
        faiss_index_path="faiss_index_bge-m3-0523"
    )
    # 48岁女性，舒张压89接近正常上限，需要监测血压变化吗？
    retrieved = retriever.combined_retrieval("患者发热，体温≤38.5℃时物理降温，体温≥38.5℃需要吃退烧药吗？",methods=["bm25","vector"])
    print(f"retrieved:{retrieved}")
    reranker = Reranker(
        rerank_model_name_or_path="/root/autodl-fs/bge-reranker",
        device='cuda'
    )
    print(reranker.rerank(retrieved,query="患者发热，体温≤38.5℃时物理降温，体温≥38.5℃需要吃退烧药吗？"))