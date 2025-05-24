import jieba
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS
from typing import List, Tuple
# from read_dataset1 import Reader
"""from langchain.utilities import google_search
search = google_search.GoogleSearchAPIWrapper()"""


class HealthEmbedding(Embeddings):
    """
    嵌入模型，用于生成代码片段或查询的向量表示
    """
    def __init__(self, emb_model_name_or_path, device='cuda', batch_size=64, max_len=512):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len

        # 加载模型和分词器
        self.model = AutoModel.from_pretrained(emb_model_name_or_path).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name_or_path)
        print("成功加载 embedding 模型......")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量处理文本
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # 分词并移动到GPU
            encoded_inputs = self.tokenizer(
                batch_texts, 
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成嵌入
            with torch.no_grad():
                model_output = self.model(**encoded_inputs)
                batch_embeddings = model_output.last_hidden_state[:, 0]
            
            # 归一化
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.extend(batch_embeddings.cpu().tolist())
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # 向量化query
        encoded_inputs = self.tokenizer(
            [text], 
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_inputs)
            embedding = model_output.last_hidden_state[:, 0]
        
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().tolist()[0]


class HealthRetriever:
    """
    代码生成任务的检索器
    """
    def __init__(self, emb_model_name_or_path, tasks: List[dict], device='cuda', language='en',faiss_index_path=None):
        self.device = device
        self.language = language
        # 处理instruction，作为被query查询的语料
        self.documents = []
        for task in tasks:
            instruction = task["instruction"]
            output = task["output"]
            department = task["department"]
            metadata = {
                "instruction": instruction,
                "department": department
            }
            self.documents.append(
                Document(
                    page_content=output,
                    metadata=metadata
                )
            )
        
        
        self.corpus = []
        for doc in self.documents:
            content = {
                "instruction":doc.metadata["instruction"],
                "output":doc.page_content
            }
            self.corpus.append(content)
        
        # 初始化 BM25 检索器
        if language == 'zh':
            tokenized_corpus = [jieba.lcut(doc["instruction"]) for doc in self.corpus]
        else:
            tokenized_corpus = [doc["instruction"].split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 初始化嵌入模型和向量数据库
        self.embedding_model = HealthEmbedding(emb_model_name_or_path, device=device)
        # 如果有提供FAISS索引路径的话，加载FAISS索引
        if faiss_index_path:
            self.vector_db = FAISS.load_local(faiss_index_path,
                                        self.embedding_model,
                                        allow_dangerous_deserialization=True)
        else:
            self.vector_db = FAISS.from_documents(self.documents, self.embedding_model)
    def save_faiss_index(self,faiss_index_path):
        self.vector_db.save_local(faiss_index_path)
    
    def bm25_retrieval(self, query: str, top_k: int = 10) -> List[str]:
        
        # 使用 BM25 方法检索最相关的代码片段
        
        if self.language == 'zh':
            tokenized_query = jieba.lcut(query)
        else:
            tokenized_query = query.split()
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)
        return results
    
    def vector_retrieval(self, query: str, top_k: int = 10) -> List[str]:
        """
        使用向量检索方法获取最相关的代码片段
        """
        results = self.vector_db.similarity_search(query, k=top_k)
        # 去重
        unique_results = set()
        retrieval_results = []
        for doc in results:
            instruction = doc.metadata["instruction"]
            output = doc.page_content
            unique_key = (instruction,output)
            
            if unique_key not in unique_results:
                
                retrieval_data ={
                    "instruction":instruction,
                    "output":output
                }
                
                retrieval_results.append(retrieval_data)
                unique_results.add(unique_key)
        # print(len(retrieval_results))
        
            
        return retrieval_results

    
    def combined_retrieval(self, query: str, top_k: int = 10, methods: List[str] = ['bm25', 'vector']) -> List[str]:
        
        # 结合 BM25 和向量检索结果，返回去重后的结果
        retrieved = []
        if 'bm25' in methods:
            bm25_results = self.bm25_retrieval(query, top_k)
            retrieved.extend(bm25_results)
        if 'vector' in methods:
            vector_results = self.vector_retrieval(query, top_k)
            retrieved.extend(vector_results)
        
        unique_results = set()
        combined_results = []
        print(f"retrieved:{retrieved}")
        for result in retrieved:
            # print(f"result:{result}")
            unique_key = (result["instruction"],result["output"])
            if unique_key not in unique_results: 
                combined_results.append(result)
                unique_results.add(unique_key)
        return combined_results
        
        
        
    
    
if __name__ == "__main__":
    from read_dataset import Reader
    dataset_path = '/root/autodl-fs/healthRAG/datatsets/sampled_medical_dialogue.json'
    reader = Reader(dataset_path)
    tasks = reader.get_tasks()
    print(tasks[0])
    retriever = HealthRetriever(
        emb_model_name_or_path="/root/autodl-fs/bge-m3",
        tasks=tasks,
        device='cuda',
        language='zh',
        faiss_index_path="/root/faiss_index_bge-m3-0523"
    )
    # retriever.save_faiss_index("faiss_index_bge-m3-0523")
    print(retriever.combined_retrieval("妊娠期糖尿病、糖尿病合并妊娠",methods=["bm25","vector"]))