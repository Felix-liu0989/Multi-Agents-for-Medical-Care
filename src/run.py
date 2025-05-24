import json
from argparse import ArgumentParser
from llm_infer import LLMPredictor
import os,sys
from retriever import HealthRetriever
from reranker import Reranker
from read_dataset import Reader
from tqdm import tqdm

def arg_parse():

    parser = ArgumentParser()
    parser.add_argument('--llm_model_name', default='deepseek-chat', help='the name of llm to use')
    parser.add_argument('--emb_model_name_or_path', default='/root/autodl-fs/bge-m3', help='the path of embedding model to use')
    parser.add_argument('--rerank_model_name_or_path', default='/root/autodl-fs/bge-reranker-v2-m3', help='the path of rerank model to use')
    parser.add_argument('--retrieval_methods', default='emb', help='the method to retrieval, you can choose from [bm25, emb]')
    parser.add_argument('--corpus_path', default="/root/autodl-fs/healthRAG/datatsets/sampled_medical_dialogue.json", help='corpus path')
    parser.add_argument('--query_path', default='/root/autodl-fs/healthRAG/datatsets/full_cases.json', help='query path')
    parser.add_argument('--faiss_index_path', default='/root/faiss_index_bge-m3-0523', help='faiss index path')
    return parser.parse_args()

def main():
    args = arg_parse()
    # 1. 读取病例数据集
    test_set_path = args.query_path
    if not os.path.exists(test_set_path):
        print(f"测试集文件 '{test_set_path}' 未找到。请确保文件存在。")
        sys.exit(1)
    
    
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    
    
    # 2. 初始化 Reader 并加载任务数据
    reader = Reader(args.corpus_path)
    task = reader.get_tasks()
    
    # 3. 初始化 Retriever
    retriever = HealthRetriever(
        emb_model_name_or_path=args.emb_model_name_or_path, 
        tasks=task,
        language='zh',
        faiss_index_path=args.faiss_index_path
    )
    # 4. 初始化 Reranker
    reranker = Reranker(
        rerank_model_name_or_path=args.rerank_model_name_or_path,
        device='cuda'
    )
    # 5. 初始化 LLMPredictor
    llm = LLMPredictor(
        model_name=args.llm_model_name
    )
    
    
    
    
    
    test_set = test_set[:]
    # 6. 定义输出路径
    output_path = '/root/autodl-fs/healthRAG/outputs/doctor_advice_test.jsonl'
    # '../outputs/doctor_advice.jsonl'
    
    # 7. 读取已经处理过的测试用例
    processed_indices = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                try:
                    test_case = json.loads(line)
                    processed_indices.add(test_case['id'])
                except json.JSONDecodeError:
                    continue
    
    with open(output_path, 'a', encoding='utf-8') as outfile:
        # 9. 批量处理测试用例
        for test_case in tqdm(test_set, desc="批量处理测试用例"):
            id = test_case.get("id", "")
            if id in processed_indices:
                continue  # 跳过已经处理过的测试用例
            
            print(f"{id} 开始处理------------",flush=True)
            
            query = test_case.get("query_intent", "")
            print(f"query:{query}")
            print(f'{}')
            try:
                with tqdm(total=3, desc=f"处理病例 {id}", unit="步", leave=False) as pbar:
                    # 1. 检索相关文档
                    retrieved_docs = retriever.combined_retrieval(
                        query, 
                        top_k=10,
                        methods=args.retrieval_methods.split(",")
                    )
                    pbar.update(1) 
                    print(f"retrieved_docs:{retrieved_docs}")
                    # 2. 重排检索结果
                    reranked_docs = reranker.rerank(retrieved_docs, query, k=5)
                    print(f"reranked_docs:{reranked_docs}")
                    pbar.update(1)  # 更新进度条
                    
                    # 3. 生成医嘱
                    # 根据相似度值进行再筛选
                    new_reranked_docs = []
                    for rerank_doc in reranked_docs:
                        if rerank_doc["score"] >= 1:
                            doc = {
                                "instruction": rerank_doc["instruction"],
                                "output": rerank_doc["output"],
                                "score": rerank_doc["score"]
                            }
                            new_reranked_docs.append(doc)
                            
                    if len(new_reranked_docs) > 0:
                        generated_advice = llm.with_retrieval_predict(new_reranked_docs,query)
                    else:
                        generated_advice = llm.simple_predict(query)
                            
                    print(f"new_reranked_docs:{new_reranked_docs}")
                    print(f'generated_advice:{generated_advice}')
                    pbar.update(1)  # 更新进度条
                            
                # 10. 保存生成的医嘱和 external documents
                generated_advice = generated_advice.replace("```json", "").replace("```", "")
                test_case['final_completion'] = generated_advice
                test_case['external_documents'] = reranked_docs
                test_case['final_external_documents'] = new_reranked_docs  # 将 external documents 保存到 test_case 中 
            
            except Exception as e:
                print(f"处理病例 {id} 时发生错误：{e}")
                test_case['completion'] = f"生成医嘱时发生错误：{str(e)}"
                test_case['external_documents'] = "" 
                test_case['final_external_documents'] = ""
            # 写入结果到 JSONL 文件
            outfile.write(json.dumps(test_case, ensure_ascii=False) + '\n')

    print(f"所有测试用例的生成结果已保存到 '{output_path}'。")

if __name__ == "__main__":
    main()
    
