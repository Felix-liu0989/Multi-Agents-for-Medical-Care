def main():
    args = arg_parse()

    reader = Reader(args.corpus_path)
    corpus = reader.corpus

    retriever = HealthRetriever(emb_model_name_or_path=args.emb_model_name_or_path, corpus=corpus)
    reranker = Reranker(rerank_model_name_or_path=args.rerank_model_name_or_path)
    llm = LLMPredictor(llm_model_name_or_path=args.llm_model_name_or_path)

    result_list = []
    with open(args.test_query_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    for i, line in enumerate(result):
        query = line['question']
        retrieval_res = retriever.retrieval(query)
        rerank_res = reranker.rerank(retrieval_res, query, k=args.num_input_docs)
        res = llm.predict('\n'.join(rerank_res), query).strip()
        line['answer'] = res
        del line['answer_1'], line['answer_2'], line['answer_3']
        result_list.append(line)
        print('question {}: '.format(i))
        print(line)
        print()

    res_file_path = 'res.json'
    with open(res_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()