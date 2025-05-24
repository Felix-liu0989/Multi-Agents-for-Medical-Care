from abc import ABC, abstractmethod
from models.Base import BaseModel

class BaseStrategy(ABC):
    def __init__(self, 
                 model: BaseModel, 
                 retriever=None, 
                 reranker=None, 
                 verbose=True,
                 **kwargs):
        self.model = model
        self.retriever = retriever
        self.reranker = reranker
        self.verbose = verbose

    @abstractmethod
    def generate(self, query, **kwargs):
        """
        子类必须实现：根据 query 生成答案
        """
        pass