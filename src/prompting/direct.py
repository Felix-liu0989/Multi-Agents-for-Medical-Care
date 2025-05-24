from .Base import BaseStrategy

class DirectStrategy(BaseStrategy):
    def __init__(self, model, verbose=True):
        super().__init__(model, verbose=verbose)

    def generate(self, query, **kwargs):
        """
        直接用大模型生成通俗解释、用药建议等内容
        """
        # 你可以根据需要拼接 prompt
        return self.model.prompt(query)