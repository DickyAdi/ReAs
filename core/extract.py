import pandas as pd

from domain.topics.interfaces.interface import extractTopicsInterface

class ExtractLogic:
    def __init__(self, extractor:extractTopicsInterface):
        self.extractor = extractor
    def extract(self, sentiment:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if sentiment not in ['Positive', 'Negative']:
            raise ValueError(f'Sentiment must be `Positive` or `Negative`. Got {sentiment}')
        text = self.extractor.get_text_by_sentiment(sentiment)
        trend_topics, frequent_topics = self.extractor.extract(text)
        return trend_topics, frequent_topics