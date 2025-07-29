import pandas as pd

from core.extract import ExtractLogic
from domain.topics.interfaces.interface import extractTopicsInterface

class ExtractService:
    def __init__(self, extractor:extractTopicsInterface):
        self.extractor = extractor
        self.logic = ExtractLogic(extractor)
    def run(self, df, sentiment) -> tuple[pd.DataFrame, pd.DataFrame]:
        trend_df, frequent_df = self.logic.extract(sentiment)
        return trend_df, frequent_df