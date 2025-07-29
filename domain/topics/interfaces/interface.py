import pandas as pd
from abc import ABC, abstractmethod

class extractTopicsInterface(ABC):
    def __init__(self, df:pd.DataFrame, vectorizer):
        self.vectorizer = vectorizer
        self.df = df

    @abstractmethod
    def extract(self, text:list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
    
    @abstractmethod
    def get_text_by_sentiment(self, sentiment:str) -> list[str]:
        pass