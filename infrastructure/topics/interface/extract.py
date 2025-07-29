from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import gc

from domain.topics.interfaces.interface import extractTopicsInterface
from ..utils import get_stopwords

class ExtractTopics(extractTopicsInterface):
    def __init__(self, df, text_column:str):
        super().__init__(df, vectorizer=TfidfVectorizer(stop_words=get_stopwords()))
        self.scaler = MinMaxScaler()
        self.text_column = text_column
        self.valid_sentiment = set(['Positive', 'Negative', 'Neutral'])
    def get_text_by_sentiment(self, sentiment):
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError(f'Value error: `df` is empty or is not pandas DataFrame. Got {type(self.df)}.')
        sentiment = sentiment.capitalize()
        if not sentiment and sentiment not in self.valid_sentiment:
            raise ValueError(f"Invalid sentiment: `sentiment` must be one of. {', '.join(self.valid_sentiment)}")
        if self.text_column not in self.df.columns.tolist():
            raise ValueError(f"Invalid text column: `text_column` must be one of. {', '.join(list(self.df.columns))}")
        if 'prediction' not in self.df.columns.tolist():
            raise ValueError('`prediction` Column not found in DataFrame.')
        return self.df[self.df['prediction'] == sentiment][self.text_column].str.lower().tolist()
    def extract(self, text):
        if not text:
            return pd.DataFrame(columns=['word', 'score']), pd.DataFrame(columns=['word', 'score'])
        if len(text) < 2:
            return pd.DataFrame(columns=['word', 'score']), pd.DataFrame(columns=['word', 'score'])
        docs = [doc for doc in text if len(doc.split()) > 2]
        if not docs:
            return pd.DataFrame(columns=['word', 'score']), pd.DataFrame(columns=['word', 'score'])
        word_matrix = self.vectorizer.fit_transform(docs)
        words = self.vectorizer.get_feature_names_out()
        # memory first code
        count_matrix = (word_matrix > 0).sum(axis=0).A1
        matrix_mean = word_matrix.mean(axis=0).A1
        matrix_sq_mean = word_matrix.power(2).mean(axis=0).A1
        std = np.sqrt(matrix_sq_mean - matrix_mean**2)
        score = matrix_mean * std
        score = self.scaler.fit_transform(score.reshape(-1,1)).flatten()
        score_trend = score * (count_matrix / len(self.df))
        score_trend = self.scaler.fit_transform(score_trend.reshape(-1,1)).flatten()
        # stop memory first code
        trend_topics = pd.DataFrame({'word' : words, 'score' : score_trend})
        frequent_topics = pd.DataFrame({'word' : words, 'score' : score})
        del word_matrix, std, matrix_mean, matrix_sq_mean
        gc.collect()
        return trend_topics, frequent_topics
