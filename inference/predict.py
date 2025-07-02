import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from model import model as model_module
from .utils import get_stopwords
from typing import Union

class batch_predictor:
    def __init__(self, model:model_module.inference_model):
        self.model = model
        self.df = None
    def __preprocess(self, df:pd.DataFrame):
        df['text'] = df['text'].str.lower()
        return df
    def __predict(self):
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError(f"Value error: `df` is empty or not pandas DataFrame. Got {type(self.df)}")
        preprocessed_df = self.__preprocess(self.df)
        self.df['prediction'] = preprocessed_df['text'].apply(lambda x: self.model.predict(x)[0])
        del preprocessed_df
    def __read(self, file):
        try:
            self.df = pd.read_csv(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"file not found: {file}")
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")
    def fit_transform(self, file):
        self.fit(file)
        return self.transform()
    def fit(self, file:Union[str, pd.DataFrame]):
        if isinstance(file, pd.DataFrame):
            self.df = file
        else:
            self.__read(file)
    def transform(self, df=None):
        if df is not None:
            self.df = df
        self.__predict()
        return self.df

class topic_extractor:
    def __init__(self):
        self.valid_sentiment = set(['Positive', 'Negative', 'Neutral'])
        self.df = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=get_stopwords(), ngram_range=(2,2))
        self.word_matrix = None
        self.words = None
    def __get_text_by_sentiment(self, sentiment:str) -> list:
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError(f'Value error: `df` is empty or is not pandas DataFrame. Got {type(self.df)}')
        sentiment = sentiment.capitalize()
        if sentiment not in self.valid_sentiment:
            raise ValueError(f"Invalid sentiment: `sentiment` must be one of. {', '.join(self.valid_sentiment)}")
        return self.df[self.df['prediction'] == sentiment]['text'].str.lower().tolist()
    def __extract(self, sentiment:str):
        text = self.__get_text_by_sentiment(sentiment)
        self.word_matrix = self.tfidf_vectorizer.fit_transform(text)
        self.words = self.tfidf_vectorizer.get_feature_names_out()
        words_df = pd.DataFrame(self.word_matrix.toarray(), columns=self.words)
        topics = pd.DataFrame({'word' : words_df.columns.tolist(), 'score' : words_df.std(axis=0).values})
        return topics
    def fit(self, df):
        self.df = df
    def transform(self, sentiment:str, df:pd.DataFrame=None) -> pd.DataFrame:
        if df:
            self.df = df
        topics = self.__extract(sentiment)
        return topics
    def fit_transform(self, df, sentiment):
        self.fit(df)
        return self.transform(sentiment=sentiment)
