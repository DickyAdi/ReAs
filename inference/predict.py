import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from model import model as model_module
from .utils import get_stopwords
from typing import Union

class batch_predictor:
    def __init__(self, model:model_module.inference_model, text_column:str):
        self.model = model
        self.df = None
        self.text_column = text_column
        self.len_valid_mask = 0
    def __preprocess(self):
        text_series = self.df[self.text_column].str.lower()
        empty_mask = text_series.apply(lambda x: isinstance(x, str) and x.strip() != "" and len(x.split()) > 1)
        return text_series, empty_mask
    def __predict(self):
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError(f"Value error: `df` is empty or not pandas DataFrame. Got {type(self.df)}")
        if not self.text_column:
            raise ValueError(f'Value error: `text_column` is empty')
        if self.text_column not in list(self.df.columns):
            raise KeyError(f'Key error: `text_column` not in index. Got {list(self.df.columns)}')
        text_series, valid_mask = self.__preprocess()
        self.__count_valid_mask(valid_mask)
        self.df.loc[valid_mask, 'prediction'] = text_series[valid_mask].apply(lambda x : self.model.predict(x)[0])
        self.df.loc[~valid_mask, 'prediction'] = None
        del text_series
    def __count_valid_mask(self, mask:pd.Series):
        self.len_valid_mask = mask[mask == True].count()
    def __read(self, file):
        try:
            self.df = pd.read_csv(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"file not found: {file}")
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty")
        except RuntimeError:
            raise RuntimeError(f'Expected file path. Got type: {type(file)}')
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
        return self.df, self.len_valid_mask

class topic_extractor:
    def __init__(self, text_column: str):
        self.valid_sentiment = set(['Positive', 'Negative', 'Neutral'])
        self.df = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=get_stopwords(), ngram_range=(2,2))
        self.word_matrix = None
        self.words = None
        self.text_column = text_column
    def __get_text_by_sentiment(self, sentiment:str) -> list:
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError(f'Value error: `df` is empty or is not pandas DataFrame. Got {type(self.df)}')
        sentiment = sentiment.capitalize()
        if sentiment not in self.valid_sentiment:
            raise ValueError(f"Invalid sentiment: `sentiment` must be one of. {', '.join(self.valid_sentiment)}")
        if self.text_column not in list(self.df.columns):
            raise ValueError(f"Invalid text column: `text_column` must be one of. {', '.join(list(self.df.columns))}")
        return self.df[self.df['prediction'] == sentiment][self.text_column].str.lower().tolist()
    def __extract(self, sentiment:str):
        text = self.__get_text_by_sentiment(sentiment)
        if text:
            self.word_matrix = self.tfidf_vectorizer.fit_transform(text)
            self.words = self.tfidf_vectorizer.get_feature_names_out()
            words_df = pd.DataFrame(self.word_matrix.toarray(), columns=self.words)
            topics = pd.DataFrame({'word' : words_df.columns.tolist(), 'score' : words_df.std(axis=0).fillna(0).values.tolist()})
            return topics
        else:
            return pd.DataFrame({'word' : [], 'score' : []})
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
