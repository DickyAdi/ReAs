import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gc

from model import model as model_module
from .utils import get_stopwords
from typing import Union
from loggers.log import get_loggers

predict_logger = get_loggers('reas.predict')
extractor_logger = get_loggers('reas.extractor')

class batch_predictor:
    def __init__(self, model:model_module.inference_model, text_column:str):
        self.model = model
        self.df = None
        self.text_column = text_column
        self.len_valid_mask = 0
        self.start_time = time.time()
        predict_logger.info('Batch predictor initiated.')
    def __preprocess(self):
        predict_logger.info('Preprocessing started with text column of %s.', self.text_column)
        text_series = self.df[self.text_column].str.lower()
        empty_mask = text_series.apply(lambda x: isinstance(x, str) and x.strip() != "" and len(x.split()) > 1)
        predict_logger.info('Preprocessing finished with %d valid rows.', empty_mask[empty_mask == True].count())
        return text_series, empty_mask
    def __predict(self):
        predict_logger.info('Prediction started with text column of %s.', self.text_column)
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            predict_logger.error('`df` is not valid DataFrame.')
            raise ValueError(f"Value error: `df` is empty or not pandas DataFrame. Got {type(self.df)}")
        if not self.text_column:
            predict_logger.error('`text_column` is empty.')
            raise ValueError(f'Value error: `text_column` is empty')
        if self.text_column not in list(self.df.columns):
            predict_logger.error('`text_column` not in DataFrame columns. Got %s', self.df.columns.tolist())
            raise KeyError(f'Key error: `text_column` not in index. Got {list(self.df.columns)}')
        text_series, valid_mask = self.__preprocess()
        self.__count_valid_mask(valid_mask)
        predict_logger.info('Running prediction on %d valid rows out of %d total rows.', self.len_valid_mask, len(text_series))
        self.df.loc[valid_mask, 'prediction'] = self.model.predict(text_series[valid_mask].tolist())
        self.df.loc[~valid_mask, 'prediction'] = None
        predict_logger.info('Prediction finished...')
        del text_series
    def __count_valid_mask(self, mask:pd.Series):
        self.len_valid_mask = mask[mask == True].count()
        predict_logger.debug('number of valid row: %d', self.len_valid_mask)
    def __read(self, file):
        predict_logger.info('Reading file %s', file)
        try:
            self.df = pd.read_csv(file)
            predict_logger.info('File loaded successfully. Got %d rows.', len(self.df))
        except FileNotFoundError:
            predict_logger.error('File not found: %s', file)
            raise FileNotFoundError(f"file not found: {file}")
        except pd.errors.EmptyDataError:
            predict_logger.error('File is empty: %s', file)
            raise ValueError("File is empty")
        except RuntimeError:
            predict_logger.error('Expected file path. Got type: %s', type(file))
            raise RuntimeError(f'Expected file path. Got type: {type(file)}')
        except Exception as e:
            predict_logger.error('Unexpected error while reading the file.')
            raise RuntimeError(f"Unexpected error: {e}")
    def fit_transform(self, file):
        predict_logger.info('Running fit_transform...')
        self.fit(file)
        return self.transform()
    def fit(self, file:Union[str, pd.DataFrame]):
        predict_logger.info('Running fit...')
        if isinstance(file, pd.DataFrame):
            self.df = file
            predict_logger.info('File is DataFrame. Initiated DataFrame with %d rows instead.', len(self.df))
        else:
            predict_logger.info('Reading file: %s', file)
            self.__read(file)
    def transform(self, df=None):
        predict_logger.info('Running transform...')
        if df is not None:
            self.df = df
        self.__predict()
        predict_logger.debug('Batch prediction of %d rows finished in %.4fs', len(self.df), round(time.time() - self.start_time, 4))
        return self.df, self.len_valid_mask

class topic_extractor:
    def __init__(self, text_column: str):
        self.valid_sentiment = set(['Positive', 'Negative', 'Neutral'])
        self.df = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=get_stopwords(), ngram_range=(2,2))
        self.text_column = text_column
        extractor_logger.info('topic_extractor initiated...')
    def __get_text_by_sentiment(self, sentiment:str) -> list:
        extractor_logger.info('Getting text by %s sentiment', sentiment)
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            extractor_logger.error('`df` is invalid. Got %s', type(self.df))
            raise ValueError(f'Value error: `df` is empty or is not pandas DataFrame. Got {type(self.df)}')
        sentiment = sentiment.capitalize()
        if sentiment not in self.valid_sentiment:
            extractor_logger.error('`sentiment` is invalid. Got %s', sentiment)
            raise ValueError(f"Invalid sentiment: `sentiment` must be one of. {', '.join(self.valid_sentiment)}")
        if self.text_column not in list(self.df.columns):
            extractor_logger.error('`text_column` is not in DataFrame column. Got %s from %s', self.text_column, self.df.columns.tolist())
            raise ValueError(f"Invalid text column: `text_column` must be one of. {', '.join(list(self.df.columns))}")
        if 'prediction' not in list(self.df.columns):
            extractor_logger.error('`prediction` Column not found in DataFrame.')
            raise ValueError('`prediction` Column not found in DataFrame.')
        extractor_logger.info('Got %d rows', len(self.df[self.df['prediction'] == sentiment]))
        return self.df[self.df['prediction'] == sentiment][self.text_column].str.lower().tolist()
    def __extract(self, sentiment:str):
        extractor_logger.info('Extracting %s topics...', sentiment)
        text = self.__get_text_by_sentiment(sentiment)

        if not text:
            extractor_logger.warning('Got empty %s sentiment in DataFrame, skipping.', sentiment)
            return pd.DataFrame(columns=['word', 'score']), pd.DataFrame(columns=['word', 'score'])
        if len(text) < 2:
            extractor_logger.warning('Not enough row of text for %s sentiment, skipping.', sentiment)
            return pd.DataFrame(columns=['word', 'score']), pd.DataFrame(columns=['word', 'score'])
        docs = [doc for doc in text if len(doc.split()) >= 2]
        if not docs:
            extractor_logger.warning('No valid multi-word string for %s sentiment, skipping.', sentiment)
            return pd.DataFrame(columns=['word', 'score']), pd.DataFrame(columns=['word', 'score'])
        extractor_logger.info('Starting extraction...')
        word_matrix = self.tfidf_vectorizer.fit_transform(docs)
        words = self.tfidf_vectorizer.get_feature_names_out()
        # start code memory saving
        matrix_mean = word_matrix.mean(axis=0).A1
        matrix_sq_mean = word_matrix.power(2).mean(axis=0).A1
        std = np.sqrt(matrix_sq_mean - matrix_mean**2)
        
        # stop code memory saving
        # words_df = pd.DataFrame(word_matrix.toarray(), columns=words)
        # topics = pd.DataFrame({'word' : words_df.columns.tolist(), 'score' : words_df.std(axis=0).fillna(0).values.tolist()})

        # addapting topics to memory saving code
        topics_std = pd.DataFrame({'word' : words, 'score' : std})
        topics_mean = pd.DataFrame({'word' : words, 'score' : matrix_mean})
        del word_matrix, std, matrix_mean, matrix_sq_mean
        gc.collect()
        extractor_logger.info('Extraction completed with %d rows topics.', len(topics_std))
        return topics_std, topics_mean
    def fit(self, df):
        extractor_logger.info('Starting fit...')
        self.df = df
    def transform(self, sentiment:str, df:pd.DataFrame=None) -> pd.DataFrame:
        extractor_logger.info('Starting transform...')
        if df:
            self.df = df
            extractor_logger.info('`df` is provided, initializing.')
        topics_std, topics_mean = self.__extract(sentiment)
        return topics_std, topics_mean
    def fit_transform(self, df, sentiment):
        extractor_logger.info('Starting fit_transform...')
        self.fit(df)
        return self.transform(sentiment=sentiment)
