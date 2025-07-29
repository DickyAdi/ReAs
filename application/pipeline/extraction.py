import pandas as pd
from typing import TypedDict
import gc

from application.predict_service import PredictService
from application.extract_service import ExtractService
from core.prediction_stats import count_sentiments

class ExtractionPipelineDictType(TypedDict):
    n_positive:int
    n_negative:int
    positive_trend_topics:pd.DataFrame
    positive_frequent_topics:pd.DataFrame
    negative_trend_topics:pd.DataFrame
    negative_frequent_topics:pd.DataFrame
    len_valid_mask:int

class ExtractionPipelineFlow:
    def __init__(self, predictor:PredictService, extractor:ExtractService):
        self.predictor = predictor
        self.extractor = extractor
    def run(self, df:pd.DataFrame, text_column:str) -> ExtractionPipelineDictType:
        predicted_df, len_valid_mask = self.predictor.run(df=df, text_column=text_column)
        positive_trend, positive_frequent = self.extractor.run(df=df, sentiment='Positive')
        negative_trend, negative_frequent = self.extractor.run(df=df, sentiment='Negative')
        if 'prediction' not in predicted_df.columns.tolist():
            raise KeyError(f'`prediction` column does not exists.')
        count = count_sentiments(predicted_df['prediction'])
        res = {
            'positive_trend_topics' : positive_trend,
            'positive_frequent_topics' : positive_frequent,
            'negative_trend_topics' : negative_trend,
            'negative_frequent_topics' : negative_frequent,
            'len_valid_mask' : len_valid_mask,
            'n_positive' : count['Positive'],
            'n_negative' : count['Negative']
        }
        del self.predictor, predicted_df, self.extractor, len_valid_mask, positive_trend, positive_frequent, negative_trend, negative_frequent, count
        gc.collect()
        return res