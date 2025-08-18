import pandas as pd
import gc
from typing import TypedDict

from .predict import batch_predictor, topic_extractor

class ExtractionPipelineDictType(TypedDict):
    n_positive:int
    n_negative:int
    positive_trend_topics:pd.DataFrame
    positive_frequent_topics:pd.DataFrame
    negative_trend_topics:pd.DataFrame
    negative_frequent_topics:pd.DataFrame
    len_valid_mask:int

def extraction_pipeline(model, df, text_column) -> ExtractionPipelineDictType:
    predictor = batch_predictor(model, text_column)
    predicted_df, len_valid_mask = predictor.fit_transform(df)
    extractor = topic_extractor(text_column)
    extractor.fit(predicted_df)
    pos_topics_std, pos_topics_mean = extractor.transform('Positive')
    neg_topics_std, neg_topics_mean = extractor.transform('Negative')
    n_positive = extractor.count_sentiment('Positive')
    n_negative = extractor.count_sentiment('Negative')
    # res = (pos_topics_std, pos_topics_mean, neg_topics_std, neg_topics_mean, len_valid_mask)
    res = {
        'positive_trend_topics': pos_topics_std,
        'positive_frequent_topics': pos_topics_mean,
        'negative_trend_topics': neg_topics_std,
        'negative_frequent_topics': neg_topics_mean,
        'len_valid_mask': len_valid_mask,
        'n_positive': n_positive,
        'n_negative': n_negative
    }

    del predictor, predicted_df, extractor, pos_topics_std, pos_topics_mean, neg_topics_std, neg_topics_mean, len_valid_mask
    gc.collect()
    return res