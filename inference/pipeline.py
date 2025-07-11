import pandas as pd
import gc
from typing import TypedDict

from .predict import batch_predictor, topic_extractor

class ExtractionPipelineDictType(TypedDict):
    positive_topics_std:pd.DataFrame
    positive_topics_mean:pd.DataFrame
    negative_topics_std:pd.DataFrame
    negative_topics_mean:pd.DataFrame
    len_valid_mask:int

def extraction_pipeline(model, df, text_column) -> ExtractionPipelineDictType:
    predictor = batch_predictor(model, text_column)
    predicted_df, len_valid_mask = predictor.fit_transform(df)
    extractor = topic_extractor(text_column)
    extractor.fit(predicted_df)
    pos_topics_std, pos_topics_mean = extractor.transform('Positive')
    neg_topics_std, neg_topics_mean = extractor.transform('Negative')
    res = (pos_topics_std, pos_topics_mean, neg_topics_std, neg_topics_mean, len_valid_mask)
    res = {
        'positive_topics_std': pos_topics_std,
        'positive_topics_mean': pos_topics_mean,
        'negative_topics_std': neg_topics_std,
        'negative_topics_mean': neg_topics_mean,
        'len_valid_mask': len_valid_mask,
    }

    del predictor, predicted_df, extractor, pos_topics_std, pos_topics_mean, neg_topics_std, neg_topics_mean, len_valid_mask
    gc.collect()
    return res