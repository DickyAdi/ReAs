import pandas as pd
import gc

from .predict import batch_predictor, topic_extractor

def extraction_pipeline(model, df, text_column):
    predictor = batch_predictor(model, text_column)
    predicted_df, len_valid_mask = predictor.fit_transform(df)
    extractor = topic_extractor(text_column)
    extractor.fit(predicted_df)
    pos_topics = extractor.transform('Positive')
    neg_topics = extractor.transform('Negative')
    res = (pos_topics, neg_topics, len_valid_mask)

    del predictor, predicted_df, extractor, pos_topics, neg_topics, len_valid_mask
    gc.collect()
    return res