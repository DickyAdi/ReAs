import pytest as pt
import pandas as pd

from inference.predict import batch_predictor, topic_extractor

class DummyModel:
    def predict(self, text):
        return ["Positive"] * len(text)

df = pd.DataFrame({
    'text' : ['Kopinya enak', 'bla bla', 'another bla bla']
})
model = DummyModel()

def test_batch_predictor_with_valid_dataframe():
    predictor = batch_predictor(model=model, text_column='text')
    res_df, len_valid_mask = predictor.fit_transform(df)

    assert 'prediction' in list(res_df.columns)
    assert res_df['prediction'].unique().tolist() == ['Positive']
    assert len(res_df['prediction']) == 3

def test_batch_predictor_with_none_dataframe():
    df = None
    with pt.raises(RuntimeError):
        predictor = batch_predictor(model=model, text_column='text')
        res_df, len_valid_mask = predictor.fit_transform(df)

def test_batch_predictor_with_wrong_text_column():
    text_cols = 'wrong'
    with pt.raises(KeyError):
        predictor = batch_predictor(model=model, text_column=text_cols)
        res_df, len_valid_mask = predictor.fit_transform(df)

def test_batch_predictor_with_invalid_dataframe():
    df = ['wrong1', ['wrong2']]
    with pt.raises(RuntimeError):
        predictor = batch_predictor(model=model, text_column='text')
        res_df, len_valid_mask = predictor.fit_transform(df)

def test_batch_predictor_with_empty_row_dataframe():
    df = pd.DataFrame(columns=['text'])

    predictor = batch_predictor(model=model, text_column='text')
    res_df, len_valid_mask = predictor.fit_transform(df)

    assert 'prediction' in list(res_df.columns)
    assert res_df['prediction'].unique().tolist() == []
    assert len(res_df['prediction']) == 0

df_topic_valid = pd.DataFrame({
    'prediction' : ['Positive', 'Positive', 'Negative', 'Negative'],
    'text' : ['Dem yada', 'Yada wutt', 'Nice coffee', 'Lorem Ipsum']
})

df_topic_empty = pd.DataFrame(columns=['prediction', 'text'])


def test_topic_extractor_with_valid_dataframe():
    extractor = topic_extractor('text')
    topics = extractor.fit_transform(df_topic_valid, 'Positive')

    assert set(['word', 'score']).issubset(set(topics.columns.tolist()))
    assert not topics.empty
    assert len(topics) == 2

@pt.mark.parametrize('df, sentiment, text_column, expected', [
    (df_topic_empty, 'Positive', 'text', pd.DataFrame(columns=['word', 'score'])),
    (None, 'Positive', 'text', pt.raises(ValueError)), # df is none
    (df_topic_valid, 'dummy', 'text', pt.raises(ValueError)), #Wrong sentiment
    (df_topic_valid, 'Positive', 'dummy', pt.raises(ValueError)), #Wrong text column
])
def test_topic_extractor_with_bad_input(df, sentiment, text_column, expected):
    if isinstance(expected, pd.DataFrame):
        extractor = topic_extractor(text_column)
        topics = extractor.fit_transform(df, sentiment)
        assert isinstance(topics, pd.DataFrame)
        assert topics.empty
        assert len(topics) == 0
        assert set(['word', 'score']).issubset(set(topics.columns.tolist()))
    else:
        with expected:
            extractor = topic_extractor(text_column)
            topics = extractor.fit_transform(df, sentiment)