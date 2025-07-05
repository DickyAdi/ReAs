import pytest as pt
import pandas as pd

from inference.predict import batch_predictor, topic_extractor

class DummyModel:
    def predict(self, text):
        return ["Positive", "Dummy attn weights"]

df = pd.DataFrame({
    'text' : ['Kopinya enak', 'bla bla', 'another bla bla']
})
model = DummyModel()

def test_batch_predictor_with_valid_dataframe():
    predictor = batch_predictor(model=model, text_column='text')
    res = predictor.fit_transform(df)

    assert 'prediction' in list(res.columns)
    assert res['prediction'].unique().tolist() == ['Positive']
    assert len(res['prediction']) == 3

def test_batch_predictor_with_none_dataframe():
    df = None
    with pt.raises(RuntimeError):
        predictor = batch_predictor(model=model, text_column='text')
        res = predictor.fit_transform(df)

def test_batch_predictor_with_wrong_text_column():
    text_cols = 'wrong'
    with pt.raises(KeyError):
        predictor = batch_predictor(model=model, text_column=text_cols)
        res = predictor.fit_transform(df)

def test_batch_predictor_with_invalid_dataframe():
    df = ['wrong1', ['wrong2']]
    with pt.raises(RuntimeError):
        predictor = batch_predictor(model=model, text_column='text')
        res = predictor.fit_transform(df)

def test_batch_predictor_with_empty_row_dataframe():
    df = pd.DataFrame(columns=['text'])

    predictor = batch_predictor(model=model, text_column='text')
    res = predictor.fit_transform(df)

    assert 'prediction' in list(res.columns)
    assert res['prediction'].unique().tolist() == []
    assert len(res['prediction']) == 0