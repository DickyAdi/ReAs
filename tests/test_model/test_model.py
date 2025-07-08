import pytest as pt
import torch

from services.startup import load_vocab
from model.model import inference_model

_vocab = load_vocab('artifacts/vocab.pkl')

class DummyModel:
    def __call__(self, text, text_length):
        batch_size = text.shape[0]
        seq_len = text.shape[1]
        logits = torch.tensor([[2.0, 0.5, 0.1]] * batch_size)
        attn = torch.rand((batch_size, seq_len))
        return logits, attn

model = DummyModel()

def test_inference_model_single_text():
    text = 'lorem ipsum'
    inference_wrapper = inference_model(model, _vocab)
    preds = inference_wrapper.predict(text)

    assert preds in inference_wrapper.class2idx.keys()
    assert preds == 'Positive'

def test_inference_model_multiple_texts():
    text = ['lorem ipsum', 'dolor sit amet', 'yada yada yada']
    inference_wrapper = inference_model(model, _vocab)
    preds = inference_wrapper.predict(text)

    assert set(preds).issubset(set(inference_wrapper.class2idx.keys()))
    assert len(preds) == 3

@pt.mark.parametrize('bad_text_input', ["", None, 123, "   "])
def test_inference_model_bad_text_input(bad_text_input):
    inference_wrapper = inference_model(model, _vocab)
    with pt.raises((TypeError, ValueError)):
        preds = inference_wrapper.predict(bad_text_input)