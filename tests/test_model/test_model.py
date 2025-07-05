import pytest as pt
import torch

from services.startup import load_vocab
from model.model import inference_model

_vocab = load_vocab('artifacts/vocab.pkl')

class DummyModel:
    def __call__(self, text, text_length):
        logits = torch.tensor([[2.0, 0.5, 0.1]])
        attn = torch.rand((1, text.shape[1]))
        return logits, attn

model = DummyModel()

def test_inference_model_valid():
    text = 'lorem ipsum'
    inference_wrapper = inference_model(model, _vocab)
    preds, attn = inference_wrapper.predict(text)

    assert preds in inference_wrapper.class2idx.keys()
    assert isinstance(attn, torch.Tensor)
    assert attn.shape[0] == 1
    assert attn.shape[1] == len(text.split())

@pt.mark.parametrize('bad_text_input', ["", None, 123, "   "])
def test_inference_model_bad_text_input(bad_text_input):
    inference_wrapper = inference_model(model, _vocab)
    with pt.raises(TypeError):
        preds, attn = inference_wrapper.predict(bad_text_input)