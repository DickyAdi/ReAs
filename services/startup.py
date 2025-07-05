from fastapi import FastAPI
import os

from model.utils import load_embedding_matrix, load_vocab
from model.model import biLSTM_Attention, inference_model

def load_model():
    # print(f'[DEBUG] CWD: {os.getcwd()}')
    _embedding_matrix = load_embedding_matrix('artifacts/embedding_matrix.npy')
    _vocab = load_vocab('artifacts/vocab.pkl')
    _model = biLSTM_Attention.load_from_checkpoint(checkpoint_path='artifacts/new_epoch=8-step=720.ckpt', embedding_matrix=_embedding_matrix, map_location='cpu')
    # app.state.model = inference_model(_model, _vocab)
    return inference_model(_model, _vocab)
