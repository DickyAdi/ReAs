from fastapi import FastAPI
import os

from model.utils import load_embedding_matrix, load_vocab
from model.model import biLSTM_Attention, inference_model
from loggers.log import get_loggers

model_loggers = get_loggers('reas.load_model')

def load_model():
    # print(f'[DEBUG] CWD: {os.getcwd()}')
    model_loggers.info('Loading embedding matrix...')
    _embedding_matrix = load_embedding_matrix('artifacts/embedding_matrix.npy')
    model_loggers.info('Loading vocab...')
    _vocab = load_vocab('artifacts/vocab.pkl')
    model_loggers.info('Loading weights into the model...')
    _model = biLSTM_Attention.load_from_checkpoint(checkpoint_path='artifacts/new_epoch=8-step=720.ckpt', embedding_matrix=_embedding_matrix, map_location='cpu')
    # app.state.model = inference_model(_model, _vocab)
    model_loggers.info('Model loaded successfully...')
    return inference_model(_model, _vocab)
