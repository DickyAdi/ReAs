import pickle
import numpy as np

def load_vocab(file):
    with open(file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def load_embedding_matrix(file):
    embedding_matrix = np.load(file, allow_pickle=True)
    return embedding_matrix