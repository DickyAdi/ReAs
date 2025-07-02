from inference import predict
from model.model import biLSTM_Attention, inference_model
from model.utils import load_vocab, load_embedding_matrix

_embedding_matrix = load_embedding_matrix('artifacts/embedding_matrix.npy')
_vocab = load_vocab('artifacts/vocab.pkl')
_model = biLSTM_Attention.load_from_checkpoint(checkpoint_path='artifacts/new_epoch=8-step=720.ckpt', embedding_matrix=_embedding_matrix, vocab=_vocab, map_location='cpu')

predictor = predict.batch_predictor(_model)
predicted_df = predictor.fit_transform('domain_data.csv')
extractor = predict.topic_extractor()
extracted_topics = extractor.fit_transform(predicted_df, sentiment='Positive')

print(extracted_topics.head(10))