import re
import torch
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

from domain.ml.interfaces.interface import inferenceInterface
from .models.model import biLSTM_Attention
from config.settings import settings


def load_vocab(file):
    with open(file, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def load_embedding_matrix(file):
    embedding_matrix = np.load(file, allow_pickle=False)
    return embedding_matrix


class inferenceModel(inferenceInterface):
    def __init__(self):
        _embedding_matrix = load_embedding_matrix("artifacts/embedding_matrix.npy")
        self.vocab = load_vocab("artifacts/vocab.pkl")
        self.model = biLSTM_Attention.load_from_checkpoint(
            checkpoint_path="artifacts/new_epoch=8-step=720.ckpt",
            embedding_matrix=_embedding_matrix,
            map_location="cpu",
        )
        self.chunk_size = settings.predict_chunk_size
        self.class2idx = {"Positive": 0, "Negative": 1, "Neutral": 2}
        self.idx2class = {v: k for k, v in self.class2idx.items()}

    def _remove_emoticons(self, text: str):
        """Don't override unless you know what you are doing!
        Remove all emojis from input text.

        Args:
            text (str): Input text.

        Returns:
            str: Input text without emojis.
        """
        return re.sub(r"[^\w\s,.]", "", text)

    def _clean_text(self, sentences: str | list[str]):
        """Don't override unless you know what you are doing!
        Cleaning, filterring, lowercasing, and tokenizing input texts.

        Args:
            sentences (str | list[str]): Input text if string, else input texts

        Raises:
            ValueError: If `sentence` not str or list[str].

        Returns:
            list[list[str]]: Cleaned tokenized input texts for mapping.
        """
        texts = []
        for text in sentences:
            if isinstance(text, str) and text.strip() != "":
                cleaned = self._remove_emoticons(text)
                tokenized_texts = word_tokenize(cleaned)
                texts.append(tokenized_texts)
            else:
                raise ValueError("Found invalid string after filtering.")
        return texts

    def _predict_prepare_data(self, sentence: list[str]):
        """Don't override unless you know what you are doing!
        Data prep before prediction, including tensor converting, padding, cleaning, and tokenizing.

        Args:
            sentence (list[str]): Input text.

        Returns:
            tuple(tensor, tensor): A tuple of text and text length tensors.
        """
        tokenized_texts = self._clean_text(sentence)
        indexed = [self._sentence2idx(t) for t in tokenized_texts]
        lengths = torch.tensor([len(x) for x in indexed])
        max_length = max(lengths) if lengths.numel() > 0 else 0
        padded = [seq + [0] * (max_length - len(seq)) for seq in indexed]
        text_tensor = torch.LongTensor(padded)
        return text_tensor, lengths

    def _sentence2idx(self, sentence: list[list[str]]):
        """Don't override unless you know what you are doing!
        Mapping each tokenized word from a sentence to a vocab index.

        Args:
            sentence (list[list[str]]): List of tokenized sentence.

        Returns:
            list[int]: List of tokens index.
        """
        sentenceidx = []
        for word in sentence:
            if re.fullmatch(r"[a-zA-Z0-9.,!?;:'\"()\[\]\-{}@#$%&+/=_\\|^~`]+", word):
                word = word.lower()
            sentenceidx.append(self.vocab.get(word, 0))
        return sentenceidx

    def _format_output(self, texts: list[str], sentiments: list[str]) -> list[dict]:
        return [
            {"text": text, "sentiment": sentiment}
            for text, sentiment in zip(texts, sentiments)
        ]

    def predict(self, texts: str | list[str]) -> dict | list[dict]:
        """A prediction step for the model to take. This includes whether single or batch predictions.
        Override this method according to the used model prediction behavior.

        Args:
            texts (str | list[str]): Input texts.

        Raises:
            TypeError: If texts is not str or list[str].

        Returns:
            dict | list[dict]: Will return str of the prediction if single text provided, else will return predicted list[str].
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        else:
            if not isinstance(texts, list):
                raise TypeError(f"Expected str or list[str]. Got {type(texts)}")
        pred_results = []
        total = len(texts)
        total_chunk = (total + self.chunk_size - 1) // self.chunk_size  # noqa
        for i in range(0, total, self.chunk_size):
            chunk = texts[i : i + self.chunk_size]
            prep_text, prep_len_text = self._predict_prepare_data(chunk)
            with torch.no_grad():
                pred, context_weights = self.model(prep_text, prep_len_text)
            indices = torch.argmax(pred, 1).numpy()
            labels = [self.idx2class[int(x)] for x in indices]
            res = self._format_output(chunk, labels)
            pred_results.extend(res)
        return pred_results[0] if is_single else pred_results
