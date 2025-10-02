import numpy as np
from nltk.corpus import stopwords
import re
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from domain.extractor import ExtractorInterface
from domain.exceptions import NotEnoughReviewsError


class ExtractionService(ExtractorInterface):
    def _get_stopwords(self, language: str = "indonesian") -> list[set]:
        if language != "indonesian":
            raise NotImplementedError(
                f"Only supports bahasa indonesia as the stopwords, expected `language` to be `indonesian`. Got type {type(language)} with value of {language}."
            )
        ind_stop = set(stopwords.words(language))
        if language == "indonesian":
            added_stop = {
                "yg",
                "nya",
                "deh",
                "ok",
                "jg",
                "aja",
                "sih",
                "kali",
                "banget",
                "bgt",
                "nih",
                "ya",
                "dg",
                "tp",
                "utk",
                "untuk",
                "kalok",
                "kalau",
                "kalo",
                "klo",
                "kl",
                "gak",
                "gk",
                "udah",
                "ga",
                "sip",
                "jember",
                "situbondo",
            }
            ind_stop |= added_stop

        preprocessed = set()
        for word in ind_stop:
            cleaned = re.sub(r"[^a-zA-Z]", "", word.lower())
            if cleaned:
                preprocessed.add(cleaned)
        return list(preprocessed)

    def extract(
        self,
        review_data: list[tuple],
        top_n: int,
        language: Optional[str] = "indonesian",
        coverage_weight: float = 0.6,
        intensity_weight: float = 0.4,
    ):
        if coverage_weight + intensity_weight != 1.0:
            raise ValueError(
                f"Invalid weight. Sum of `coverage_weight` and `intensity_weight` must be equal to `1.0`. Got `{coverage_weight + intensity_weight}`"
            )
        vectorizer = TfidfVectorizer(
            stop_words=self._get_stopwords(language=language), ngram_range=(2, 2)
        )
        docs = [data for data in review_data if len(data[1].split()) > 2]
        if len(docs) < 2:
            raise NotEnoughReviewsError(n=len(docs))
        word_matrix = vectorizer.fit_transform([text for _, text in docs])
        words = vectorizer.get_feature_names_out()
        n_docs = word_matrix.shape[0]

        X = word_matrix.tocsc()
        doc_counts = np.diff(X.indptr)
        coverage = doc_counts / n_docs

        tfidf_sum = np.array(X.sum(axis=0).A1)
        mean_tfidf = tfidf_sum / np.maximum(doc_counts, 1)

        relevancy_score = (coverage**coverage_weight) * (mean_tfidf**intensity_weight)
        consistency_score = np.sqrt(coverage)

        if consistency_score.shape[0] < top_n:
            top_n_idx = np.argsort(-consistency_score)
        else:
            top_n_idx = np.argpartition(-consistency_score, top_n)[:top_n]

        top_n_idx = top_n_idx[np.argsort(-consistency_score[top_n_idx])]

        review_idx, word_idx = word_matrix.nonzero()
        review_ids = np.array([docs[idx][0] for idx in review_idx])

        extracted = []
        for i in top_n_idx:
            mask = word_idx == i
            res = {
                "topic": words[i],
                "review_id": review_ids[mask].tolist(),
                "relevancy_score": relevancy_score[i],
                "consistency_score": consistency_score[i],
            }
            extracted.append(res)
        return extracted
