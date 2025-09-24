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

        # Preprocess stopwords the same way scikit-learn would tokenize
        preprocessed = set()
        for word in ind_stop:
            cleaned = re.sub(r"[^a-zA-Z]", "", word.lower())
            if cleaned:
                preprocessed.add(cleaned)
        return list(preprocessed)

    # def extract(self, text: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     vectorizer = TfidfVectorizer(
    #         stop_words=self._get_stopwords(), ngram_range=(2, 2)
    #     )
    #     docs = [doc for doc in text if len(doc.split()) > 2]
    #     if not text or len(text) < 2 or not docs:
    #         return (
    #             pd.DataFrame(columns=["word", "score"]),
    #             pd.DataFrame(columns=["word", "score"]),
    #         )
    #     word_matrix = vectorizer.fit_transform(docs)
    #     words = vectorizer.get_feature_names_out()

    #     # memory first code

    #     count_matrix = (word_matrix > 0).sum(axis=0).A1
    #     matrix_mean = word_matrix.mean(axis=0).A1
    #     matrix_sq_mean = word_matrix.power(2).mean(axis=0).A1
    #     std = np.sqrt(matrix_sq_mean - matrix_mean**2)
    #     score = matrix_mean * std
    #     score_trend = score * (count_matrix / len(docs))

    #     # end memory first code

    #     trend_topics = pd.DataFrame({"word": words, "score": score_trend})
    #     frequent_topics = pd.DataFrame({"word": words, "score": score})
    #     del word_matrix, std, matrix_mean, matrix_sq_mean
    #     gc.collect()
    #     return trend_topics, frequent_topics

    def extract(
        self,
        review_data: list[tuple],
        top_n: int,
        language: Optional[str] = "indonesian",
    ):
        vectorizer = TfidfVectorizer(
            stop_words=self._get_stopwords(language=language), ngram_range=(2, 2)
        )
        docs = [data for data in review_data if len(data[1].split()) > 2]
        if len(docs) < 2:
            raise NotEnoughReviewsError(n=len(docs))
        word_matrix = vectorizer.fit_transform([text for _, text in docs])
        words = vectorizer.get_feature_names_out()
        n_docs = word_matrix.shape[0]

        count_matrix = (word_matrix > 0).sum(axis=0).A1
        matrix_mean = word_matrix.mean(axis=0).A1
        matrix_sq_mean = word_matrix.power(2).mean(axis=0).A1
        std = np.sqrt(matrix_sq_mean - matrix_mean**2)
        emerging_score = matrix_mean * std
        score_trend = emerging_score * (count_matrix / n_docs)

        review_idx, word_idx = word_matrix.nonzero()
        review_ids = np.array([docs[idx][0] for idx in review_idx])

        top_n_idx = np.argpartition(-emerging_score, top_n)[:top_n]
        top_n_idx = top_n_idx[np.argsort(-emerging_score[top_n_idx])]

        extracted = []
        for i in top_n_idx:
            mask = word_idx == i
            res = {
                "topic": words[i],
                "review_id": review_ids[mask].tolist(),
                "emerging_score": emerging_score[i],
                "trend_score": score_trend[i],
            }
            extracted.append(res)
        return extracted

        # should've been done i guess
