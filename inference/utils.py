from nltk.corpus import stopwords
import re


def get_stopwords():
    ind_stop = set(stopwords.words('indonesian'))
    added_stop = {'yg', 'nya', 'deh', 'ok', 'jg', 'aja', 'sih', 'kali', 'banget', 'bgt', 'nih', 'ya', 
                  'dg', 'tp', 'utk', 'untuk', 'kalok', 'kalau', 'kalo', 'klo', 'kl', 'gak', 'gk', 
                  'udah', 'ga', 'sip', 'jember', 'situbondo'}
    ind_stop |= added_stop

    # Preprocess stopwords the same way scikit-learn would tokenize
    preprocessed = set()
    for word in ind_stop:
        cleaned = re.sub(r"[^a-zA-Z]", "", word.lower())
        if cleaned:
            preprocessed.add(cleaned)
    return list(preprocessed)