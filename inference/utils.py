from nltk.corpus import stopwords


def get_stopwords():
    ind_stop = set(stopwords.words('indonesian'))
    added_stop = ['yg', 'nya', 'deh', 'ok', 'jg', 'aja', 'sih', 'kali', 'banget', 'bgt', 'nih', 'ya', 'dg', 'tp', 'utk', 'untuk', 'kalok', 'kalau', 'kalo', 'klo', 'kl', 'gak', 'gk', 'udah', 'ga', 'sip', 'jember', 'situbondo']
    for stop in added_stop:
        ind_stop.add(stop)
    list_stopwords = list(ind_stop)
    return list_stopwords