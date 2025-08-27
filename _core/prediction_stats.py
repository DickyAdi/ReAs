from collections import Counter

def count_sentiments(sentiment:list[str]) -> dict[str, int]:
    return dict(Counter(sentiment))