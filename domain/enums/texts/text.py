from enum import Enum


class TextPlatform(str, Enum):
    gmaps = "google maps"
    user = "user"


class TextSentiment(str, Enum):
    positive = "Positive"
    negative = "Negative"
    neutral = "Neutral"


class TextLanguage(str, Enum):
    ID = "indonesia"
    EN = "english"
