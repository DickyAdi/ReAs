from enum import Enum

class TextPlatform(str, Enum):
    gmaps = "google maps"

class TextSentiment(str, Enum):
    positive = "Positive"
    negative = "Negative"
    neutral = "neutral"

class TextLanguage(str, Enum):
    ID = 'indonesia'
    EN = 'english'