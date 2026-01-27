CANON_6 = ["anger", "disgust", "fear", "happy", "sad", "surprise"]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CANON_6)}

ALIASES = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
    "contempt": "contempt",
}


def normalize_emotion(name: str) -> str:
    key = name.strip().lower()
    return ALIASES.get(key, key)


__all__ = ["CANON_6", "CLASS_TO_IDX", "normalize_emotion"]
