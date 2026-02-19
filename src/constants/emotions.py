CANON_6 = ["anger", "disgust", "fear", "happy", "sad", "surprise"]

CLASS_TO_IDX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
}

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
    """Normalize label variants to a shared emotion naming scheme."""
    key = name.strip().lower()
    return ALIASES.get(key, key)


__all__ = ["CANON_6", "CLASS_TO_IDX", "normalize_emotion"]
