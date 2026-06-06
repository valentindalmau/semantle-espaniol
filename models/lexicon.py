import os
import re
from typing import Set


LEXICON_PATH = os.getenv("LEXICON_PATH", "data/lexicon.txt")
TARGETS_PATH = os.getenv("TARGETS_PATH", "data/targets_es.txt")
WORD_RE = re.compile(r"^[a-záéíóúüñ]+(?:[-'][a-záéíóúüñ]+)?$", re.IGNORECASE)


def load_list(path: str) -> Set[str]:
    words: Set[str] = set()
    if not os.path.exists(path):
        return words

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                words.add(word)
    return words


LEXICON = load_list(LEXICON_PATH)
TARGETS = load_list(TARGETS_PATH)


def is_valid_word(word: str) -> bool:
    word = (word or "").strip().lower()
    return bool(word and len(word) <= 40 and WORD_RE.match(word))


def is_known_word(word: str) -> bool:
    word = (word or "").strip().lower()
    return word in LEXICON or word in TARGETS
