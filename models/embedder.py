from typing import List

import fasttext
import numpy as np


class SpanishFastTextEmbedder:
    def __init__(self, model_path: str = "data/cc.es.300.bin"):
        self.model = fasttext.load_model(model_path)

    @staticmethod
    def normalize(text: str) -> str:
        return (text or "").strip().lower()

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        vectors = [self.model.get_word_vector(self.normalize(text)) for text in texts]
        arr = np.vstack(vectors).astype(np.float32)
        if normalize:
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
