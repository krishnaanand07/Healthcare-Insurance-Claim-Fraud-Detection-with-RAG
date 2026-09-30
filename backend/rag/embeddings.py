import os
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Suppress unnecessary logs
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.model = SentenceTransformer(self.model_name)
            print(f"[EmbeddingService] SentenceTransformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"[EmbeddingService] Warning: Could not load SentenceTransformer ('{e}'). Using fallback TF-IDF/Hash embedding engine.")
            self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                # Normalize vectors for cosine similarity
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (embeddings / norms).astype(np.float32)
            except Exception as e:
                print(f"[EmbeddingService] Error during SentenceTransformer encoding: {e}. Falling back.")

        # Fallback embedding using TF-IDF / Scikit-Learn HashingVectorizer
        return self._fallback_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer
        vectorizer = HashingVectorizer(n_features=384, norm='l2', alternate_sign=False)
        matrix = vectorizer.transform(texts).toarray()
        return matrix.astype(np.float32)

embedding_service = EmbeddingService()
