from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from functools import lru_cache

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            
        return embeddings[0]
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        return np.vstack([self.get_embedding(text) for text in texts])
        
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return self._cosine_similarity(emb1, emb2)
        
    def find_most_similar(self, query: str, candidates: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find most similar candidates to the query text."""
        query_emb = self.get_embedding(query)
        candidate_embs = self.get_embeddings(candidates)
        
        # Compute similarities
        similarities = [
            self._cosine_similarity(query_emb, candidate_emb)
            for candidate_emb in candidate_embs
        ]
        
        # Sort by similarity
        results = list(zip(candidates, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        return [r for r in results if r[1] >= threshold]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Initialize embedding service at module level
embedding_service = EmbeddingService() 
