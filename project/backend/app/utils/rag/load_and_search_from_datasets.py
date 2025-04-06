import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
_INDEX_CACHE = None
_LINES_CACHE = None

def load_dataset_embedding(index_file: str = "网安RAG_index", text_file: str = "../data/texts_10000.txt"):   
    global _INDEX_CACHE, _LINES_CACHE 
    
    delimiter = "<<<DOC>>>"
    if _INDEX_CACHE is None:
        index_path = index_file
        try:
            _INDEX_CACHE = faiss.read_index(index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {index_path}: {e}")

    if _LINES_CACHE is None:
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                content = f.read()
                _LINES_CACHE = [seg.strip() for seg in content.split(delimiter) if seg.strip()]
        except Exception as e:
            raise RuntimeError(f"Failed to load text file {text_file}: {e}")
    return _INDEX_CACHE, _LINES_CACHE
    


def search_from_index(query,  k=5):

    model = SentenceTransformer("../models/bge-m3")
    query_vector = model.encode(query, convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    distances, indices = _INDEX_CACHE.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(_LINES_CACHE):
            results.append((_LINES_CACHE[idx], distances[0][i]))

    print(f"Example results: {results}")
    return results