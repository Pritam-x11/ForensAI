import faiss
import numpy as np


class FAISSService:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def add_vector(self, vector):
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        self.index.add(vector)

    def search(self, vector, top_k=5):
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        distances, indices = self.index.search(vector, top_k)

        return distances, indices

    def total_vectors(self):
        return self.index.ntotal