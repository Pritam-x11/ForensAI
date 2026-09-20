import numpy as np


class TemporalLayerService:

    def aggregate_embeddings(self, embeddings):
        if not embeddings:
            return None

        vectors = np.asarray(embeddings, dtype=np.float32)

        aggregated_embedding = np.mean(vectors, axis=0)

        return aggregated_embedding