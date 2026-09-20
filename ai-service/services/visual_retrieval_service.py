from services.faiss_service import FAISSService


class VisualRetrievalService:

    def __init__(self):
        self.faiss_service = FAISSService(384)
        self.candidates = []

    def load_candidates(self, candidates):

        self.candidates = []
        self.faiss_service = FAISSService(384)

        for candidate in candidates:

            visual_features = candidate["visual_features"]

            self.faiss_service.add_vector(
                visual_features
            )

            self.candidates.append({
                "candidate_index": candidate["candidate_index"],
                "filename": candidate["filename"]
            })

        return self.candidates

    def search(self, visual_features, top_k=5):

        if self.faiss_service.total_vectors() == 0:
            return [], []

        top_k = min(
            top_k,
            self.faiss_service.total_vectors()
        )

        distances, indices = self.faiss_service.search(
            visual_features,
            top_k
        )

        return distances[0], indices[0]

    def get_candidate(self, faiss_index):

        if faiss_index < 0:
            return None

        if faiss_index >= len(self.candidates):
            return None

        return self.candidates[faiss_index]