from services.faiss_service import FAISSService


class IdentityRetrievalService:

    def __init__(self):
        self.faiss_service = FAISSService(512)
        self.candidates = []

    def load_candidates(self, candidates):
        self.candidates = []

        # FAISS index ko fresh start karo
        self.faiss_service = FAISSService(512)

        for candidate in candidates:
            embedding = candidate["embedding"]

            self.faiss_service.add_vector(embedding)

            self.candidates.append({
                "candidate_index": candidate["candidate_index"],
                "filename": candidate["filename"]
            })

        return self.candidates

    def search_identity(self, embedding, top_k=5):

        if self.faiss_service.total_vectors() == 0:
            return [], []

        top_k = min(
            top_k,
            self.faiss_service.total_vectors()
        )

        distances, indices = self.faiss_service.search(
            embedding,
            top_k
        )

        return distances[0], indices[0]

    def total_identities(self):
        return self.faiss_service.total_vectors()

    def get_candidate(self, faiss_index):

        if faiss_index < 0 or faiss_index >= len(self.candidates):
            return None

        return self.candidates[faiss_index]