class TopCandidatesService:

    def get_top_candidates(
        self,
        indices,
        distances,
        candidate_metadata=None
    ):
        candidates = []

        for position, (index, distance) in enumerate(
            zip(indices, distances)
        ):
            candidate = {
                "candidate_index": int(index),
                "distance": float(distance)
            }

            if candidate_metadata is not None:
                if position < len(candidate_metadata):
                    candidate["filename"] = candidate_metadata[position]

            candidates.append(candidate)

        return candidates