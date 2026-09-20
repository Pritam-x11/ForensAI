class RefinedSearchService:

    def refine_candidates(
        self,
        candidates,
        feedback_candidate_index
    ):
        refined_candidates = []

        for candidate in candidates:

            candidate_copy = candidate.copy()

            if candidate_copy.get("candidate_index") == feedback_candidate_index:
                candidate_copy["feedback_relevant"] = True
            else:
                candidate_copy["feedback_relevant"] = False

            refined_candidates.append(candidate_copy)

        refined_candidates.sort(
            key=lambda candidate: (
                candidate.get("feedback_relevant", False),
                candidate.get("final_score", 0.0)
            ),
            reverse=True
        )

        return refined_candidates