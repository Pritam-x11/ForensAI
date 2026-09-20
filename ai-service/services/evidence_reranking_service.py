class EvidenceRerankingService:

    def rerank(
        self,
        candidates,
        evidence,
        quality_weight=None,
        uncertainty=None
    ):

        if not candidates:
            return {
                "candidates": [],
                "evidence": evidence,
                "quality_weight": quality_weight,
                "uncertainty": uncertainty
            }

        # ---------------------------------------------
        # Extract Quality Weight
        # ---------------------------------------------

        if isinstance(quality_weight, dict):

            quality_value = quality_weight.get(
                "quality_weight",
                1.0
            )

        else:

            quality_value = (
                quality_weight
                if quality_weight is not None
                else 1.0
            )


        quality_value = float(quality_value)


        # ---------------------------------------------
        # Extract Uncertainty / Confidence
        # ---------------------------------------------

        if isinstance(uncertainty, dict):

            confidence = uncertainty.get(
                "confidence",
                100.0
            )

        else:

            confidence = (
                uncertainty
                if uncertainty is not None
                else 100.0
            )


        confidence = float(confidence)


        # ---------------------------------------------
        # Normalize Confidence
        # ---------------------------------------------

        confidence_weight = max(
            0.0,
            min(confidence / 100.0, 1.0)
        )


        # ---------------------------------------------
        # Combined Evidence Weight
        # ---------------------------------------------

        evidence_weight = (
            quality_value *
            confidence_weight
        )


        # ---------------------------------------------
        # Distance Range
        # ---------------------------------------------

        distances = [

            float(candidate.get(
                "distance",
                0.0
            ))

            for candidate in candidates
        ]


        min_distance = min(distances)

        max_distance = max(distances)


        # ---------------------------------------------
        # Calculate Ranking Score
        # ---------------------------------------------

        reranked_candidates = []


        for candidate in candidates:

            distance = float(
                candidate.get(
                    "distance",
                    0.0
                )
            )


            # Lower distance = better match
            if max_distance == min_distance:

                similarity_score = 1.0

            else:

                similarity_score = (
                    (max_distance - distance)
                    /
                    (max_distance - min_distance)
                )


            # Apply quality + confidence
            final_score = (
                similarity_score *
                evidence_weight
            )


            reranked_candidate = dict(
                candidate
            )


            reranked_candidate[
                "similarity_score"
            ] = round(
                similarity_score,
                4
            )


            reranked_candidate[
                "evidence_weight"
            ] = round(
                evidence_weight,
                4
            )


            reranked_candidate[
                "final_score"
            ] = round(
                final_score,
                4
            )


            reranked_candidates.append(
                reranked_candidate
            )


        # ---------------------------------------------
        # Sort by Final Score
        # ---------------------------------------------

        reranked_candidates.sort(
            key=lambda candidate:
                candidate["final_score"],
            reverse=True
        )


        # ---------------------------------------------
        # Return Result
        # ---------------------------------------------

        return {

            "candidates":
                reranked_candidates,

            "evidence":
                evidence,

            "quality_weight":
                quality_weight,

            "uncertainty":
                uncertainty
        }