class ExplainableResultsService:

    def build_result(
        self,
        candidates,
        evidence,
        quality_weight=None,
        uncertainty=None
    ):

        return {
            "candidates": candidates,
            "evidence": evidence,
            "quality_weight": quality_weight,
            "uncertainty": uncertainty
        }