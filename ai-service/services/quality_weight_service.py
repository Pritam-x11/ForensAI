class QualityWeightService:

    def calculate_weight(self, quality_score):

        if quality_score is None:
            quality_score = 0.0

        quality_score = float(quality_score)

        quality_score = max(
            0.0,
            min(quality_score, 100.0)
        )

        quality_weight = quality_score / 100.0

        return {
            "quality_score": round(quality_score, 2),
            "quality_weight": round(quality_weight, 4)
        }