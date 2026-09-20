import cv2


class SketchQualityService:

    def calculate_quality(self, image):

        if image is None:
            return {
                "quality_score": 0.0,
                "quality_status": "Poor"
            }

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Estimate sketch sharpness
        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        sharpness_score = min(
            sharpness / 10.0,
            100.0
        )

        quality_score = round(
            sharpness_score,
            2
        )

        if quality_score >= 70:
            quality_status = "Good"

        elif quality_score >= 40:
            quality_status = "Medium"

        else:
            quality_status = "Poor"

        return {
            "quality_score": quality_score,
            "quality_status": quality_status
        }