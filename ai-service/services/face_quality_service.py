import cv2


class FaceQualityService:

    def calculate_quality(self, aligned_face):

        if aligned_face is None:
            return {
                "quality_score": 0.0,
                "quality_status": "Poor"
            }

        gray = cv2.cvtColor(
            aligned_face,
            cv2.COLOR_BGR2GRAY
        )

        # Sharpness / blur measurement
        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        # Normalize sharpness into a 0-100 score
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