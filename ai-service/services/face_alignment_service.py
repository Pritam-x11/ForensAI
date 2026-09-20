import cv2
import numpy as np


class FaceAlignmentService:

    def align_face(self, image, face):

        if face.kps is None:
            return None

        source_points = np.array(
            face.kps,
            dtype=np.float32
        )

        target_points = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ],
            dtype=np.float32
        )

        transform_matrix, _ = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.LMEDS
        )

        if transform_matrix is None:
            return None

        aligned_face = cv2.warpAffine(
            image,
            transform_matrix,
            (112, 112)
        )

        return aligned_face