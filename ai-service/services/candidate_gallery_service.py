import os
import cv2


class CandidateGalleryService:

    def __init__(
        self,
        arcface_service,
        scrfd_service,
        alignment_service,
        dinov3_service
    ):
        self.arcface_service = arcface_service
        self.scrfd_service = scrfd_service
        self.alignment_service = alignment_service
        self.dinov3_service = dinov3_service

        self.candidates = []

    def load_candidates(self, gallery_path):

        self.candidates = []

        for filename in sorted(os.listdir(gallery_path)):

            print(f"Gallery file found: {filename}")

            file_path = os.path.join(
                gallery_path,
                filename
            )

            if not os.path.isfile(file_path):
                continue

            image = cv2.imread(file_path)

            if image is None:
                print(
                    f"Candidate skipped - image could not be read: {filename}"
                )
                continue

            faces = self.scrfd_service.detect_faces(image)

            if len(faces) == 0:
                print(
                    f"Candidate skipped - no face detected: {filename}"
                )
                continue

            face = faces[0]

            aligned_face = self.alignment_service.align_face(
                image,
                face
            )

            if aligned_face is None:
                print(
                    f"Candidate skipped - alignment failed: {filename}"
                )
                continue

            # ArcFace identity feature
            embedding = self.arcface_service.get_embedding(
                aligned_face
            )

            if embedding is None:
                print(
                    f"Candidate skipped - embedding failed: {filename}"
                )
                continue

            # DINOv3 visual feature
            visual_features = self.dinov3_service.get_features(
                aligned_face
            )

            if visual_features is None:
                print(
                    f"Candidate skipped - DINOv3 feature extraction failed: {filename}"
                )
                continue

            self.candidates.append({
                "candidate_index": len(self.candidates),
                "filename": filename,
                "embedding": embedding,
                "visual_features": visual_features
            })

            print(
                f"Candidate loaded: "
                f"{len(self.candidates) - 1} - {filename}"
            )

            print(
                f"  ArcFace dimension: {len(embedding)}"
            )

            print(
                f"  DINOv3 dimension: {len(visual_features)}"
            )

        return self.candidates

    def get_candidates(self):
        return self.candidates