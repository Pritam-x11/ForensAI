from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from services.scrfd_service import SCRFDService
from services.preprocessing_service import PreprocessingService
from services.face_alignment_service import FaceAlignmentService
from services.face_quality_service import FaceQualityService
from services.arcface_service import ArcFaceService
from services.dinov3_service import DINOv3Service
from services.feature_layer_service import FeatureLayerService
from services.multi_cue_evidence_service import MultiCueEvidenceService
from services.quality_weight_service import QualityWeightService
from services.uncertainty_service import UncertaintyService
from services.identity_retrieval_service import IdentityRetrievalService
from services.top_candidates_service import TopCandidatesService
from services.evidence_reranking_service import EvidenceRerankingService
from services.explainable_results_service import ExplainableResultsService
from services.candidate_gallery_service import CandidateGalleryService


router = APIRouter(
    prefix="/api/photo",
    tags=["Photo"]
)


# Services
scrfd_service = SCRFDService()
preprocessing_service = PreprocessingService()
alignment_service = FaceAlignmentService()
quality_service = FaceQualityService()
arcface_service = ArcFaceService()
dinov3_service = DINOv3Service()
feature_layer_service = FeatureLayerService()
multi_cue_service = MultiCueEvidenceService()
quality_weight_service = QualityWeightService()
uncertainty_service = UncertaintyService()
identity_retrieval_service = IdentityRetrievalService()
top_candidates_service = TopCandidatesService()
evidence_reranking_service = EvidenceRerankingService()
explainable_results_service = ExplainableResultsService()


# Candidate Gallery
candidate_gallery_service = CandidateGalleryService(
    arcface_service,
    scrfd_service,
    alignment_service,
    dinov3_service
)

loaded_candidates = candidate_gallery_service.load_candidates(
    "gallery/candidates"
)


# Load candidate embeddings into FAISS
identity_retrieval_service.load_candidates(
    loaded_candidates
)


print(
    f"ForensAI Photo Candidate Gallery Loaded: "
    f"{len(loaded_candidates)} candidates"
)


@router.post("/upload")
async def upload_photo(
    file: UploadFile = File(...)
):

    # Read uploaded image
    image_bytes = await file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    # Invalid image check
    if image is None:
        return {
            "message": "Invalid image file"
        }


    # Preprocessing
    image = preprocessing_service.preprocess(
        image
    )


    # Face Detection
    faces = scrfd_service.detect_faces(
        image
    )


    detected_faces = []


    # Process every detected face
    for face in faces:

        # Face Alignment
        aligned_face = alignment_service.align_face(
            image,
            face
        )


        # Face Quality
        quality = quality_service.calculate_quality(
            aligned_face
        )


        # Quality Weight
        quality_weight = (
            quality_weight_service.calculate_weight(
                quality["quality_score"]
            )
        )


        # Uncertainty
        uncertainty = (
            uncertainty_service.calculate_uncertainty(
                quality["quality_score"]
            )
        )


        # ArcFace Identity Embedding
        embedding = arcface_service.get_embedding(
            aligned_face
        )


        # DINOv3 Visual Features
        dinov3_features = dinov3_service.get_features(
            aligned_face
        )


        # Feature Layer
        feature_representation = (
            feature_layer_service.build_feature_representation(
                embedding,
                dinov3_features
            )
        )


        # Multi-Cue Evidence
        evidence = multi_cue_service.build_evidence(
            identity=feature_representation[
                "identity_embedding"
            ],
            structure=None,
            attributes=None
        )


        # -------------------------------------------------
        # FAISS Candidate Gallery Search
        # -------------------------------------------------

        if embedding is not None:

            retrieval_distances, retrieval_indices = (
                identity_retrieval_service.search_identity(
                    embedding,
                    top_k=2
                )
            )

        else:

            retrieval_distances = []
            retrieval_indices = []


        # -------------------------------------------------
        # Top Candidates
        # -------------------------------------------------

        candidate_metadata = []

        for index in retrieval_indices:

            candidate = (
                identity_retrieval_service.get_candidate(
                    int(index)
                )
            )

            if candidate is not None:
                candidate_metadata.append(
                    candidate["filename"]
                )


        top_candidates = (
            top_candidates_service.get_top_candidates(
                retrieval_indices,
                retrieval_distances,
                candidate_metadata
            )
        )


        # -------------------------------------------------
        # Evidence Reranking
        # -------------------------------------------------

        reranked_result = (
            evidence_reranking_service.rerank(
                top_candidates,
                evidence
            )
        )


        # -------------------------------------------------
        # Explainable Results
        # -------------------------------------------------

        explainable_result = (
            explainable_results_service.build_result(
                reranked_result["candidates"],
                reranked_result["evidence"]
            )
        )


        # -------------------------------------------------
        # Final Face Result
        # -------------------------------------------------

        detected_faces.append({

            "bounding_box": (
                face.bbox.tolist()
            ),

            "detection_score": (
                float(face.det_score)
            ),

            "alignment_success": (
                aligned_face is not None
            ),

            "quality_score": (
                quality["quality_score"]
            ),

            "quality_status": (
                quality["quality_status"]
            ),

            "quality_weight": (
                quality_weight
            ),

            "uncertainty": (
                uncertainty
            ),

            "embedding_dimension": (
                len(embedding)
                if embedding is not None
                else 0
            ),

            "dino_feature_dimension": (
                len(dinov3_features)
            ),

            "top_candidates": (
                top_candidates
            ),

            "explainable_result": {

                "candidates": (
                    explainable_result[
                        "candidates"
                    ]
                ),

                "evidence": {

                    "identity_dimension": (
                        len(
                            explainable_result[
                                "evidence"
                            ]["identity"]
                        )
                    ),

                    "structure": (
                        explainable_result[
                            "evidence"
                        ]["structure"]
                    ),

                    "attributes": (
                        explainable_result[
                            "evidence"
                        ]["attributes"]
                    )
                }
            }
        })


    # Final response
    return {

        "message": (
            "Photo processed successfully"
        ),

        "filename": (
            file.filename
        ),

        "faces_detected": (
            len(faces)
        ),

        "faces": (
            detected_faces
        )
    }