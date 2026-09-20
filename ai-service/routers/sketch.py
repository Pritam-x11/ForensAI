from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from services.sketch_preprocessing_service import (
    SketchPreprocessingService
)
from services.sketch_quality_service import (
    SketchQualityService
)
from services.vlm_attributes_service import (
    VLMAttributesService
)
from services.sketch_representation_service import (
    SketchRepresentationService
)
from services.cross_modal_features_service import (
    CrossModalFeaturesService
)
from services.feature_layer_service import (
    FeatureLayerService
)
from services.multi_cue_evidence_service import (
    MultiCueEvidenceService
)
from services.quality_weight_service import (
    QualityWeightService
)
from services.uncertainty_service import (
    UncertaintyService
)
from services.identity_retrieval_service import (
    IdentityRetrievalService
)
from services.candidate_gallery_service import (
    CandidateGalleryService
)
from services.scrfd_service import SCRFDService
from services.arcface_service import ArcFaceService
from services.face_alignment_service import FaceAlignmentService
from services.top_candidates_service import TopCandidatesService
from services.evidence_reranking_service import (
    EvidenceRerankingService
)
from services.explainable_results_service import (
    ExplainableResultsService
)
from services.dinov3_service import DINOv3Service
from services.visual_retrieval_service import (
    VisualRetrievalService
)


router = APIRouter(
    prefix="/api/sketch",
    tags=["Sketch"]
)


# =========================================================
# Services
# =========================================================

preprocessing_service = SketchPreprocessingService()

quality_service = SketchQualityService()

vlm_attributes_service = VLMAttributesService()

representation_service = SketchRepresentationService()

cross_modal_service = CrossModalFeaturesService()

feature_layer_service = FeatureLayerService()

multi_cue_service = MultiCueEvidenceService()

quality_weight_service = QualityWeightService()

uncertainty_service = UncertaintyService()

identity_retrieval_service = IdentityRetrievalService()

scrfd_service = SCRFDService()

arcface_service = ArcFaceService()

alignment_service = FaceAlignmentService()

top_candidates_service = TopCandidatesService()

evidence_reranking_service = EvidenceRerankingService()

explainable_results_service = ExplainableResultsService()

dinov3_service = DINOv3Service()

visual_retrieval_service = VisualRetrievalService()


# =========================================================
# Candidate Gallery
# =========================================================

candidate_gallery_service = CandidateGalleryService(
    arcface_service,
    scrfd_service,
    alignment_service,
    dinov3_service
)


loaded_candidates = candidate_gallery_service.load_candidates(
    "gallery/candidates"
)


# =========================================================
# Load ArcFace Identity Embeddings into FAISS
# =========================================================

identity_retrieval_service.load_candidates(
    loaded_candidates
)


# =========================================================
# Load DINOv3 Visual Features into Visual FAISS
# =========================================================

visual_retrieval_service.load_candidates(
    loaded_candidates
)


print(
    f"ForensAI Sketch Candidate Gallery Loaded: "
    f"{len(loaded_candidates)} candidates"
)


# =========================================================
# Sketch Upload
# =========================================================

@router.post("/upload")
async def upload_sketch(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Read uploaded sketch
    # -----------------------------------------------------

    image_bytes = await file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {
            "message": "Invalid sketch file"
        }


    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    image = preprocessing_service.preprocess(
        image
    )


    # -----------------------------------------------------
    # Quality Check
    # -----------------------------------------------------

    quality = quality_service.calculate_quality(
        image
    )


    # -----------------------------------------------------
    # VLM Attributes
    # -----------------------------------------------------

    attributes = (
        vlm_attributes_service.extract_attributes(
            image
        )
    )


    # -----------------------------------------------------
    # DINOv3 Visual Feature Extraction
    # -----------------------------------------------------

    visual_features = dinov3_service.get_features(
        image
    )


    # -----------------------------------------------------
    # Sketch Representation
    # -----------------------------------------------------

    sketch_representation = (
        representation_service.build_representation(
            image,
            attributes,
            visual_features
        )
    )


    # -----------------------------------------------------
    # Cross-modal Features
    # -----------------------------------------------------

    photo_features = [
        {
            "candidate_index": candidate["candidate_index"],
            "filename": candidate["filename"],
            "embedding": candidate["embedding"],
            "visual_features": candidate["visual_features"]
        }
        for candidate in loaded_candidates
    ]


    cross_modal_features = (
        cross_modal_service.build_features(
            sketch_representation,
            photo_features
        )
    )


    # -----------------------------------------------------
    # Feature Layer
    # -----------------------------------------------------

    feature_representation = (
        feature_layer_service.build_feature_representation(
            identity_embedding=None,
            visual_features=cross_modal_features
        )
    )


    # -----------------------------------------------------
    # Multi-Cue Evidence
    # -----------------------------------------------------

    evidence = multi_cue_service.build_evidence(
        identity=feature_representation[
            "identity_embedding"
        ],
        structure=None,
        attributes=attributes
    )


    # -----------------------------------------------------
    # Quality Weight
    # -----------------------------------------------------

    quality_weight = (
        quality_weight_service.calculate_weight(
            quality["quality_score"]
        )
    )


    # -----------------------------------------------------
    # Uncertainty Handling
    # -----------------------------------------------------

    uncertainty = (
        uncertainty_service.calculate_uncertainty(
            quality["quality_score"]
        )
    )


    # -----------------------------------------------------
    # Visual FAISS Retrieval
    # -----------------------------------------------------

    visual_distances, visual_indices = (
        visual_retrieval_service.search(
            visual_features,
            top_k=5
        )
    )


    # -----------------------------------------------------
    # Candidate Metadata
    # -----------------------------------------------------

    candidate_metadata = []

    for index in visual_indices:

        candidate = (
            visual_retrieval_service.get_candidate(
                int(index)
            )
        )

        if candidate is not None:

            candidate_metadata.append(
                candidate["filename"]
            )


    # -----------------------------------------------------
    # Top Candidates
    # -----------------------------------------------------

    top_candidates = (
        top_candidates_service.get_top_candidates(
            visual_indices,
            visual_distances,
            candidate_metadata
        )
    )


    # -----------------------------------------------------
    # Evidence Reranking
    # -----------------------------------------------------

    reranked_result = (
    evidence_reranking_service.rerank(
        top_candidates,
        evidence,
        quality_weight,
        uncertainty
    )
)

    # -----------------------------------------------------
    # Explainable Results
    # -----------------------------------------------------

    explainable_result = (
    explainable_results_service.build_result(
        reranked_result["candidates"],
        reranked_result["evidence"],
        quality_weight,
        uncertainty
    )
)


    # -----------------------------------------------------
    # Final Response
    # -----------------------------------------------------

    return {

        "message": "Sketch processed successfully",

        "filename": file.filename,


        "quality": quality,


        "vlm_attributes": attributes,


        "sketch_representation": {
            "status": "Generated"
        },


        "cross_modal_features": {

            "status": "Generated",

            "matching_ready": (
                cross_modal_features[
                    "matching_ready"
                ]
            ),

            "sketch_visual_feature_dimension": (
                len(
                    cross_modal_features[
                        "sketch_representation"
                    ][
                        "visual_features"
                    ]
                )
                if cross_modal_features[
                    "sketch_representation"
                ][
                    "visual_features"
                ] is not None
                else 0
            ),

            "photo_candidates_available": len(
                cross_modal_features[
                    "photo_features"
                ]
            )
        },


        "feature_layer": {
            "status": "Generated"
        },


        "multi_cue_evidence": {

            "identity": (
                "Not generated yet"
                if evidence["identity"] is None
                else "Generated"
            ),

            "structure": evidence["structure"],

            "attributes": evidence["attributes"]
        },


        "quality_weight": quality_weight,


        "uncertainty": uncertainty,


        "top_candidates": top_candidates,


        "evidence_reranking": {

            "candidates": (
                reranked_result["candidates"]
            )
        },


        "explainable_results": {

            "candidates": (
                explainable_result["candidates"]
            ),

            "evidence": {

                "identity_dimension": (
                    len(
                        explainable_result[
                            "evidence"
                        ][
                            "identity"
                        ]
                    )
                    if explainable_result[
                        "evidence"
                    ][
                        "identity"
                    ] is not None
                    else 0
                ),

                "structure": (
                    explainable_result[
                        "evidence"
                    ][
                        "structure"
                    ]
                ),

                "attributes": (
                    explainable_result[
                        "evidence"
                    ][
                        "attributes"
                    ]
                )
            }
        },


        "candidate_gallery": {

            "candidates_loaded": len(
                loaded_candidates
            ),

            "identity_faiss_ready": (
                identity_retrieval_service
                .total_identities() > 0
            ),

            "visual_faiss_ready": (
                visual_retrieval_service
                .faiss_service
                .total_vectors() > 0
            )
        }
    }