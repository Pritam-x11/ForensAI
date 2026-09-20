from fastapi import APIRouter, UploadFile, File
import tempfile
import os
import numpy as np
import supervision as sv

from services.frame_sampling_service import FrameSamplingService
from services.scrfd_service import SCRFDService
from services.video_tracking_service import VideoTrackingService
from services.face_quality_service import FaceQualityService
from services.arcface_service import ArcFaceService
from services.face_alignment_service import FaceAlignmentService
from services.temporal_layer_service import TemporalLayerService
from services.multi_cue_evidence_service import MultiCueEvidenceService
from services.quality_weight_service import QualityWeightService
from services.uncertainty_service import UncertaintyService
from services.identity_retrieval_service import IdentityRetrievalService
from services.top_candidates_service import TopCandidatesService
from services.evidence_reranking_service import EvidenceRerankingService
from services.explainable_results_service import ExplainableResultsService
from services.candidate_gallery_service import CandidateGalleryService
from services.dinov3_service import DINOv3Service


router = APIRouter(
    prefix="/api/video",
    tags=["Video"]
)


# =========================================================
# Services
# =========================================================

frame_sampling_service = FrameSamplingService()

scrfd_service = SCRFDService()

tracking_service = VideoTrackingService()

quality_service = FaceQualityService()

arcface_service = ArcFaceService()

alignment_service = FaceAlignmentService()

temporal_layer_service = TemporalLayerService()

multi_cue_service = MultiCueEvidenceService()

quality_weight_service = QualityWeightService()

uncertainty_service = UncertaintyService()

identity_retrieval_service = IdentityRetrievalService()

top_candidates_service = TopCandidatesService()

evidence_reranking_service = EvidenceRerankingService()

explainable_results_service = ExplainableResultsService()

dinov3_service = DINOv3Service()


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
# Load Candidate Embeddings into FAISS
# =========================================================

identity_retrieval_service.load_candidates(
    loaded_candidates
)


print(
    f"ForensAI Candidate Gallery Loaded: "
    f"{len(loaded_candidates)} candidates"
)


# =========================================================
# Video Upload
# =========================================================

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Read uploaded video
    # -----------------------------------------------------

    video_bytes = await file.read()

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    try:

        temp_video.write(video_bytes)
        temp_video.close()


        # =================================================
        # Frame Sampling
        # =================================================

        frames = frame_sampling_service.sample_frames(
            temp_video.name,
            sample_rate=5
        )


        # =================================================
        # Counters
        # =================================================

        frames_with_faces = 0

        total_faces_detected = 0

        total_tracked_detections = 0


        # =================================================
        # Track-Level Storage
        # =================================================

        track_ids = set()

        track_quality = {}

        track_embeddings = {}

        track_embedding_history = {}

        track_evidence = {}

        track_candidates = {}

        track_explainable_results = {}


        # =================================================
        # Process Sampled Frames
        # =================================================

        for frame in frames:

            # ---------------------------------------------
            # Face Detection
            # ---------------------------------------------

            faces = scrfd_service.detect_faces(
                frame
            )


            if len(faces) > 0:

                frames_with_faces += 1

                total_faces_detected += len(faces)


            # ---------------------------------------------
            # Convert Detection to ByteTrack Format
            # ---------------------------------------------

            xyxy = []

            confidence = []


            for face in faces:

                xyxy.append(
                    face.bbox.tolist()
                )

                confidence.append(
                    float(face.det_score)
                )


            detections = sv.Detections(

                xyxy=(
                    np.array(
                        xyxy,
                        dtype=np.float32
                    )
                    if xyxy
                    else np.empty(
                        (0, 4),
                        dtype=np.float32
                    )
                ),

                confidence=(
                    np.array(
                        confidence,
                        dtype=np.float32
                    )
                    if confidence
                    else np.empty(
                        (0,),
                        dtype=np.float32
                    )
                )
            )


            # ---------------------------------------------
            # ByteTrack Tracking
            # ---------------------------------------------

            tracked_detections = (
                tracking_service.update_tracks(
                    detections
                )
            )


            # =================================================
            # Process Tracked Faces
            # =================================================

            if len(tracked_detections) > 0:

                for i in range(
                    len(tracked_detections)
                ):

                    # -----------------------------------------
                    # Bounding Box
                    # -----------------------------------------

                    x1, y1, x2, y2 = (
                        tracked_detections.xyxy[i]
                    )


                    x1 = max(
                        0,
                        int(x1)
                    )

                    y1 = max(
                        0,
                        int(y1)
                    )

                    x2 = min(
                        frame.shape[1],
                        int(x2)
                    )

                    y2 = min(
                        frame.shape[0],
                        int(y2)
                    )


                    face_crop = frame[
                        y1:y2,
                        x1:x2
                    ]


                    # -----------------------------------------
                    # Face Quality
                    # -----------------------------------------

                    quality = (
                        quality_service.calculate_quality(
                            face_crop
                        )
                    )


                    # -----------------------------------------
                    # Quality Weight
                    # -----------------------------------------

                    quality_weight = (
                        quality_weight_service.calculate_weight(
                            quality["quality_score"]
                        )
                    )


                    # -----------------------------------------
                    # Uncertainty
                    # -----------------------------------------

                    uncertainty = (
                        uncertainty_service.calculate_uncertainty(
                            quality["quality_score"]
                        )
                    )


                    # -----------------------------------------
                    # ArcFace Embedding
                    # -----------------------------------------

                    embedding = (
                        arcface_service.get_embedding(
                            face_crop
                        )
                    )


                    # -----------------------------------------
                    # Track ID
                    # -----------------------------------------

                    track_id = (
                        tracked_detections.tracker_id[i]
                    )


                    if track_id is not None:

                        track_id_int = int(
                            track_id
                        )


                        # =====================================
                        # Store Track Quality
                        # =====================================

                        track_quality[
                            track_id_int
                        ] = quality


                        track_quality[
                            track_id_int
                        ]["quality_weight"] = (
                            quality_weight
                        )


                        track_quality[
                            track_id_int
                        ]["uncertainty"] = (
                            uncertainty
                        )


                        # =====================================
                        # Embedding History
                        # =====================================

                        if embedding is not None:

                            if (
                                track_id_int
                                not in track_embedding_history
                            ):

                                track_embedding_history[
                                    track_id_int
                                ] = []


                            track_embedding_history[
                                track_id_int
                            ].append(
                                embedding.tolist()
                            )


                            # =================================
                            # Temporal Layer
                            # =================================

                            aggregated_embedding = (
                                temporal_layer_service
                                .aggregate_embeddings(
                                    track_embedding_history[
                                        track_id_int
                                    ]
                                )
                            )


                            # =================================
                            # Store Aggregated Embedding
                            # =================================

                            track_embeddings[
                                track_id_int
                            ] = (
                                aggregated_embedding.tolist()
                            )


                            # =================================
                            # Identity Retrieval
                            # =================================

                            (
                                retrieval_distances,
                                retrieval_indices
                            ) = (
                                identity_retrieval_service
                                .search_identity(
                                    aggregated_embedding,
                                    top_k=2
                                )
                            )


                            # =================================
                            # Candidate Metadata
                            # =================================

                            candidate_metadata = []

                            for index in retrieval_indices:

                                candidate = (
                                    identity_retrieval_service
                                    .get_candidate(
                                        int(index)
                                    )
                                )

                                if candidate is not None:

                                    candidate_metadata.append(
                                        candidate["filename"]
                                    )


                            # =================================
                            # Top Candidates
                            # =================================

                            top_candidates = (
                                top_candidates_service
                                .get_top_candidates(
                                    retrieval_indices,
                                    retrieval_distances,
                                    candidate_metadata
                                )
                            )


                            # =================================
                            # Multi-Cue Evidence
                            # =================================

                            evidence = (
                                multi_cue_service.build_evidence(
                                    identity=aggregated_embedding,
                                    structure=None,
                                    attributes=None
                                )
                            )


                            # =================================
                            # JSON-Safe Evidence
                            # =================================

                            track_evidence[
                                track_id_int
                            ] = {

                                "identity": (
                                    aggregated_embedding.tolist()
                                ),

                                "structure": (
                                    evidence["structure"]
                                ),

                                "attributes": (
                                    evidence["attributes"]
                                )
                            }


                            # =================================
                            # Evidence Reranking
                            # =================================

                            reranked_result = (
                                evidence_reranking_service.rerank(
                                    top_candidates,
                                    evidence,
                                    quality_weight,
                                    uncertainty
                                )
                            )


                            # =================================
                            # Explainable Results
                            # =================================

                            explainable_result = (
                                explainable_results_service.build_result(
                                    reranked_result["candidates"],
                                    reranked_result["evidence"],
                                    quality_weight,
                                    uncertainty
                                )
                            )


                            # =================================
                            # Store Explainable Result
                            # =================================

                            track_explainable_results[
                                track_id_int
                            ] = {

                                "candidates": (
                                    explainable_result[
                                        "candidates"
                                    ]
                                ),

                                "evidence": {

                                    "identity_dimension": len(
                                        explainable_result[
                                            "evidence"
                                        ]["identity"]
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
                                },

                                "quality_weight": (
                                    explainable_result[
                                        "quality_weight"
                                    ]
                                ),

                                "uncertainty": (
                                    explainable_result[
                                        "uncertainty"
                                    ]
                                )
                            }


                            # =================================
                            # Store Track Candidates
                            # =================================

                            track_candidates[
                                track_id_int
                            ] = (
                                reranked_result[
                                    "candidates"
                                ]
                            )


            # =================================================
            # Track IDs
            # =================================================

            if (
                tracked_detections.tracker_id
                is not None
            ):

                valid_track_ids = [

                    int(track_id)

                    for track_id
                    in tracked_detections.tracker_id

                    if track_id is not None
                ]


                total_tracked_detections += len(
                    valid_track_ids
                )


                track_ids.update(
                    valid_track_ids
                )


        # =================================================
        # Final Response
        # =================================================

        return {

            "message":
                "Video processed successfully",


            "filename":
                file.filename,


            "frames_sampled":
                len(frames),


            "frames_with_faces":
                frames_with_faces,


            "total_faces_detected":
                total_faces_detected,


            "total_tracked_detections":
                total_tracked_detections,


            "unique_track_ids":
                sorted(track_ids),


            "track_quality":
                track_quality,


            "track_embeddings":
                track_embeddings,


            "track_evidence":
                track_evidence,


            "track_candidates":
                track_candidates,


            "track_explainable_results":
                track_explainable_results
        }


    finally:

        # -----------------------------------------------------
        # Delete Temporary Video
        # -----------------------------------------------------

        if os.path.exists(
            temp_video.name
        ):

            os.remove(
                temp_video.name
            )