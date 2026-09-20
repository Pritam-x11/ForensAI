from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from services.investigator_feedback_service import InvestigatorFeedbackService
from services.refined_search_service import RefinedSearchService


router = APIRouter(
    prefix="/api/investigator",
    tags=["Investigator"]
)


feedback_service = InvestigatorFeedbackService()
refined_search_service = RefinedSearchService()


@router.post("/feedback")
def submit_feedback(
    candidate_index: int,
    feedback: str
):

    feedback_result = feedback_service.submit_feedback(
        candidate_index,
        feedback
    )

    return {
        "feedback": feedback_result,
        "refined_search": {
            "candidate_index": candidate_index,
            "status": "Refinement requested"
        }
    }


class RefinedSearchRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    feedback_candidate_index: int


@router.post("/refined-search")
def refined_search(
    request: RefinedSearchRequest
):

    refined_candidates = refined_search_service.refine_candidates(
        request.candidates,
        request.feedback_candidate_index
    )

    return {
        "message": "Refined search completed",
        "candidates": refined_candidates
    }