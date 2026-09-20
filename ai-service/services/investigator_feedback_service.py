class InvestigatorFeedbackService:

    def submit_feedback(
        self,
        candidate_index,
        feedback
    ):
        return {
            "candidate_index": candidate_index,
            "feedback": feedback
        }