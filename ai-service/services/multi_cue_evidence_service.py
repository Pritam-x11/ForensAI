class MultiCueEvidenceService:

    def build_evidence(
        self,
        identity,
        structure,
        attributes
    ):

        return {
            "identity": identity,
            "structure": structure,
            "attributes": attributes
        }