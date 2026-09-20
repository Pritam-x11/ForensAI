class FeatureLayerService:

    def build_feature_representation(
        self,
        identity_embedding,
        visual_features
    ):

        return {
            "identity_embedding": identity_embedding,
            "visual_features": visual_features
        }