import numpy as np


class CrossModalFeaturesService:

    def build_features(
        self,
        sketch_representation,
        photo_features=None
    ):

        if sketch_representation is None:
            return {
                "sketch_representation": None,
                "photo_features": photo_features,
                "matching_ready": False,
                "sketch_feature": None,
                "photo_feature_dimensions": []
            }

        sketch_visual_features = (
            sketch_representation.get(
                "visual_features"
            )
        )

        photo_feature_dimensions = []

        if photo_features is not None:

            for candidate in photo_features:

                visual_features = candidate.get(
                    "visual_features"
                )

                if visual_features is not None:

                    photo_feature_dimensions.append(
                        len(visual_features)
                    )

        matching_ready = (
            sketch_visual_features is not None
            and photo_features is not None
            and len(photo_features) > 0
        )

        return {
            "sketch_representation":
                sketch_representation,

            "photo_features":
                photo_features,

            "matching_ready":
                matching_ready,

            "sketch_feature":
                sketch_visual_features,

            "photo_feature_dimensions":
                photo_feature_dimensions
        }