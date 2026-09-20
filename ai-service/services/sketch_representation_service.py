class SketchRepresentationService:

    def build_representation(
        self,
        image,
        attributes,
        visual_features=None
    ):

        return {
            "sketch_image": image,
            "attributes": attributes,
            "visual_features": visual_features
        }