from insightface.app import FaceAnalysis


class ArcFaceService:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

    def get_embedding(self, aligned_face):

        if aligned_face is None:
            return None

        embedding = self.app.models["recognition"].get_feat(
            aligned_face
        )

        return embedding[0]