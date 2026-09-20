from insightface.app import FaceAnalysis


class SCRFDService:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

    def detect_faces(self, image):
        faces = self.app.get(image)
        return faces