import supervision as sv


class VideoTrackingService:

    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update_tracks(self, detections):
        tracked_detections = self.tracker.update_with_detections(
            detections
        )

        return tracked_detections