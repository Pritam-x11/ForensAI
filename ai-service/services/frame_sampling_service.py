import cv2


class FrameSamplingService:

    def sample_frames(self, video_path, sample_rate=5):

        video = cv2.VideoCapture(video_path)

        frames = []
        frame_count = 0

        while True:

            success, frame = video.read()

            if not success:
                break

            if frame_count % sample_rate == 0:
                frames.append(frame)

            frame_count += 1

        video.release()

        return frames