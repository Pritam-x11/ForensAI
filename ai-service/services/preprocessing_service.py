import cv2


class PreprocessingService:

    def preprocess(self, image):

        # Resize image if it is too large
        max_size = 1600

        height, width = image.shape[:2]

        if max(height, width) > max_size:
            scale = max_size / max(height, width)

            new_width = int(width * scale)
            new_height = int(height * scale)

            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

        # Convert image to standard BGR format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image