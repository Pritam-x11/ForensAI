import cv2

from services.vlm_attributes_service import (
    VLMAttributesService
)


IMAGE_PATH = r"gallery\candidates\candidate_2.webp"


def main():

    print("Loading image...")

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        print("ERROR: Image could not be loaded.")
        return

    print(
        "Image loaded:",
        image.shape
    )

    print(
        "\nCreating VLM service..."
    )

    vlm_service = (
        VLMAttributesService()
    )

    print(
        "Running VLM inference..."
    )

    result = (
        vlm_service.extract_attributes(
            image
        )
    )

    print(
        "\n=============================="
    )

    print(
        "VLM SERVICE RESULT"
    )

    print(
        "=============================="
    )

    print(result)


if __name__ == "__main__":
    main()