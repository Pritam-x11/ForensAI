import os
import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = (
    r"J:\ForensAI\ai-service\models\smolvlm\onnx"
    r"\vision_encoder_q4f16.onnx"
)


def main():

    image_path = r"gallery\candidates\candidate_2.webp"

    image = cv2.imread(image_path)

    if image is None:
        print("Image could not be loaded.")
        return

    print("Image loaded:", image.shape)

    # Convert BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the model's required 512x512 input
    image = cv2.resize(
        image,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    # Convert uint8 -> float32
    image = image.astype(np.float32) / 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch and image dimensions
    pixel_values = image[
        np.newaxis,
        np.newaxis,
        :
    ]

    # Attention mask
    pixel_attention_mask = np.ones(
        (1, 1, 512, 512),
        dtype=bool
    )

    print("pixel_values shape:", pixel_values.shape)
    print(
        "pixel_attention_mask shape:",
        pixel_attention_mask.shape
    )

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    outputs = session.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask
        }
    )

    image_features = outputs[0]

    print(
        "Vision encoder output shape:",
        image_features.shape
    )

    print(
        "Vision encoder test successful!"
    )


if __name__ == "__main__":
    main()