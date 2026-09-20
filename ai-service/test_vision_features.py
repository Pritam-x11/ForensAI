import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

MODEL_DIR = r"J:\ForensAI\ai-service\models\smolvlm\onnx"

VISION_MODEL = os.path.join(
    MODEL_DIR,
    "vision_encoder_q4f16.onnx"
)

IMAGE_PATH = r"gallery\candidates\candidate_2.webp"


def main():

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": "Describe the facial attributes."
                }
            ]
        }
    ]

    print("Preparing multimodal input...")

    processed = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np"
    )

    pixel_values = processed[
        "pixel_values"
    ].astype(np.float32)

    pixel_attention_mask = processed[
        "pixel_attention_mask"
    ].astype(bool)

    print(
        "Pixel values:",
        pixel_values.shape
    )

    print(
        "Pixel attention mask:",
        pixel_attention_mask.shape
    )

    print("\nLoading Vision Encoder...")

    session = ort.InferenceSession(
        VISION_MODEL,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    print("Vision Encoder loaded.")

    print("\nGenerating image features...")

    outputs = session.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask":
                pixel_attention_mask
        }
    )

    image_features = outputs[0]

    print(
        "Image features shape:",
        image_features.shape
    )

    print(
        "Image features dtype:",
        image_features.dtype
    )

    print(
        "Image features minimum:",
        float(image_features.min())
    )

    print(
        "Image features maximum:",
        float(image_features.max())
    )

    print(
        "Image features mean:",
        float(image_features.mean())
    )

    print(
        "Image features contain NaN:",
        bool(np.isnan(image_features).any())
    )

    print(
        "Image features contain Inf:",
        bool(np.isinf(image_features).any())
    )

    print(
        "\nVision feature extraction test successful!"
    )


if __name__ == "__main__":
    main()