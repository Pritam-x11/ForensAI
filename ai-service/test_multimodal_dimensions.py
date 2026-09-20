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

EMBED_MODEL = os.path.join(
    MODEL_DIR,
    "embed_tokens_int8.onnx"
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

    # -----------------------------
    # IMAGE SIDE
    # -----------------------------

    pixel_values = processed[
        "pixel_values"
    ].astype(np.float32)

    pixel_attention_mask = processed[
        "pixel_attention_mask"
    ].astype(bool)

    print(
        "\nPixel values:",
        pixel_values.shape
    )

    print(
        "Pixel attention mask:",
        pixel_attention_mask.shape
    )

    print("\nLoading Vision Encoder...")

    vision_session = ort.InferenceSession(
        VISION_MODEL,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    vision_outputs = vision_session.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask":
                pixel_attention_mask
        }
    )

    image_features = vision_outputs[0]

    print(
        "Image features:",
        image_features.shape
    )

    # -----------------------------
    # TEXT SIDE
    # -----------------------------

    input_ids = processed[
        "input_ids"
    ]

    print(
        "\nInput IDs:",
        input_ids.shape
    )

    print("\nLoading Embedding model...")

    embed_session = ort.InferenceSession(
        EMBED_MODEL,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    text_outputs = embed_session.run(
        None,
        {
            "input_ids":
                input_ids.astype(np.int64)
        }
    )

    text_embeddings = text_outputs[0]

    print(
        "Text embeddings:",
        text_embeddings.shape
    )

    # -----------------------------
    # DIMENSION CHECK
    # -----------------------------

    image_dimension = image_features.shape[-1]

    text_dimension = text_embeddings.shape[-1]

    print("\n==============================")
    print("MULTIMODAL DIMENSION CHECK")
    print("==============================")

    print(
        "Image feature dimension:",
        image_dimension
    )

    print(
        "Text embedding dimension:",
        text_dimension
    )

    if image_dimension == text_dimension:

        print(
            "\nSUCCESS:"
            " Image and text feature dimensions match."
        )

    else:

        print(
            "\nERROR:"
            " Image and text feature dimensions do not match."
        )

    print(
        "\nMultimodal dimension test completed!"
    )


if __name__ == "__main__":
    main()