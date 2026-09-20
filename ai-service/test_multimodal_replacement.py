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

    input_ids = processed["input_ids"]

    pixel_values = processed[
        "pixel_values"
    ].astype(np.float32)

    pixel_attention_mask = processed[
        "pixel_attention_mask"
    ].astype(bool)

    # -----------------------------
    # IMAGE FEATURES
    # -----------------------------

    print("\nLoading Vision Encoder...")

    vision_session = ort.InferenceSession(
        VISION_MODEL,
        providers=["CPUExecutionProvider"]
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
        "Vision features:",
        image_features.shape
    )

    # -----------------------------
    # TEXT EMBEDDINGS
    # -----------------------------

    print("\nLoading Embedding model...")

    embed_session = ort.InferenceSession(
        EMBED_MODEL,
        providers=["CPUExecutionProvider"]
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
    # FIND IMAGE TOKENS
    # -----------------------------

    image_token_id = (
        processor.tokenizer.convert_tokens_to_ids(
            "<image>"
        )
    )

    image_token_mask = (
        input_ids[0] == image_token_id
    )

    image_token_positions = np.where(
        image_token_mask
    )[0]
    
    print(
        "\nImage token count:",
        len(image_token_positions)
    )

    # Flatten visual tokens:
    # (17, 64, 576)
    #        ↓
    # (1088, 576)

    visual_tokens = image_features.reshape(
        -1,
        image_features.shape[-1]
    )

    print(
        "Flattened visual tokens:",
        visual_tokens.shape
    )

    # -----------------------------
    # CHECK TOKEN COUNT
    # -----------------------------

    if len(image_token_positions) != len(
        visual_tokens
    ):

        print(
            "\nERROR:"
            " Image token count and visual"
            " token count do not match."
        )

        return

    print(
        "\nImage token count matches"
        " visual token count."
    )

    # -----------------------------
    # REPLACE IMAGE TOKEN EMBEDDINGS
    # -----------------------------

    multimodal_embeddings = (
        text_embeddings.copy()
    )

    multimodal_embeddings[
        0,
        image_token_positions,
        :
    ] = visual_tokens

    # -----------------------------
    # FINAL CHECK
    # -----------------------------

    print("\n==============================")
    print("MULTIMODAL EMBEDDING RESULT")
    print("==============================")

    print(
        "Final embeddings shape:",
        multimodal_embeddings.shape
    )

    print(
        "Final embeddings dtype:",
        multimodal_embeddings.dtype
    )

    print(
        "Contains NaN:",
        bool(
            np.isnan(
                multimodal_embeddings
            ).any()
        )
    )

    print(
        "Contains Inf:",
        bool(
            np.isinf(
                multimodal_embeddings
            ).any()
        )
    )

    print(
        "\nMultimodal embedding replacement"
        " test successful!"
    )


if __name__ == "__main__":
    main()