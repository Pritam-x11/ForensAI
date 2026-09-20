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

    print(
        "\nInput IDs shape:",
        input_ids.shape
    )

    print(
        "Pixel values shape:",
        pixel_values.shape
    )

    # --------------------------------
    # STEP 1: IMAGE FEATURES
    # --------------------------------

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
        "Image features shape:",
        image_features.shape
    )

    # --------------------------------
    # STEP 2: TEXT EMBEDDINGS
    # --------------------------------

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
        "Text embeddings shape:",
        text_embeddings.shape
    )

    # --------------------------------
    # STEP 3: FIND IMAGE TOKENS
    # --------------------------------

    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<image>"
    )

    print(
        "\nImage token ID:",
        image_token_id
    )

    image_token_positions = np.where(
        input_ids[0] == image_token_id
    )[0]

    print(
        "Number of image token positions:",
        len(image_token_positions)
    )

    if len(image_token_positions) > 0:

        print(
            "First image token position:",
            int(image_token_positions[0])
        )

        print(
            "Last image token position:",
            int(image_token_positions[-1])
        )

    # --------------------------------
    # STEP 4: FEATURE DIMENSION CHECK
    # --------------------------------

    image_dimension = image_features.shape[-1]

    text_dimension = text_embeddings.shape[-1]

    print("\n==============================")
    print("MULTIMODAL INTEGRATION CHECK")
    print("==============================")

    print(
        "Image feature dimension:",
        image_dimension
    )

    print(
        "Text embedding dimension:",
        text_dimension
    )

    if image_dimension != text_dimension:

        print(
            "ERROR: Feature dimensions do not match."
        )

        return

    print(
        "Feature dimensions match."
    )

    # --------------------------------
    # STEP 5: INSPECT IMAGE TOKEN AREA
    # --------------------------------

    if len(image_token_positions) > 0:

        first_position = int(
            image_token_positions[0]
        )

        last_position = int(
            image_token_positions[-1]
        )

        image_token_embeddings = text_embeddings[
            0,
            first_position:last_position + 1,
            :
        ]

        print(
            "\nEmbeddings currently occupying "
            "image-token region:"
        )

        print(
            image_token_embeddings.shape
        )

        print(
            "\nVision encoder produces:"
        )

        print(
            image_features.shape
        )

        print(
            "\nImportant:"
        )

        print(
            "We will NOT replace these positions yet."
        )

        print(
            "This test only verifies where the "
            "processor placed image tokens."
        )

    print(
        "\nMultimodal integration inspection successful!"
    )


if __name__ == "__main__":
    main()