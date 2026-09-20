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

DECODER_MODEL = os.path.join(
    MODEL_DIR,
    "decoder_model_merged_q4f16.onnx"
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

    input_ids = processed[
        "input_ids"
    ]

    attention_mask = processed[
        "attention_mask"
    ]

    pixel_values = processed[
        "pixel_values"
    ].astype(np.float32)

    pixel_attention_mask = processed[
        "pixel_attention_mask"
    ].astype(bool)

    # --------------------------------
    # VISION ENCODER
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
        "Vision features:",
        image_features.shape
    )

    # --------------------------------
    # TEXT EMBEDDINGS
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
        "Text embeddings:",
        text_embeddings.shape
    )

    # --------------------------------
    # IMAGE TOKEN REPLACEMENT
    # --------------------------------

    image_token_id = (
        processor.tokenizer.convert_tokens_to_ids(
            "<image>"
        )
    )

    image_token_positions = np.where(
        input_ids[0] == image_token_id
    )[0]

    visual_tokens = image_features.reshape(
        -1,
        image_features.shape[-1]
    )

    print(
        "\nImage token positions:",
        len(image_token_positions)
    )

    print(
        "Visual tokens:",
        len(visual_tokens)
    )

    if len(image_token_positions) != len(
        visual_tokens
    ):
        print(
            "ERROR: Image token count mismatch."
        )
        return

    multimodal_embeddings = (
        text_embeddings.copy()
    )

    multimodal_embeddings[
        0,
        image_token_positions,
        :
    ] = visual_tokens

    print(
        "Multimodal embeddings:",
        multimodal_embeddings.shape
    )

    # --------------------------------
    # DECODER
    # --------------------------------

    print("\nLoading Decoder...")

    decoder_session = ort.InferenceSession(
        DECODER_MODEL,
        providers=["CPUExecutionProvider"]
    )

    print("Decoder loaded.")

    print("\nDecoder inputs:")

    for input_info in decoder_session.get_inputs():
        print(
            input_info.name,
            input_info.shape,
            input_info.type
        )

    # --------------------------------
    # FIRST DECODER PASS
    # --------------------------------

    print("\nRunning multimodal decoder...")

    decoder_inputs = {
        "inputs_embeds":
            multimodal_embeddings.astype(np.float32),

        "attention_mask":
            attention_mask.astype(np.int64),

        "position_ids":
            np.arange(
                input_ids.shape[1],
                dtype=np.int64
            ).reshape(1, -1)
    }

    # Empty KV cache for first pass
    for layer in range(30):

        decoder_inputs[
            f"past_key_values.{layer}.key"
        ] = np.empty(
            (1, 3, 0, 64),
            dtype=np.float16
        )

        decoder_inputs[
            f"past_key_values.{layer}.value"
        ] = np.empty(
            (1, 3, 0, 64),
            dtype=np.float16
        )

    outputs = decoder_session.run(
        None,
        decoder_inputs
    )

    logits = outputs[0]

    print(
        "\nLogits shape:",
        logits.shape
    )

    print(
        "Logits dtype:",
        logits.dtype
    )

    # Last token logits
    next_token_logits = logits[:, -1, :]

    next_token_id = int(
        np.argmax(
            next_token_logits,
            axis=-1
        )[0]
    )

    next_token = processor.tokenizer.decode(
        [next_token_id]
    )

    print(
        "\nNext token ID:",
        next_token_id
    )

    print(
        "Next token:",
        repr(next_token)
    )

    print(
        "\nMultimodal decoder first-pass test completed!"
    )


if __name__ == "__main__":
    main()