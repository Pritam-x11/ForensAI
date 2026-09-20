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

MAX_NEW_TOKENS = 20


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
                    "text": (
                        
                            "Are the person's eyes visible in the image? "
    "Answer only YES or NO."
                            )
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
            "pixel_attention_mask": pixel_attention_mask
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
            "input_ids": input_ids.astype(np.int64)
        }
    )

    text_embeddings = text_outputs[0]

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

    if len(image_token_positions) != len(
        visual_tokens
    ):
        print(
            "ERROR: image token count mismatch."
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

    # --------------------------------
    # FIRST PASS
    # --------------------------------

    print("\nRunning first decoder pass...")

    sequence_length = input_ids.shape[1]

    decoder_inputs = {
        "inputs_embeds":
            multimodal_embeddings.astype(np.float32),

        "attention_mask":
            attention_mask.astype(np.int64),

        "position_ids":
            np.arange(
                sequence_length,
                dtype=np.int64
            ).reshape(1, -1)
    }

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

    next_token_id = int(
        np.argmax(
            logits[:, -1, :],
            axis=-1
        )[0]
    )

    generated_ids = [
        next_token_id
    ]

    print(
        "First token:",
        repr(
            processor.tokenizer.decode(
                [next_token_id]
            )
        )
    )

    # --------------------------------
    # GET KV CACHE
    # --------------------------------

    past_key_values = {}

    for layer in range(30):

        past_key_values[
            f"past_key_values.{layer}.key"
        ] = outputs[
            1 + layer * 2
        ]

        past_key_values[
            f"past_key_values.{layer}.value"
        ] = outputs[
            2 + layer * 2
        ]

    # --------------------------------
    # AUTOREGRESSIVE GENERATION
    # --------------------------------

    current_token_id = next_token_id

    for step in range(
        1,
        MAX_NEW_TOKENS
    ):

        token_ids = np.array(
            [[current_token_id]],
            dtype=np.int64
        )

        token_output = embed_session.run(
            None,
            {
                "input_ids": token_ids
            }
        )

        token_embedding = token_output[0]

        past_length = past_key_values[
            "past_key_values.0.key"
        ].shape[2]

        decoder_inputs = {
            "inputs_embeds":
                token_embedding.astype(
                    np.float32
                ),

            "attention_mask":
                np.ones(
                    (1, past_length + 1),
                    dtype=np.int64
                ),

            "position_ids":
                np.array(
                    [[past_length]],
                    dtype=np.int64
                )
        }

        decoder_inputs.update(
            past_key_values
        )

        outputs = decoder_session.run(
            None,
            decoder_inputs
        )

        logits = outputs[0]

        current_token_id = int(
            np.argmax(
                logits[:, -1, :],
                axis=-1
            )[0]
        )

        generated_ids.append(
            current_token_id
        )

        token_text = (
            processor.tokenizer.decode(
                [current_token_id]
            )
        )

        print(
            f"Step {step + 1}:",
            repr(token_text)
        )

        # --------------------------------
        # UPDATE KV CACHE
        # --------------------------------

        new_past_key_values = {}

        for layer in range(30):

            new_past_key_values[
                f"past_key_values.{layer}.key"
            ] = outputs[
                1 + layer * 2
            ]

            new_past_key_values[
                f"past_key_values.{layer}.value"
            ] = outputs[
                2 + layer * 2
            ]

        past_key_values = (
            new_past_key_values
        )

        # --------------------------------
        # STOP ON EOS
        # --------------------------------

        eos_token_id = (
            processor.tokenizer.eos_token_id
        )

        end_of_utterance_id = (
            processor.tokenizer.convert_tokens_to_ids(
                "<end_of_utterance>"
            )
        )

        if current_token_id in [
            eos_token_id,
            end_of_utterance_id
        ]:
            break

    # --------------------------------
    # FINAL TEXT
    # --------------------------------

    generated_text = (
        processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
    )

    print("\n==============================")
    print("GENERATED RESPONSE")
    print("==============================")

    print(generated_text)

    print(
        "\nMultimodal generation test completed!"
    )


if __name__ == "__main__":
    main()