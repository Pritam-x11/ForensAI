import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_DIR = r"J:\ForensAI\ai-service\models\smolvlm\onnx"

EMBED_MODEL = os.path.join(
    MODEL_DIR,
    "embed_tokens_int8.onnx"
)

DECODER_MODEL = os.path.join(
    MODEL_DIR,
    "decoder_model_merged_q4f16.onnx"
)


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct"
    )

    text = "Describe the facial attributes."

    encoded = tokenizer(
        text,
        return_tensors="np"
    )

    input_ids = encoded["input_ids"].astype(
        np.int64
    )

    print("Input IDs shape:", input_ids.shape)

    print("\nLoading embedding model...")

    embed_session = ort.InferenceSession(
        EMBED_MODEL,
        providers=["CPUExecutionProvider"]
    )

    inputs_embeds = embed_session.run(
        None,
        {
            "input_ids": input_ids
        }
    )[0]

    print(
        "Embeddings shape:",
        inputs_embeds.shape
    )

    print("\nLoading decoder...")

    decoder_session = ort.InferenceSession(
        DECODER_MODEL,
        providers=["CPUExecutionProvider"]
    )

    batch_size = inputs_embeds.shape[0]
    sequence_length = inputs_embeds.shape[1]

    attention_mask = np.ones(
        (batch_size, sequence_length),
        dtype=np.int64
    )

    position_ids = np.arange(
        sequence_length,
        dtype=np.int64
    ).reshape(1, -1)

    decoder_inputs = {
        "inputs_embeds": inputs_embeds.astype(
            np.float32
        ),
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }

    # Empty KV cache for the first decoder pass.
    for i in range(30):

        decoder_inputs[
            f"past_key_values.{i}.key"
        ] = np.zeros(
            (batch_size, 3, 0, 64),
            dtype=np.float16
        )

        decoder_inputs[
            f"past_key_values.{i}.value"
        ] = np.zeros(
            (batch_size, 3, 0, 64),
            dtype=np.float16
        )

    print("Running decoder...")

    outputs = decoder_session.run(
        None,
        decoder_inputs
    )

    logits = outputs[0]

    print(
        "Logits shape:",
        logits.shape
    )

    print(
        "Logits type:",
        logits.dtype
    )

    print(
        "Decoder test successful!"
    )


if __name__ == "__main__":
    main()