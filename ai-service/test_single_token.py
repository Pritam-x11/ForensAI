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

    embed_session = ort.InferenceSession(
        EMBED_MODEL,
        providers=["CPUExecutionProvider"]
    )

    inputs_embeds = embed_session.run(
        None,
        {"input_ids": input_ids}
    )[0]

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
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }

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

    outputs = decoder_session.run(
        None,
        decoder_inputs
    )

    logits = outputs[0]

    # Last position's logits
    next_token_logits = logits[:, -1, :]

    # Select token with highest probability
    next_token_id = int(
        np.argmax(next_token_logits, axis=-1)[0]
    )

    next_token = tokenizer.decode(
        [next_token_id]
    )

    print("Next token ID:", next_token_id)
    print("Next token:", repr(next_token))

    print(
        "\nSingle-token generation test successful!"
    )


if __name__ == "__main__":
    main()