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

    prompt = "Describe the facial attributes."

    encoded = tokenizer(
        prompt,
        return_tensors="np"
    )

    input_ids = encoded["input_ids"].astype(
        np.int64
    )

    embed_session = ort.InferenceSession(
        EMBED_MODEL,
        providers=["CPUExecutionProvider"]
    )

    decoder_session = ort.InferenceSession(
        DECODER_MODEL,
        providers=["CPUExecutionProvider"]
    )

    generated_ids = input_ids.tolist()[0]

    # Generate maximum 10 new tokens
    for step in range(10):

        current_ids = np.array(
            [generated_ids],
            dtype=np.int64
        )

        inputs_embeds = embed_session.run(
            None,
            {
                "input_ids": current_ids
            }
        )[0]

        sequence_length = inputs_embeds.shape[1]

        attention_mask = np.ones(
            (1, sequence_length),
            dtype=np.int64
        )

        position_ids = np.arange(
            sequence_length,
            dtype=np.int64
        ).reshape(1, -1)

        decoder_inputs = {
            "inputs_embeds":
                inputs_embeds.astype(np.float32),

            "attention_mask":
                attention_mask,

            "position_ids":
                position_ids
        }

        # Empty KV cache for this standalone
        # generation test.
        for i in range(30):

            decoder_inputs[
                f"past_key_values.{i}.key"
            ] = np.zeros(
                (1, 3, 0, 64),
                dtype=np.float16
            )

            decoder_inputs[
                f"past_key_values.{i}.value"
            ] = np.zeros(
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

        generated_ids.append(
            next_token_id
        )

        token_text = tokenizer.decode(
            [next_token_id]
        )

        print(
            f"Step {step + 1}: "
            f"ID={next_token_id} "
            f"Token={token_text!r}"
        )

        # Stop if EOS token is generated
        if (
            tokenizer.eos_token_id is not None
            and next_token_id ==
            tokenizer.eos_token_id
        ):
            print("EOS token generated.")
            break

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    print("\nGenerated text:")
    print(generated_text)

    print(
        "\nMulti-token generation test completed!"
    )


if __name__ == "__main__":
    main()