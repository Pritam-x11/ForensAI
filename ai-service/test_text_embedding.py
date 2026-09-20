import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_PATH = (
    r"J:\ForensAI\ai-service\models\smolvlm\onnx"
    r"\embed_tokens_int8.onnx"
)

TOKENIZER_PATH = (
    r"C:\Users\pk993\.cache\huggingface\hub"
    r"\\models--HuggingFaceTB--SmolVLM-256M-Instruct"
)


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct"
    )

    text = "Describe the facial attributes in this image."

    encoded = tokenizer(
        text,
        return_tensors="np"
    )

    input_ids = encoded["input_ids"].astype(
        np.int64
    )

    print("Input IDs shape:", input_ids.shape)
    print("Input IDs:", input_ids)

    print("\nLoading embedding ONNX model...")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    inputs_embeds = session.run(
        None,
        {
            "input_ids": input_ids
        }
    )[0]

    print(
        "Embedding output shape:",
        inputs_embeds.shape
    )

    print(
        "Embedding output type:",
        inputs_embeds.dtype
    )

    print(
        "Text embedding test successful!"
    )


if __name__ == "__main__":
    main()