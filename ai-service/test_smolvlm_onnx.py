import os
import cv2
import numpy as np
import onnxruntime as ort


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


def main():

    print("Loading ONNX models...")

    vision_session = ort.InferenceSession(
        VISION_MODEL,
        providers=["CPUExecutionProvider"]
    )

    embed_session = ort.InferenceSession(
        EMBED_MODEL,
        providers=["CPUExecutionProvider"]
    )

    decoder_session = ort.InferenceSession(
        DECODER_MODEL,
        providers=["CPUExecutionProvider"]
    )

    print("Vision Encoder loaded")
    print("Embedding model loaded")
    print("Decoder loaded")

    print("\nAll ONNX models loaded successfully.")


if __name__ == "__main__":
    main()