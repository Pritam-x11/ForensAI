import os
import onnxruntime as ort

model_path = r"J:\ForensAI\ai-service\models\smolvlm\onnx"

files = [
    "vision_encoder_q4f16.onnx",
    "embed_tokens_int8.onnx",
    "decoder_model_merged_q4f16.onnx"
]

for file_name in files:

    print("\n==============================")
    print(file_name)
    print("==============================")

    file_path = os.path.join(model_path, file_name)

    session = ort.InferenceSession(
        file_path,
        providers=["CPUExecutionProvider"]
    )

    print("\nINPUTS:")

    for item in session.get_inputs():
        print(
            "Name:",
            item.name,
            "| Shape:",
            item.shape,
            "| Type:",
            item.type
        )

    print("\nOUTPUTS:")

    for item in session.get_outputs():
        print(
            "Name:",
            item.name,
            "| Shape:",
            item.shape,
            "| Type:",
            item.type
        )