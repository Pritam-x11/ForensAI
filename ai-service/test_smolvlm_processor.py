from transformers import AutoProcessor
from PIL import Image


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

IMAGE_PATH = r"gallery\candidates\candidate_2.webp"


def main():

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print("Processor loaded.")

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

    print("\nApplying SmolVLM chat template...")
    processed = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="np"
)
    print("\nProcessed keys:")
    print(processed.keys())

    print("\nInput IDs shape:")
    print(processed["input_ids"].shape)

    print("\nAttention mask shape:")
    print(processed["attention_mask"].shape)

    if "pixel_values" in processed:
        print("\nPixel values shape:")
        print(processed["pixel_values"].shape)

    if "pixel_attention_mask" in processed:
        print("\nPixel attention mask shape:")
        print(
            processed[
                "pixel_attention_mask"
            ].shape
        )

    print("\nDecoded input:")
    print(
        processor.tokenizer.decode(
            processed["input_ids"][0]
        )
    )

    print("\nProcessor inspection successful!")


if __name__ == "__main__":
    main()