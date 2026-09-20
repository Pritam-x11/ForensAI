import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor


class VLMAttributesService:

    def __init__(self):

        self.model_id = (
            "HuggingFaceTB/SmolVLM-256M-Instruct"
        )

        self.model_dir = (
            r"J:\ForensAI\ai-service\models\smolvlm\onnx"
        )

        self.vision_model = os.path.join(
            self.model_dir,
            "vision_encoder_q4f16.onnx"
        )

        self.embed_model = os.path.join(
            self.model_dir,
            "embed_tokens_int8.onnx"
        )

        self.decoder_model = os.path.join(
            self.model_dir,
            "decoder_model_merged_q4f16.onnx"
        )

        self.processor = None
        self.vision_session = None
        self.embed_session = None
        self.decoder_session = None

        self.model_status = (
            "VLM model not loaded"
        )

    def load_model(self):

        if (
            self.processor is not None
            and self.vision_session is not None
            and self.embed_session is not None
            and self.decoder_session is not None
        ):
            return True

        try:

            self.processor = AutoProcessor.from_pretrained(
                self.model_id
            )

            self.processor.tokenizer.pad_token_id = 2

            self.vision_session = (
                ort.InferenceSession(
                    self.vision_model,
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )
            )

            self.embed_session = (
                ort.InferenceSession(
                    self.embed_model,
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )
            )

            self.decoder_session = (
                ort.InferenceSession(
                    self.decoder_model,
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )
            )

            self.model_status = (
                "VLM model loaded"
            )

            return True

        except Exception as error:

            self.model_status = (
                f"VLM loading failed: {error}"
            )

            return False

    def extract_attributes(
        self,
        image,
        prompt=None
    ):

        if image is None:

            return {
                "attributes": None,
                "confidence": None,
                "model_status":
                    "Image unavailable"
            }

        model_loaded = self.load_model()

        if not model_loaded:

            return {
                "attributes": None,
                "confidence": None,
                "model_status":
                    self.model_status
            }

        # --------------------------------
        # OPENCV BGR → RGB
        # --------------------------------

        image_rgb = (
            image[:, :, ::-1]
        )

        pil_image = Image.fromarray(
            image_rgb
        )

        # --------------------------------
        # CONVERSATION
        # --------------------------------

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image
                    },
                    {
                        "type": "text",
                        "text": (
                            prompt
                            or
                            "Are the person's eyes visible in the image? Answer only YES or NO."
                        )
                    }
                ]
            }
        ]

        # --------------------------------
        # PROCESS IMAGE
        # --------------------------------

        processed = (
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="np"
            )
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

        pixel_attention_mask = (
            processed[
                "pixel_attention_mask"
            ].astype(bool)
        )

        # --------------------------------
        # VISION ENCODER
        # --------------------------------

        vision_outputs = (
            self.vision_session.run(
                None,
                {
                    "pixel_values":
                        pixel_values,

                    "pixel_attention_mask":
                        pixel_attention_mask
                }
            )
        )

        image_features = (
            vision_outputs[0]
        )

        # --------------------------------
        # TEXT EMBEDDINGS
        # --------------------------------

        text_outputs = (
            self.embed_session.run(
                None,
                {
                    "input_ids":
                        input_ids.astype(
                            np.int64
                        )
                }
            )
        )

        text_embeddings = (
            text_outputs[0]
        )

        # --------------------------------
        # IMAGE TOKEN REPLACEMENT
        # --------------------------------

        image_token_id = (
            self.processor.tokenizer
            .convert_tokens_to_ids(
                "<image>"
            )
        )

        image_token_positions = (
            np.where(
                input_ids[0]
                == image_token_id
            )[0]
        )

        visual_tokens = (
            image_features.reshape(
                -1,
                image_features.shape[-1]
            )
        )

        if (
            len(image_token_positions)
            != len(visual_tokens)
        ):

            return {
                "attributes": None,
                "confidence": None,
                "model_status":
                    "Image token mismatch"
            }

        multimodal_embeddings = (
            text_embeddings.copy()
        )

        multimodal_embeddings[
            0,
            image_token_positions,
            :
        ] = visual_tokens

        # --------------------------------
        # FIRST DECODER PASS
        # --------------------------------

        sequence_length = (
            input_ids.shape[1]
        )

        decoder_inputs = {

            "inputs_embeds":
                multimodal_embeddings.astype(
                    np.float32
                ),

            "attention_mask":
                attention_mask.astype(
                    np.int64
                ),

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

        outputs = (
            self.decoder_session.run(
                None,
                decoder_inputs
            )
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

        # --------------------------------
        # KV CACHE
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
        # GENERATE RESPONSE
        # --------------------------------

        current_token_id = (
            next_token_id
        )

        max_new_tokens = 10

        for _ in range(
            1,
            max_new_tokens
        ):

            token_ids = np.array(
                [[current_token_id]],
                dtype=np.int64
            )

            token_output = (
                self.embed_session.run(
                    None,
                    {
                        "input_ids":
                            token_ids
                    }
                )
            )

            token_embedding = (
                token_output[0]
            )

            past_length = (
                past_key_values[
                    "past_key_values.0.key"
                ].shape[2]
            )

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

            outputs = (
                self.decoder_session.run(
                    None,
                    decoder_inputs
                )
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

            eos_token_id = (
                self.processor.tokenizer
                .eos_token_id
            )

            end_token_id = (
                self.processor.tokenizer
                .convert_tokens_to_ids(
                    "<end_of_utterance>"
                )
            )

            if current_token_id in [
                eos_token_id,
                end_token_id
            ]:
                break

        # --------------------------------
        # DECODE RESPONSE
        # --------------------------------

        generated_text = (
            self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()
        )

        # --------------------------------
        # FINAL RESULT
        # --------------------------------

        return {
            "attributes": {
                "response": generated_text
            },
            "confidence": None,
            "model_status":
                "VLM inference successful"
        }