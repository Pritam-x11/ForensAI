import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv3Service:

    def __init__(self):

        self.model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id
        )

        self.model = AutoModel.from_pretrained(
            self.model_id
        )

        self.model.eval()

    def get_features(self, image):

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        features = outputs.last_hidden_state[:, 0, :]

        return features[0].numpy()