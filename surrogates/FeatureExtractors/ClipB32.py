import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms


class ClipB32FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipB32FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    # def forward(self, x):
    #     # x = torch.clamp(x, min=0, max=1)
    #     inputs = dict(pixel_values=self.normalizer(x))
    #     image_features = self.model.get_image_features(**inputs)
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     return image_features

    def forward(self, x):
        pixel_values = self.normalizer(x).to(x.device)
        outputs = self.model.get_image_features(pixel_values=pixel_values)

        if isinstance(outputs, torch.Tensor):
            image_features = outputs
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            image_features = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            image_features = outputs.last_hidden_state[:, 0, :]
        else:
            raise TypeError(f"Unexpected output type: {type(outputs)}")

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
