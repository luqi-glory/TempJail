# from .get_gpt4_image_model import get_gpt4_image_model
# from .Blip2 import Blip2VisionModel, Blip2PredictModel
# from .InstructBlip import InstructBlipVisionModel, InstructBlipPredictModel
from .FeatureExtractors import *
from typing import Dict

# Mapping from backbone names to model classes
BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}