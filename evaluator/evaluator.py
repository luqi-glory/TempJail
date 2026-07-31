import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Tuple
from transformers import CLIPProcessor, CLIPModel

class CLIPScorer:
    def __init__(self, device):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device 
        self.model.to(self.device)
        self.model.eval()

    # def compute_clip_score(self, text1: str, text2: str) -> float:
    #     inputs1 = self.processor(text=[text1], return_tensors="pt", padding=True, truncation=True)
    #     inputs2 = self.processor(text=[text2], return_tensors="pt", padding=True, truncation=True)
    #     inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
    #     inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}

    #     with torch.no_grad():
    #         text_features1 = self.model.get_text_features(**inputs1)  # [1, embed_dim]
    #         text_features2 = self.model.get_text_features(**inputs2)  # [1, embed_dim]

    #     text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
    #     text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)

    #     clip_score = (text_features1 @ text_features2.T).item()

    #     return clip_score
    
    def compute_clip_score(self, text1: str, text2: str) -> float:
        inputs1 = self.processor(text=[text1], return_tensors="pt", padding=True, truncation=True)
        inputs2 = self.processor(text=[text2], return_tensors="pt", padding=True, truncation=True)
        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}

        with torch.no_grad():
            outputs1 = self.model.text_model(
                input_ids=inputs1["input_ids"],
                attention_mask=inputs1["attention_mask"]
            )
            outputs2 = self.model.text_model(
                input_ids=inputs2["input_ids"],
                attention_mask=inputs2["attention_mask"]
            )

        text_features1 = outputs1.pooler_output
        text_features2 = outputs2.pooler_output

        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
        text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)

        clip_score = (text_features1 @ text_features2.T).item()
        return clip_score

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.models = {}

        if "blip2" in cfg.evaluate.open_source_models:
            self.models["blip2"] = {
                "processor": Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True),
                "model": Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
            }

        if "minigpt4" in cfg.evaluate.open_source_models:
            # Placeholder for actual integration (MiniGPT4 typically uses a separate repo)
            self.models["minigpt4"] = None

        if "unidiffuser" in cfg.evaluate.open_source_models:
            # Placeholder for actual integration
            self.models["unidiffuser"] = None
        
        # Metrics
        if "clip_score" in cfg.evaluate.metrics:
            self.clip_scorer = CLIPScorer(self.device)

    def generate_blip2(self, image_tensor: torch.Tensor) -> str:
        image_tensor = image_tensor * 255.0
        model_data = self.models["blip2"]
        inputs = model_data["processor"](images=image_tensor, return_tensors="pt").to(self.device)
        model_data["model"].eval()
        with torch.no_grad():
            generated_ids = model_data["model"].generate(**inputs, max_new_tokens=50)
        return model_data["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]

    def generate_llava(self, image_tensor: torch.Tensor) -> str:
        model_data = self.models["llava"]
        inputs = model_data["processor"](images=image_tensor, return_tensors="pt").to(self.device)
        generated_ids = model_data["model"].generate(**inputs, max_new_tokens=50)
        return model_data["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]

    def generate_minigpt4(self, image_tensor: torch.Tensor) -> str:
        raise NotImplementedError("MiniGPT4 model integration requires external repo.")

    def generate_unidiffuser(self, image_tensor: torch.Tensor) -> str:
        raise NotImplementedError("Unidiffuser model integration requires external repo.")

    def compute_clipscore(self, text1: str, text2: str) -> float:
        return self.clip_scorer.compute_clip_score(text1, text2)

    def evaluate(self, adv_image: torch.Tensor, ori_image: torch.Tensor, target_text: str) -> Tuple[str, float]:
        metrics = {}
        metrics["target_text"] = target_text
        adv_image = adv_image.to(self.device)
        ori_image = ori_image.to(self.device)
        perturbation = adv_image - ori_image 
        n_pixels = torch.numel(perturbation)
        l1_norm = torch.abs(perturbation).sum() / n_pixels
        l2_norm = torch.sqrt(torch.sum(perturbation**2) / n_pixels)
        l_inf_norm = torch.max(torch.abs(perturbation))
        metrics["L_1"] = l1_norm.item()
        metrics["L_2"] = l2_norm.item()
        metrics["L_inf"] = l_inf_norm.item()
        for model in self.cfg.evaluate.open_source_models:
            if model == "blip2":
                description = self.generate_blip2(adv_image)
            elif model == "llava":
                description = self.generate_llava(adv_image)
            metrics[f"{model}_output"] = description

            for eva in self.cfg.evaluate.metrics:
                if eva == "clip_score":
                    score = self.compute_clipscore(description, target_text)
                    metrics[f"{model}_clip_score"] = score
        return metrics