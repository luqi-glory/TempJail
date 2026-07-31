import torch
import hydra
from omegaconf import DictConfig

class BaseAttack:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def attack(self, image_org: torch.Tensor, image_tgt: torch.Tensor, img_index: int = 0):
        return image_org