"""Shared utilities for adversarial attack and text generation models."""

import os
import json
import yaml
import shutil
import hashlib
import base64
import random
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union
from omegaconf import OmegaConf
import wandb
from omegaconf import DictConfig
from dataclasses import asdict, is_dataclass

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureExtractor,
    EnsembleFeatureLoss,
    BACKBONE_MAP,
)


def load_api_keys() -> Dict[str, str]:
    """Load API keys from the api_keys file.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys for different models
        
    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    for ext in ['yaml', 'yml', 'json']:
        file_path = f'api_keys.{ext}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                if ext in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    
    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_key(model_name: str) -> str:
    """Get API key for specified model.
    
    Args:
        model_name: Name of the model to get API key for
        
    Returns:
        str: API key for the specified model
        
    Raises:
        KeyError: If API key for model is not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )
    return api_keys[model_name]


def hash_training_config(cfg) -> str:
    """Create a deterministic hash from any Hydra config (including dataclass-based)."""
    # Convert OmegaConf or dataclass to regular nested dict
    if OmegaConf.is_config(cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    elif is_dataclass(cfg):
        cfg_dict = asdict(cfg)
    else:
        raise TypeError("Unsupported config type")

    # Convert to JSON string with sorted keys
    json_str = json.dumps(cfg_dict, sort_keys=True, indent=None, separators=(",", ":"))
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: DictConfig) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        cfg: Configuration object containing wandb settings
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name + "_" + hash_training_config(cfg),
        config=config_dict,
        tags=cfg.wandb.tags,
        settings=wandb.Settings(console="off", silent=True),
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def clear_and_ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_output_paths(cfg: DictConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.
    
    Args:
        cfg: Configuration object
        config_hash: Hash of training config
        
    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    return {
        'output_dir': os.path.join(cfg.data.output, "img", config_hash),
        'desc_output_dir': os.path.join(cfg.data.output, "description", config_hash)
    } 

def set_environment(seed: int = 2023)-> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def to_tensor(pic: Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch tensor.
    
    Args:
        pic: PIL Image to convert
        
    Returns:
        torch.Tensor: Converted tensor
    """
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Load image from path and preprocess it.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = to_tensor(img)
        return img_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    

def get_models(cfg: DictConfig):
    """Get models based on configuration.

    Args:
        cfg: Configuration object containing model settings

    Returns:
        Tuple of (feature_extractor, list of models)

    Raises:
        ValueError: If ensemble=False but multiple backbones specified
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor(models)
    else:
        ensemble_extractor = models[0]  # Use single model directly

    return ensemble_extractor, models



def get_ensemble_loss(models: List[nn.Module]):
    ensemble_loss = EnsembleFeatureLoss(models)
    return ensemble_loss



# Dataset with image paths
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)
    
    
def log_metrics(pbar, metrics, img_index, epoch=None, log_to_wandb=False):
    """
    Log metrics to progress bar and wandb.

    Args:
        pbar: tqdm progress bar to update
        metrics: Dictionary of metrics to log
        img_index: Index of the image (for wandb logging)
        epoch: Optional epoch number for logging
    """
    # Format metrics for progress bar
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)

    # Prepare wandb metrics with image index
    wandb_metrics = {f"img{img_index:02d}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        wandb_metrics["epoch"] = epoch

    # Log to wandb
    if log_to_wandb:
        wandb.log(wandb_metrics)


import os
import warnings
import logging
import contextlib

def setup_quiet_env():
    os.environ["WANDB_SILENT"] = "true"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore",
        message="The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default.*"
    )

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("wandb").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def silent_call(fn, *args, **kwargs):
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            return fn(*args, **kwargs)