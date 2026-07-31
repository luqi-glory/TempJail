import os
import json
import hashlib
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image
import hydra
from omegaconf import DictConfig
import os
from functools import partial
from typing import List, Dict, Optional
from torch import nn
from pytorch_lightning import seed_everything
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
import itertools

from utils import hash_training_config, setup_wandb, ensure_dir, to_tensor, set_environment, get_models, get_ensemble_loss, ImageFolderWithPaths, clear_and_ensure_dir, setup_quiet_env
setup_quiet_env()

# from attacks import ATTACKS, Mattack, AgdAttack
from attacks import ATTACKS
from evaluator import Evaluator
from pprint import pprint
from collections import defaultdict


@hydra.main(version_base=None, config_path="config", config_name="tempjail")
def main(cfg: DictConfig):
    set_environment(cfg.seed)
    
    # Get config hash for output directory
    config_hash = hash_training_config(cfg)

    # Initialize wandb using shared utility
    setup_wandb(cfg)
    wandb.log({"cfg_hash": config_hash})
    
    # Define metrics relationship for logging multiple images
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    data_loader_origin = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )


    # Main Body
    attack = ATTACKS[cfg.attack_class](cfg)
    evaluator = Evaluator(cfg)
    
    
    adv_image_dir = os.path.join(cfg.data.tgt_data_path, "1")
    caption_path = os.path.join(adv_image_dir, "caption.json")
    keyword_path = os.path.join(adv_image_dir, "keywords.json")
    with open(caption_path, "r") as f:
        caption_list = json.load(f)
    with open(keyword_path, "r") as f:
        keyword_list = json.load(f)

    all_metrics = defaultdict(list)
    target_iter = itertools.cycle(data_loader_target)

    if cfg.data.cle_data_path == "resources/images_all":
        cfg.data.output += "_all" 

    for i, (ori_image, _, path_org) in enumerate(data_loader_origin):
        tar_image, _, path_tgt = next(target_iter)

        if cfg.specific is not None and i not in cfg.specific:
            continue
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break

        
        print(f"\n\nProcessing image {i}/{(cfg.data.num_samples-1)//cfg.data.batch_size}")
        adv_image = attack.attack(ori_image, tar_image, i)

        tar_text = caption_list[i % 100]["caption"][0]
        metrics = evaluator.evaluate(adv_image, torch.clamp(ori_image / 255.0, 0.0, 1.0), tar_text)


        print("📊 Evaluation Metrics:")
        pprint(metrics, indent=2)
        for k, v in metrics.items():
             if isinstance(v, torch.Tensor):
                 v = v.item()
             all_metrics[k].append(v)
        # wandb_metrics = {
        #     f"img{i:02d}_{k}": (v.item() if torch.is_tensor(v) else v)
        #     for k, v in metrics.items()
        # }
        # wandb.log(wandb_metrics)

        # Save images
        for path_idx in range(len(path_org)):
            folder, name = (
                path_org[path_idx].split("/")[-2],
                path_org[path_idx].split("/")[-1],
            )
            # Use config hash in output path
            for dir in [f"all/{config_hash}", f"latest_{cfg.wandb.name}"]:
                adv_folder_to_save = os.path.join(cfg.data.output, dir, "img", folder)
                pert_folder_to_save = os.path.join(cfg.data.output, dir, "pert", folder)
                ensure_dir(adv_folder_to_save)
                ensure_dir(pert_folder_to_save)
                if "JPEG" in name:
                    torchvision.utils.save_image(
                        adv_image[path_idx], os.path.join(adv_folder_to_save, name[:-4]) + "png"
                    )
                elif "png" in name:
                    torchvision.utils.save_image(
                        adv_image[path_idx], os.path.join(adv_folder_to_save, name)
                    )
                    torchvision.utils.save_image(
                        (adv_image[path_idx] - torch.clamp(ori_image[path_idx] / 255.0, 0.0, 1.0).to(adv_image.device)) * 3, os.path.join(pert_folder_to_save, name)
                    )
    
    print("\n📈 Average Metrics Summary:")
    avg_metrics = {}

    for k, values in all_metrics.items():
        first_val = values[0]
        if isinstance(first_val, (float, int, torch.Tensor)):
            # 转成 float 再求平均
            clean_vals = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in values]
            mean_value = sum(clean_vals) / len(clean_vals)
            avg_metrics[f"{k}"] = mean_value
            print(f"  avg_{k:<12}: {mean_value:.4f}")

    wandb.log(avg_metrics)

    print(f"Config Hash: {config_hash}")
    wandb.finish()


if __name__ == "__main__":
    main()
