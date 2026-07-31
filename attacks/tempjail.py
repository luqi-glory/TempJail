from .base import BaseAttack
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
from omegaconf import OmegaConf
from tqdm import tqdm
from surrogates import EnsembleFeatureLoss
from typing import Union, Tuple

from utils import log_metrics, get_models, get_ensemble_loss, hash_training_config, ensure_dir

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
import torchvision.models
import torch.nn.functional as F


def to_512(img_tensor: torch.Tensor) -> torch.Tensor:
    if img_tensor.max() > 1.1:
        img_tensor = img_tensor / 255.0
    img_tensor = F.interpolate(
        img_tensor, size=(512, 512),
        mode='bicubic',
        align_corners=False
    )
    img_tensor = img_tensor * 2 - 1
    return img_tensor.clamp(-1, 1)

def tensor_255_to_neg1_1(x: torch.Tensor) -> torch.Tensor:
    if x.max() <= 10:
        raise ValueError("Not 0-255")
    return x.div(255.0).mul(2).sub(1).clamp(-1, 1)

def tensor_neg1_1_to_0_1(x: torch.Tensor) -> torch.Tensor:
    return x.add(1).div(2).clamp(0, 1)

def tensor_neg1_1_to_float255(x: torch.Tensor) -> torch.Tensor:
    return x.add(1).div(2).mul(255).clamp(0, 255)

def ddpm_mean(noise_pred, t, x_t, sched):
    """
    Compute μ_t = E[x_{t-1} | x_t] for DDPM reverse process.

    Parameters
    ----------
    noise_pred : torch.Tensor
        UNet prediction ε̂(x_t, t).
    t : int or scalar tensor
        Current integer timestep.
    x_t : torch.Tensor
        Latent at timestep t.
    sched : DDPMScheduler
        Diffusers scheduler (has alphas_cumprod, previous_timestep, one).

    Returns
    -------
    torch.Tensor
        Deterministic mean of x_{t-1}.
    """
    # ensure Python int
    t_int   = t.item() if torch.is_tensor(t) else int(t)
    prev_t  = sched.previous_timestep(t_int)

    # cumulative α̅
    alpha_bar_t    = sched.alphas_cumprod[t_int]
    alpha_bar_prev = sched.alphas_cumprod[prev_t] if prev_t >= 0 else sched.one

    # per-step α, β
    alpha_t = alpha_bar_t / alpha_bar_prev          # α_t
    beta_t  = 1.0 - alpha_t                         # β_t

    # x0̂ from predicted noise
    x0_hat = (x_t - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

    # coefficients
    coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t)
    coeff_xt = (alpha_t.sqrt()        * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)

    return coeff_x0 * x0_hat + coeff_xt * x_t

def ddpm_step(noise_pred, t, x_t, sched, n):
    t = t.item() if torch.is_tensor(t) else int(t)
    prev_t = sched.previous_timestep(t)

    alpha_bar_t    = sched.alphas_cumprod[t]                       # \bar α_t
    alpha_bar_prev = sched.alphas_cumprod[prev_t] if prev_t >= 0 else sched.one

    alpha_t = alpha_bar_t / alpha_bar_prev                         # α_t
    beta_t  = 1.0 - alpha_t                                        # β_t

    x0_hat = (x_t - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

    coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t)
    coeff_xt = (alpha_t.sqrt()        * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
    mean     = coeff_x0 * x0_hat + coeff_xt * x_t

    var = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t    # σ_t²
    if t > 0:
        noise = n[t]
        x_prev  = mean + noise
    else:  # t == 0 → deterministic
        x_prev  = mean

    return x_prev.detach()

def collect_xt_and_noise(
    x0,                                      # clean latent image
    t_star,                                  # last timestep to generate
    unet,                                    # ε̂-network
    scheduler,                               # DDPMScheduler
    text_embeddings,
    device):
    """
    Returns
    -------
    X : list[Tensor]      # x_t for t = 0 … t_star-1
    n : list[Tensor]      # n_t  for t = 0 … t_star-1 (n_0 = 0)
    """
    # ---------- forward: build X ----------
    X = []                                  # will hold x_0 … x_{t*-1}
    for t in range(t_star + 1):
        t_tensor = torch.tensor([t], device=device)

        eps_t   = torch.randn_like(x0)      # fresh noise each step
        x_t     = scheduler.add_noise(x0, eps_t, t_tensor)

        X.append(x_t)

    # ---------- reverse: build n ----------
    n = [torch.zeros_like(x0)]              # n_0 is unused / zero by def.
    for t in range(1, t_star + 1):
        t_tensor   = torch.tensor([t], device=device)

        with torch.no_grad():
            noise_pred = unet(X[t], t_tensor, encoder_hidden_states=text_embeddings)["sample"]

        mu_t       = ddpm_mean(noise_pred, t, X[t], scheduler)
        n_t        = X[t - 1] - mu_t
        n.append(n_t)

    return X, n

def get_salient_bbox(image: torch.Tensor, img_index: int, 
                     npy_path: str = "./object/img_dix.npy") -> torch.Tensor:
    if not os.path.exists(npy_path):
        return torch.tensor([0, 0, image.shape[-1], image.shape[-2]])

    sal_map = np.load(npy_path)
    # Randomly pick one saliency mask
    idx = random.randrange(sal_map.shape[0])
    mask = torch.as_tensor(sal_map[idx])

    return mask

def random_bbox_crop(
    img: torch.Tensor,
    ref_box: Tuple[int, int, int, int],
    scale_range: Tuple[float, float] = (0.4, 0.9),
    *,
    target_size: Tuple[int, int] = None,
    return_box: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int, int, int]]]:
    """
    Randomly crop a patch from `img`, whose size is a random ratio of `ref_box`,
    and optionally resize to `target_size`.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor of shape (C, H, W) or (B, C, H, W).
    ref_box : tuple (x, y, w, h)
        Reference bounding box (top-left x, top-left y, width, height) in pixels.
    scale_range : tuple (low, high)
        Uniform sampling range for relative scale s.
    target_size : tuple (H, W), optional
        If given, resize the crop to this size.
    return_box : bool, default=False
        If True, return the coordinates of the sampled box.

    Returns
    -------
    cropped : torch.Tensor
        Cropped (and optionally resized) tensor.
    bbox : tuple (x, y, w, h) [optional]
        Coordinates of the sampled box on the original image.
    """
    has_batch = img.dim() == 4
    if has_batch:
        _, _, H, W = img.shape
    else:
        _, H, W = img.shape

    x0, y0, w0, h0 = ref_box

    # ---- sample scale & size ----
    s = random.uniform(*scale_range)
    w = max(1, int(w0 * s))
    h = max(1, int(h0 * s))

    # ---- sample valid top-left corner ----
    x = random.randint(0, max(0, W - w))
    y = random.randint(0, max(0, H - h))

    # ---- crop ----
    cropped = img[..., y:y + h, x:x + w]

    # ---- resize if needed ----
    if target_size is not None:
        if not has_batch:
            cropped = cropped.unsqueeze(0)  # (1, C, h, w)
        cropped = F.interpolate(cropped, size=target_size, mode='bilinear', align_corners=False)
        if not has_batch:
            cropped = cropped.squeeze(0)

    return (cropped, (x, y, w, h)) if return_box else cropped

def interpolate_bbox(
    box0: Tuple[int, int, int, int],
    box1: Tuple[int, int, int, int],
    r: Union[float, torch.Tensor]
) -> Tuple[int, int, int, int]:
    """
    Linear-interpolate between box0 and box1 with ratio r∈[0,1].

    Parameters
    ----------
    box0, box1 : (x, y, w, h)
        Bounding boxes in pixels.
    r : float or 0-D tensor
        Interpolation factor. 0 → box0, 1 → box1.

    Returns
    -------
    (x, y, w, h) : tuple[int, int, int, int]
        Interpolated box. Values are clamped to >=0 and cast to int.
    """
    if torch.is_tensor(r):
        r = r.item()

    x0, y0, w0, h0 = box0
    x1, y1, w1, h1 = box1

    x = (1 - r) * x0 + r * x1
    y = (1 - r) * y0 + r * y1
    w = (1 - r) * w0 + r * w1
    h = (1 - r) * h0 + r * h1

    x, y, w, h = map(int, (x, y, w, h))
    w = max(1, w)
    h = max(1, h)
    return x, y, w, h

def crop_tensor(
    img: torch.Tensor,
    box: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = None
) -> torch.Tensor:
    """
    Crop a patch from `img` specified by `box`, optionally resizing it.

    Parameters
    ----------
    img : Tensor
        Shape (C, H, W) or (B, C, H, W).
    box : (x, y, w, h)
        Bounding box (top-left x, y, width, height).
    target_size : (target_H, target_W), optional
        If provided, resize the cropped patch to this size.

    Returns
    -------
    Tensor
        Cropped (and possibly resized) patch with same ndim as input.
    """
    has_batch = img.dim() == 4
    _, H, W = img.shape[-3:]

    x, y, w, h = box
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    # Crop
    patch = img[..., y:y + h, x:x + w]  # (B,C,h,w) or (C,h,w)

    # Resize if target_size is specified
    if target_size is not None:
        if not has_batch:
            patch = patch.unsqueeze(0)  # (1, C, h, w)
        patch = F.interpolate(patch, size=target_size, mode='bilinear', align_corners=False)
        if not has_batch:
            patch = patch.squeeze(0)  # back to (C, H, W)

    return patch

class TEMPJAIL(BaseAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        print(cfg)
        self.ensemble_extractor, models = get_models(cfg)
        self.ensemble_loss = get_ensemble_loss(models)
        model_id = "AI-ModelScope/stable-diffusion-2-1"

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id,
            torch_dtype=torch.float16 if cfg.model.use_fp16 else torch.float32
        ).to(cfg.device)
        self.pipe.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    def attack(self, image_org: torch.Tensor, image_tgt: torch.Tensor, img_index: int = 0):
        cfg = self.cfg.agd
        device = self.cfg.device
        vae = self.pipe.vae
        unet = self.pipe.unet
        scheduler = self.pipe.scheduler

        
        image_org = image_org.to(self.pipe.device)
        image_tgt = image_tgt.to(self.pipe.device)

        text_input = self.pipe.tokenizer(
            [""],  # Empty prompt for unconditional generation
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)  # Move tokenizer output to device
        self.pipe.text_encoder = self.pipe.text_encoder.to(device)  # Move text encoder to device
        text_embeddings = self.pipe.text_encoder(text_input.input_ids)[0]

        self.pipe.unet = self.pipe.unet.to(device)

        scaling_factor = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        
        N = cfg.iterations
        s = cfg.scale
        delta = cfg.clip_eps
        t_star_idx = int((1 - cfg.t_star) * scheduler.timesteps.shape[0])
        print(f"###########\nt_star_idx: {t_star_idx}\n###########")
        t_star = scheduler.timesteps[t_star_idx]
        print(f"############\nt_star: {t_star}\n###########")
        for n in range(N):
            print(f"TEMPJAIL Attack Iteration {n+1} of {N}")
            # Step 1: Inversion: Add noise to x0 → x_t*
            if n == 0:
                with torch.no_grad():
                    x0 = vae.encode(to_512(image_org)).latent_dist.sample() * scaling_factor
            else:
                x0 = xt_adv.clone().detach()
            X, n = collect_xt_and_noise(
                x0, t_star, unet, scheduler, text_embeddings, device
            )
            xt_adv = X[t_star].clone().detach()
            
            pbar = tqdm(scheduler.timesteps[t_star_idx:], desc="denoising steps")
            print(f"###########\n{scheduler.timesteps[t_star_idx:]}\n###########")
            for t in pbar:
                t_tensor = torch.tensor([t], device=device)
                
                # Step 2: Inversion: co-evolving selection
                ot_box = get_salient_bbox(image_tgt, img_index)         # (x, y, w, h)
                full_box =image_tgt.shape
                # print(f"###########\n{full_box}\n###########")
                ratio = t / t_star            # 0〜1
                rt_box = interpolate_bbox(ot_box, full_box, ratio)
                rt_crop = crop_tensor(image_tgt, rt_box, target_size=(224, 224))
                rt_box = torch.tensor([0, 0, 224, 224], device=device)   # 注意这里修改了！
                # print(f"###########\n{ot_box}\n{full_box}\n{rt_box}\n{rt_crop}###########")
                with torch.no_grad():
                    self.ensemble_loss.set_ground_truth(rt_crop)

                x_hat = xt_adv.clone().detach().requires_grad_(True)
                
                x_hat = xt_adv

                x_hat.requires_grad = True

                x_hat_pixel = self.pipe.vae.decode(x_hat / scaling_factor).sample
                x_hat_pixel = F.interpolate(x_hat_pixel, size=(224, 224), mode='bilinear', align_corners=False)
                x_hat_pixel = tensor_neg1_1_to_float255(x_hat_pixel)
               
                cand_feats = []
                cand_imgs  = []
                for _ in range(cfg.N_cand): 
                    a_crop = random_bbox_crop(x_hat_pixel, rt_box, scale_range=cfg.s_range, target_size=(224, 224))
                    cand_imgs.append(a_crop)
                    cand_feats.append(self.ensemble_extractor(a_crop))
                sim = torch.stack([self.ensemble_loss(cf) for cf in cand_feats])
                at_feat = cand_feats[sim.argmax()]      # 最相似者

                loss = self.ensemble_loss(at_feat)
                pbar.set_postfix({'alignment objective': f'{loss.item():.4f}'})
                loss.backward()

                # Step 3: update
                grad = x_hat.grad
                grad = torch.clamp(grad, -delta, delta)
                x_hat = x_hat + s * grad

                with torch.no_grad():
                    noise_pred = unet(x_hat, t_tensor, encoder_hidden_states=text_embeddings)["sample"]

                xt_adv = ddpm_step(noise_pred, t_tensor, x_hat, scheduler, n).detach()

        with torch.no_grad():
            adv_image = vae.decode(xt_adv / scaling_factor).sample
            adv_image = tensor_neg1_1_to_0_1(adv_image)
            adv_image = F.interpolate(adv_image, size=(224, 224), mode='bilinear', align_corners=False)
        return adv_image
    