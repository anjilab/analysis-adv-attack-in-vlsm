import pyrootutils
import copy
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import os
from typing import List, Tuple


import hydra
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger

from src import utils

torch.set_float32_matmul_precision("medium")

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def gen_perturbed_imgs(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.use_ckpt:
        assert cfg.ckpt_path, "You must provide a checkpoint path when `use_ckpt=True`"

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model
    }
        
    test_loader = datamodule.test_dataloader()
    if cfg.use_ckpt:
        log.info(f"Loading checkpoint from <{cfg.ckpt_path}>")
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

    if cfg.trainer.accelerator == "gpu":
        if cfg.trainer.devices:
            device = torch.device(f"cuda:{cfg.trainer.devices[0]}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)
    
    log.info("Generating perturbed images...")
    # Validate the existence of the directory to save the perturbed images
    if not os.path.exists(cfg["adv_imgs_dir"]):
        os.makedirs(cfg["adv_imgs_dir"])

        
    # Hyperparameters
    epsilon = 8/255
    alpha = 2/255 
    num_iterations = 40
    epsilons = [0.01, 0.03, 0.1, 0.5]
    
    
    for eps in epsilons:
        epsilon_dir = os.path.join(cfg["adv_imgs_dir"], f"{eps}")
        epsilon_dir_perturbed= os.path.join(cfg["perturbed_imgs_dir"], f"{eps}")
        
        if not os.path.exists(epsilon_dir_perturbed):
            
            os.makedirs(epsilon_dir_perturbed, exist_ok=True)  # Create the directory if it doesn't exist
            
        if not os.path.exists(epsilon_dir):
            os.makedirs(epsilon_dir, exist_ok=True)  # Create the directory if it doesn't exist
        for batch in tqdm(test_loader):
            # Move data to the target device
            batch["pixel_values"] = batch["pixel_values"].to(device)
            batch["mask"] = batch["mask"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["input_ids"] = batch["input_ids"].to(device)
            
            
            image_names = batch["image_name"]
            orig_images = batch["pixel_values"].clone().detach()
            perturbed_images = orig_images.clone().detach()

            # PGD Attack
            for _ in range(num_iterations):  
                perturbed_images.requires_grad = True
                batch["pixel_values"] = perturbed_images
                step_out = model.step(batch)
                loss = step_out["loss"]
                model.zero_grad()
                loss.backward()
                
                grad_signs = torch.sign(perturbed_images.grad)
                perturbed_images = perturbed_images.detach() + alpha * grad_signs
                delta = torch.clamp(perturbed_images - orig_images, min=-eps, max=eps)
                # perturbed_images = torch.clamp(orig_images + delta, min=0, max=1).detach() # normalization concept understand. 
                perturbed_images = (orig_images + delta).detach()
            
            # Denormalize the perturbed images after PGD attack
            pert_imgs = denormalize(perturbed_images, mean=cfg["img_mean"], std=cfg["img_std"]) 
            for img, name, delta in zip(pert_imgs, image_names, delta):
                torchvision.utils.save_image(delta.double(), os.path.join(cfg["perturbed_imgs_dir"], str(eps), name))
                torchvision.utils.save_image(img.double(), os.path.join(cfg["adv_imgs_dir"], f"{eps}", name))
    
    return {}, object_dict

def denormalize(
    image: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    return image.mul_(std[:, None, None]).add_(mean[:, None, None])


@hydra.main(version_base="1.2", config_path="../configs", config_name="adv.yaml")
def main(cfg: DictConfig) -> None:
    gen_perturbed_imgs(cfg)


if __name__ == "__main__":
    main()
