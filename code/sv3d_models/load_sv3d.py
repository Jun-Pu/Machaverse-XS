import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import sys
sys.path
sys.path.append("/home/vipuser/Setting_InpaintSV3D/sv3d_models")

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
# from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor


def load_model(
    ckpts: str,
    pipe_status: str,
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpts
    config.model.params.pipe_status = pipe_status
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device)
    else:
        model = instantiate_from_config(config.model).to(device)

    # filter = DeepFloydDataFiltering(verbose=False, device=device)

    return model


# debug
#  pip install git+https://github.com/openai/CLIP.git
#  pip install imageio[pyav] -i https://pypi.tuna.tsinghua.edu.cn/simple
#  pip install imageio[ffmpeg] -i https://pypi.tuna.tsinghua.edu.cn/simple