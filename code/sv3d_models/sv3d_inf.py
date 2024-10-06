import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove

import sys
sys.path.append("/home/vipuser/camp_sv3d/sv3d_models")

# from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor


def sample(
    pretrain_path: str = "/home/disk2/SVDMerchandiseVidGen_assets/PRETRAINS/sv3d/sv3d_p.safetensors",
    input_path: str = "/home/vipuser/camp_sv3d/DATA/te/merchandise/0.55_背提包_779199287921_1_0.jpg",  # Can either be image file or folder with image files
    num_steps: Optional[int] = None,
    version: str = "sv3d_p",
    fps_id: int = 7,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 123,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,

    elevations_deg: Union[float, List[float]] = 0.0,
    # elevations_deg = [-90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],

    azimuths_deg: Optional[List[float]] = None,
 
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    if version == "sv3d_p":
        num_frames = 21
        imgW, imgH = 1024, 1024
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "/home/vipuser/camp_sv3d/OUTPUTS")
        model_config = "/home/vipuser/camp_sv3d/sv3d_models/scripts/sampling/configs/sv3d_p.yaml"
        cond_aug = 1e-5
        if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
            elevations_deg = [elevations_deg] * num_frames
        assert (
            len(elevations_deg) == num_frames
        ), f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
        if azimuths_deg is None:
            azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360

        # change to clockwise orbital direction  (invalid to the vanilla version)
        # sub_azimuths_deg = azimuths_deg[:-1]
        # sub_azimuths_deg = sorted(sub_azimuths_deg, reverse=False)
        # azimuths_deg[:-1] = sub_azimuths_deg

        # ---------------------------------------- DIY orbital camera pose ---------------------------------------------
        # azimuths_deg = np.linspace(0, 60, num_frames + 1)[1:] % 60

        assert (
                len(azimuths_deg) == num_frames
        ), f"Please provide a list of {num_frames} values for azimuths_deg! Given {len(azimuths_deg)}"

        azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    else:
        raise ValueError(f"We do not consider version {version} here.")

    model = load_model(
        pretrain_path,
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in all_img_paths:
        image = Image.open(input_img_path)
        image = image.resize((imgW, imgH), Image.Resampling.LANCZOS)
        image_tensor = torch.from_numpy(np.array(image)).float()

        # normalize the image by scaling pixel values to [-1, 1]
        image_normalized = image_tensor / 127.5 - 1

        # Rearrange channels if necessary
        image_normalized = image_normalized.permute(
                2, 0, 1)  # For RGB images
        image_normalized = image_normalized.unsqueeze(0).to(device)

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image_normalized  # [1,3,576,576]
        value_dict["motion_bucket_id"] = motion_bucket_id  # [127]
        value_dict["fps_id"] = fps_id  # [10]
        value_dict["cond_aug"] = cond_aug  # [1e-05]
        value_dict["cond_frames"] = image_normalized + cond_aug * torch.randn_like(image_normalized)  # [1,3,576,576]

        value_dict["polars_rad"] = polars_rad  # [21]
        value_dict["azimuths_rad"] = azimuths_rad  # [21]

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                    weight_dtype= torch.float32
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )
                # c=uc={crossattn: [1,1,1024], concat:[1,4,72,72], vector:[21,1280]}

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                shape = (num_frames, 4, imgH // 8, imgW // 8)
                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )
                # input: [42,4,72,72]; sigma: [42]; c: [[42,1,1024], [42,4,72,72], [42,1280]];
                # additional_model_inputs: {image_only_indicator: [2,21], num_video_frames:21}

                # c=uc={crossattn: [21,1,1024], concat:[21,4,72,72], vector:[21,1280]}
                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)  # [21,4,72,72]
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                if "sv3d" in version:
                    samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))

                imageio.imwrite(
                    os.path.join(output_folder, f"{base_count:06d}.jpg"), image)

                samples = embed_watermark(samples)
                # samples = filter(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                # video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                video_path = os.path.join(output_folder, input_path.split("/")[-1][:-4] + ".mp4")
                imageio.mimwrite(video_path, vid, fps=value_dict["fps_id"])


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device, weight_dtype):  # N [1,21]; T 21
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(weight_dtype).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(weight_dtype).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    ckpt: str,
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpt
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
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    # filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model


if __name__ == "__main__":
    Fire(sample)


# debug
#  pip install git+https://github.com/openai/CLIP.git
#  pip install imageio[pyav] -i https://pypi.tuna.tsinghua.edu.cn/simple
#  pip install imageio[ffmpeg] -i https://pypi.tuna.tsinghua.edu.cn/simple


#  exact environment:
# # Name                    Version                   Build  Channel
# _libgcc_mutex             0.1                        main    https://repo.anaconda.com/pkgs/main
# _openmp_mutex             5.1                       1_gnu    https://repo.anaconda.com/pkgs/main
# accelerate                0.28.0                   pypi_0    pypi
# addict                    2.4.0                    pypi_0    pypi
# aiofiles                  23.2.1                   pypi_0    pypi
# aiohttp                   3.9.3                    pypi_0    pypi
# aiosignal                 1.3.1                    pypi_0    pypi
# albumentations            0.4.6                    pypi_0    pypi
# altair                    5.3.0                    pypi_0    pypi
# annotated-types           0.7.0                    pypi_0    pypi
# antlr4-python3-runtime    4.9.3                    pypi_0    pypi
# anyio                     3.7.1                    pypi_0    pypi
# appdirs                   1.4.4                    pypi_0    pypi
# asttokens                 2.4.1                    pypi_0    pypi
# async-timeout             4.0.3                    pypi_0    pypi
# av                        12.0.0                   pypi_0    pypi
# backcall                  0.2.0                    pypi_0    pypi
# beautifulsoup4            4.12.3                   pypi_0    pypi
# bleach                    6.1.0                    pypi_0    pypi
# braceexpand               0.1.7                    pypi_0    pypi
# ca-certificates           2024.3.11            h06a4308_0    https://repo.anaconda.com/pkgs/main
# certifi                   2024.2.2                 pypi_0    pypi
# charset-normalizer        3.3.2                    pypi_0    pypi
# click                     8.1.7                    pypi_0    pypi
# coloredlogs               15.0.1                   pypi_0    pypi
# contourpy                 1.1.1                    pypi_0    pypi
# cycler                    0.12.1                   pypi_0    pypi
# decorator                 5.1.1                    pypi_0    pypi
# decord                    0.6.0                    pypi_0    pypi
# defusedxml                0.7.1                    pypi_0    pypi
# diffusers                 0.24.0                   pypi_0    pypi
# dnspython                 2.6.1                    pypi_0    pypi
# docker-pycreds            0.4.0                    pypi_0    pypi
# docopt                    0.6.2                    pypi_0    pypi
# easydict                  1.13                     pypi_0    pypi
# einops                    0.7.0                    pypi_0    pypi
# email-validator           2.1.1                    pypi_0    pypi
# exceptiongroup            1.2.1                    pypi_0    pypi
# executing                 2.0.1                    pypi_0    pypi
# fastapi                   0.111.0                  pypi_0    pypi
# fastapi-cli               0.0.4                    pypi_0    pypi
# fastjsonschema            2.19.1                   pypi_0    pypi
# ffmpy                     0.3.2                    pypi_0    pypi
# filelock                  3.13.3                   pypi_0    pypi
# fire                      0.6.0                    pypi_0    pypi
# flatbuffers               24.3.25                  pypi_0    pypi
# fonttools                 4.51.0                   pypi_0    pypi
# frozenlist                1.4.1                    pypi_0    pypi
# fsspec                    2024.3.1                 pypi_0    pypi
# ftfy                      6.2.0                    pypi_0    pypi
# gitdb                     4.0.11                   pypi_0    pypi
# gitpython                 3.1.43                   pypi_0    pypi
# gradio                    4.31.5                   pypi_0    pypi
# gradio-client             0.16.4                   pypi_0    pypi
# h11                       0.14.0                   pypi_0    pypi
# httpcore                  1.0.5                    pypi_0    pypi
# httptools                 0.6.1                    pypi_0    pypi
# httpx                     0.27.0                   pypi_0    pypi
# huggingface-hub           0.23.0                   pypi_0    pypi
# humanfriendly             10.0                     pypi_0    pypi
# hydra-core                1.3.2                    pypi_0    pypi
# idna                      3.6                      pypi_0    pypi
# imageio                   2.27.0                   pypi_0    pypi
# imageio-ffmpeg            0.4.9                    pypi_0    pypi
# imgaug                    0.4.0                    pypi_0    pypi
# importlib-metadata        7.1.0                    pypi_0    pypi
# importlib-resources       6.4.0                    pypi_0    pypi
# invisible-watermark       0.2.0                    pypi_0    pypi
# ipython                   8.12.3                   pypi_0    pypi
# jaxtyping                 0.2.19                   pypi_0    pypi
# jedi                      0.19.1                   pypi_0    pypi
# jinja2                    3.1.2                    pypi_0    pypi
# joblib                    1.4.2                    pypi_0    pypi
# jupyter-client            8.6.2                    pypi_0    pypi
# jupyter-core              5.7.2                    pypi_0    pypi
# jupyterlab-pygments       0.3.0                    pypi_0    pypi
# kiwisolver                1.4.5                    pypi_0    pypi
# kornia                    0.7.2                    pypi_0    pypi
# kornia-rs                 0.1.2                    pypi_0    pypi
# lazy-loader               0.3                      pypi_0    pypi
# ld_impl_linux-64          2.38                 h1181459_1    https://repo.anaconda.com/pkgs/main
# libffi                    3.4.4                h6a678d5_0    https://repo.anaconda.com/pkgs/main
# libgcc-ng                 11.2.0               h1234567_1    https://repo.anaconda.com/pkgs/main
# libgomp                   11.2.0               h1234567_1    https://repo.anaconda.com/pkgs/main
# libstdcxx-ng              11.2.0               h1234567_1    https://repo.anaconda.com/pkgs/main
# lightning                 2.3.0                    pypi_0    pypi
# lightning-utilities       0.11.2                   pypi_0    pypi
# linkify-it-py             2.0.3                    pypi_0    pypi
# llvmlite                  0.41.1                   pypi_0    pypi
# markdown-it-py            3.0.0                    pypi_0    pypi
# markupsafe                2.1.3                    pypi_0    pypi
# matplotlib                3.7.5                    pypi_0    pypi
# matplotlib-inline         0.1.7                    pypi_0    pypi
# mdit-py-plugins           0.4.1                    pypi_0    pypi
# mdurl                     0.1.2                    pypi_0    pypi
# mistune                   3.0.2                    pypi_0    pypi
# mpmath                    1.3.0                    pypi_0    pypi
# multidict                 6.0.5                    pypi_0    pypi
# natsort                   8.4.0                    pypi_0    pypi
# nbclient                  0.10.0                   pypi_0    pypi
# nbconvert                 7.16.4                   pypi_0    pypi
# nbformat                  5.10.4                   pypi_0    pypi
# ncurses                   6.4                  h6a678d5_0    https://repo.anaconda.com/pkgs/main
# networkx                  3.0                      pypi_0    pypi
# numba                     0.58.1                   pypi_0    pypi
# numpy                     1.24.4                   pypi_0    pypi
# nvidia-cublas-cu11        11.11.3.6                pypi_0    pypi
# nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
# nvidia-cuda-cupti-cu11    11.8.87                  pypi_0    pypi
# nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
# nvidia-cuda-nvrtc-cu11    11.8.89                  pypi_0    pypi
# nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
# nvidia-cuda-runtime-cu11  11.8.89                  pypi_0    pypi
# nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
# nvidia-cudnn-cu11         8.7.0.84                 pypi_0    pypi
# nvidia-cudnn-cu12         8.9.2.26                 pypi_0    pypi
# nvidia-cufft-cu11         10.9.0.58                pypi_0    pypi
# nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
# nvidia-curand-cu11        10.3.0.86                pypi_0    pypi
# nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
# nvidia-cusolver-cu11      11.4.1.48                pypi_0    pypi
# nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
# nvidia-cusparse-cu11      11.7.5.86                pypi_0    pypi
# nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
# nvidia-nccl-cu11          2.19.3                   pypi_0    pypi
# nvidia-nccl-cu12          2.19.3                   pypi_0    pypi
# nvidia-nvjitlink-cu12     12.4.127                 pypi_0    pypi
# nvidia-nvtx-cu11          11.8.86                  pypi_0    pypi
# nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
# omegaconf                 2.3.0                    pypi_0    pypi
# onnx                      1.16.1                   pypi_0    pypi
# onnxruntime               1.17.1                   pypi_0    pypi
# open-clip-torch           2.24.0                   pypi_0    pypi
# openai-clip               1.0.1                    pypi_0    pypi
# opencv-python             4.9.0.80                 pypi_0    pypi
# opencv-python-headless    4.9.0.80                 pypi_0    pypi
# openssl                   3.0.13               h7f8727e_0    https://repo.anaconda.com/pkgs/main
# orjson                    3.10.3                   pypi_0    pypi
# packaging                 24.0                     pypi_0    pypi
# pandas                    2.0.3                    pypi_0    pypi
# pandocfilters             1.5.1                    pypi_0    pypi
# parso                     0.8.4                    pypi_0    pypi
# pickleshare               0.7.5                    pypi_0    pypi
# pillow                    10.2.0                   pypi_0    pypi
# pip                       24.0                     pypi_0    pypi
# pipreqs                   0.5.0                    pypi_0    pypi
# platformdirs              4.2.0                    pypi_0    pypi
# plotly                    5.22.0                   pypi_0    pypi
# plyfile                   1.0.3                    pypi_0    pypi
# pooch                     1.8.1                    pypi_0    pypi
# prompt-toolkit            3.0.45                   pypi_0    pypi
# protobuf                  4.25.3                   pypi_0    pypi
# psutil                    5.9.8                    pypi_0    pypi
# pure-eval                 0.2.2                    pypi_0    pypi
# pycocotools               2.0.7                    pypi_0    pypi
# pycryptodome              3.20.0                   pypi_0    pypi
# pydantic                  2.7.1                    pypi_0    pypi
# pydantic-core             2.18.2                   pypi_0    pypi
# pydub                     0.25.1                   pypi_0    pypi
# pygments                  2.18.0                   pypi_0    pypi
# pymatting                 1.1.12                   pypi_0    pypi
# pyparsing                 3.1.2                    pypi_0    pypi
# python                    3.8.19               h955ad1f_0    https://repo.anaconda.com/pkgs/main
# python-dateutil           2.9.0.post0              pypi_0    pypi
# python-dotenv             1.0.1                    pypi_0    pypi
# python-multipart          0.0.9                    pypi_0    pypi
# pytorch-lightning         2.2.1                    pypi_0    pypi
# pytz                      2024.1                   pypi_0    pypi
# pywavelets                1.4.1                    pypi_0    pypi
# pyyaml                    6.0.1                    pypi_0    pypi
# pyzmq                     26.0.3                   pypi_0    pypi
# readline                  8.2                  h5eee18b_0    https://repo.anaconda.com/pkgs/main
# regex                     2023.12.25               pypi_0    pypi
# rembg                     2.0.56                   pypi_0    pypi
# requests                  2.31.0                   pypi_0    pypi
# rich                      13.7.1                   pypi_0    pypi
# ruff                      0.4.5                    pypi_0    pypi
# safetensors               0.4.2                    pypi_0    pypi
# scikit-image              0.21.0                   pypi_0    pypi
# scikit-learn              1.3.2                    pypi_0    pypi
# scipy                     1.10.0                   pypi_0    pypi
# semantic-version          2.10.0                   pypi_0    pypi
# sentencepiece             0.2.0                    pypi_0    pypi
# sentry-sdk                1.44.0                   pypi_0    pypi
# setproctitle              1.3.3                    pypi_0    pypi
# setuptools                68.2.0                   pypi_0    pypi
# shapely                   2.0.4                    pypi_0    pypi
# shellingham               1.5.4                    pypi_0    pypi
# smmap                     5.0.1                    pypi_0    pypi
# sniffio                   1.3.1                    pypi_0    pypi
# soupsieve                 2.5                      pypi_0    pypi
# sqlite                    3.41.2               h5eee18b_0    https://repo.anaconda.com/pkgs/main
# stack-data                0.6.3                    pypi_0    pypi
# starlette                 0.37.2                   pypi_0    pypi
# supervision               0.20.0                   pypi_0    pypi
# sympy                     1.12                     pypi_0    pypi
# tenacity                  8.4.1                    pypi_0    pypi
# termcolor                 2.4.0                    pypi_0    pypi
# threadpoolctl             3.5.0                    pypi_0    pypi
# tifffile                  2023.7.10                pypi_0    pypi
# timm                      0.9.16                   pypi_0    pypi
# tinycss2                  1.3.0                    pypi_0    pypi
# tk                        8.6.12               h1ccaba5_0    https://repo.anaconda.com/pkgs/main
# tokenizers                0.19.1                   pypi_0    pypi
# tomli                     2.0.1                    pypi_0    pypi
# tomlkit                   0.12.0                   pypi_0    pypi
# toolz                     0.12.1                   pypi_0    pypi
# torch                     2.2.2                    pypi_0    pypi
# torchaudio                2.2.2                    pypi_0    pypi
# torchmetrics              1.3.2                    pypi_0    pypi
# torchvision               0.17.2                   pypi_0    pypi
# tornado                   6.4                      pypi_0    pypi
# tqdm                      4.66.2                   pypi_0    pypi
# traitlets                 5.14.3                   pypi_0    pypi
# transformers              4.41.2                   pypi_0    pypi
# triton                    2.2.0                    pypi_0    pypi
# typeguard                 4.3.0                    pypi_0    pypi
# typer                     0.12.3                   pypi_0    pypi
# typing                    3.7.4.3                  pypi_0    pypi
# typing-extensions         4.10.0                   pypi_0    pypi
# tzdata                    2024.1                   pypi_0    pypi
# uc-micro-py               1.0.3                    pypi_0    pypi
# ujson                     5.10.0                   pypi_0    pypi
# urllib3                   2.2.1                    pypi_0    pypi
# uvicorn                   0.29.0                   pypi_0    pypi
# uvloop                    0.19.0                   pypi_0    pypi
# wandb                     0.16.5                   pypi_0    pypi
# watchfiles                0.21.0                   pypi_0    pypi
# wcwidth                   0.2.13                   pypi_0    pypi
# webdataset                0.2.86                   pypi_0    pypi
# webencodings              0.5.1                    pypi_0    pypi
# websockets                11.0.3                   pypi_0    pypi
# wheel                     0.41.2                   pypi_0    pypi
# xformers                  0.0.25.post1             pypi_0    pypi
# xz                        5.4.6                h5eee18b_0    https://repo.anaconda.com/pkgs/main
# yapf                      0.40.2                   pypi_0    pypi
# yarg                      0.1.9                    pypi_0    pypi
# yarl                      1.9.4                    pypi_0    pypi
# zipp                      3.18.1                   pypi_0    pypi
# zlib                      1.2.13               h5eee18b_0    https://repo.anaconda.com/pkgs/main