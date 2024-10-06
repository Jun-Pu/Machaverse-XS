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
sys.path.append("/home/vipuser/Setting_InpaintSV3D/sv3d_models")

# from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor


def sample(
    # pretrain_path: str = "/home/disk2/SV4D_assets/PRETRAINS/sv3d_p.safetensors",
    pretrain_path: str = "/home/vipuser/Setting_InpaintSV3D/ckpts_20241003/ckpts_InpaintSV3D_00003000.safetensors",
    input_path: str = "/home/vipuser/Setting_InpaintSV3D/DATA/test",
    num_steps: int = 25, # being consistent w/ the training config
    version: str = "sv3d_inpaint",
    fps_id: int = 7, # being consistent w/ the training config
    motion_bucket_id: int = 127, # being consistent w/ the training config
    cond_aug: float = 0.02, # being consistent w/ the training config
    seed: int = 123, # being consistent w/ the training config
    decoding_t: int = 14,  # number of frames decoded at a time; reduce if necessary.
    device: str = "cuda",
    output_folder: str =  "/home/vipuser/Setting_InpaintSV3D/res_ckpts_20241003_steps_00003000_w_ref_cpmplete",
    verbose: bool = False, # being consistent w/ the training config
    num_frames: int = 20, # being consistent w/ the training config
    img_width: int = 1024, # being consistent w/ the training config
    img_height: int = 576, # being consistent w/ the training config
    model_config: str = "/home/vipuser/Setting_InpaintSV3D/sv3d_models/scripts/sampling/configs/sv3d_inpaint.yaml",
    azimuths_rad: list =
    [            0.0,
                 0.3141592653589793, 0.6283185307179586, 0.9424777960769379, 1.2566370614359172,
                 1.5707963267948966, 1.8849555921538759, 2.199114857512855,  2.5132741228718345,
                 2.827433388230814,  3.141592653589793,  3.4557519189487724, 3.7699111843077517,
                 4.084070449666731,  4.39822971502571,   4.71238898038469,   5.026548245743669,
                 5.340707511102648,  5.654866776461628,
                 5.969026041820607
    ], # being consistent w/ the training config
    polars_rad: list =
    [            1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
                 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
                 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
                 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
    ], # being consistent w/ the training config

    pipe_status: str = "test",

    # polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    # azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    # azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
):

    # load the inpainting model
    model = load_model(
        pretrain_path,
        model_config,
        pipe_status,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    # load the testing data & conduct model inference
    os.makedirs(output_folder, exist_ok=True)
    refIDs = os.listdir(os.path.join(input_path))
    if ".DS_Store" in refIDs: refIDs.remove(".DS_Store")
    for ref_id in refIDs:
        imgDefect = cv2.imread(os.path.join(input_path, ref_id), flags=cv2.IMREAD_UNCHANGED)
        imgDefect[:, :, 0] = imgDefect[:, :, 0] * (imgDefect[:, :, -1] / 255) + 255 - imgDefect[:, :, -1]
        imgDefect[:, :, 1] = imgDefect[:, :, 1] * (imgDefect[:, :, -1] / 255) + 255 - imgDefect[:, :, -1]
        imgDefect[:, :, 2] = imgDefect[:, :, 2] * (imgDefect[:, :, -1] / 255) + 255 - imgDefect[:, :, -1]
        imgDefect = imgDefect[:, :, :3]

        imgDefect = cv2.resize(imgDefect, (img_width, img_height))

        imgDefect_tensor = torch.from_numpy(imgDefect).float()

        imgDefect_normalized = imgDefect_tensor / 127.5 - 1
        imgDefect_normalized = imgDefect_normalized.permute(2, 0, 1).unsqueeze(0).to(device)

        value_dict = {}
        value_dict["cond_frames_without_noise"] = imgDefect_normalized
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = imgDefect_normalized + cond_aug * torch.randn_like(imgDefect_normalized)

        value_dict["polars_rad"] = polars_rad
        value_dict["azimuths_rad"] = azimuths_rad

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
                # crossattn: [1,1,1024]; concat: [1,4,XX,XX]; vector: [num_frames, 1280]

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
                # crossattn: [num_frames, 1, 1024]; concat: [num_frames, 4, XX, XX]

                shape = (num_frames, 4, img_height // 8, img_width // 8)
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

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)

                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples_x = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                samples_x = embed_watermark(samples_x)

                # rearrange the r g b
                samples_x_chn0 = samples_x[:, 0, :, :].unsqueeze(1)
                samples_x_chn1 = samples_x[:, 1, :, :].unsqueeze(1)
                samples_x_chn2 = samples_x[:, 2, :, :].unsqueeze(1)
                samples_x_rearranged = torch.cat([samples_x_chn2, samples_x_chn1, samples_x_chn0], dim=1)

                vid = (
                    (rearrange(samples_x_rearranged, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                video_pth = os.path.join(output_folder, ref_id + ".mp4")
                imageio.mimwrite(video_pth, vid, fps=value_dict["fps_id"])
        print(ref_id + " inference done;")
    print("All done.")


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device, weight_dtype):
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
    pipe_status: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpt
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
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    # filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model


if __name__ == "__main__":
    Fire(sample)