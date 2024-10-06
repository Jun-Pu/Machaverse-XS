"""Script to fine-tune stable-video-to-3D."""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune stable video 3D"
    )

    # data
    parser.add_argument(
        "--videos_complete_dir",
        type=str,
        default="/home/disk2/CMSlots_assets/MachaverseXS_enhanced_rgba_tr",
        required=False,
        help="Path to the complete object renderings of the Machaverse training set",
    )
    parser.add_argument(
        "--videos_defect_dir",
        type=str,
        default="/home/disk2/CMSlots_assets/MachaverseXS_enhanced_inpaints_tr",
        required=False,
        help="Path to the defect objecy renderings of the Machaverse training set",
    )
    parser.add_argument(
        "--defect_type",
        type=str,
        default="scan_b2f_25p",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        default="cam_01",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--azimuths_rad",
        type=list,
        default=[0.0,
                 0.3141592653589793, 0.6283185307179586, 0.9424777960769379, 1.2566370614359172,
                 1.5707963267948966, 1.8849555921538759, 2.199114857512855,  2.5132741228718345,
                 2.827433388230814,  3.141592653589793,  3.4557519189487724, 3.7699111843077517,
                 4.084070449666731,  4.39822971502571,   4.71238898038469,   5.026548245743669,
                 5.340707511102648,  5.654866776461628,
                 5.969026041820607
                 ],
    )
    parser.add_argument(
        "--polars_rad",
        type=list,
        default=[1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
                 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
                 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
                 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
                 ],
    )

    # pretrains
    parser.add_argument(
        "--pipe_status",
        type=str,
        default="train",
        required=False,
        help="The status of the model, either train or test",
    )
    parser.add_argument(
        "--pretrains_pth",
        type=str,
        default="/home/disk2/SV4D_assets/PRETRAINS/sv3d_p.safetensors",
        required=False,
        help="Path to pre-trained models",
    )
    parser.add_argument(
        "--pretrained_model_configs",
        type=str,
        default= "/home/vipuser/Setting_InpaintSV3D/sv3d_models/scripts/sampling/configs/sv3d_inpaint.yaml",
        required=False,
        help="Path to pre-trained model configs",
    )

    # validation during training
    parser.add_argument(
        "--check_data_flow_steps",
        type=int,
        default=50,
        help=(
            "Check if the data have been loaded correctly during the model training process"
        ),
    )

    # save models
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/vipuser/Setting_InpaintSV3D/ckpts_20241007",
        help="The output directory where the model architectural information and checkpoints will be written",
    )
    parser.add_argument(
        "--output_model_info_dir",
        type=str,
        default="/home/vipuser/Setting_InpaintSV3D/ckpts_20241007/model_info",
    )

    # training pipeline
    parser.add_argument(
        "--cond_aug",
        type=float,
        default=0.02,
        help="sv3d hyper=params"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=3000,
        help="Save a checkpoint of the training state every X updates",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.5e-5,
        help="Constant learning rate for the model fine-tuning.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    # model hyper-parameters
    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--vae_scaling_factor",
        type=float,
        default=0.18215,
        help=("The dimension of the LoRA update matrices."),
    )

    return parser.parse_args()

args = parse_args()