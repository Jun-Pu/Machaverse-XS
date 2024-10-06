"""Script to fine-tune stable-video-to-3D."""
import logging
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from tqdm.auto import tqdm
from einops import rearrange, repeat
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from configs import args
from utils import (rand_log_normal, print_network, sv3d_get_batch,
                   sv3d_get_unique_embedder_keys_from_conditioner)
from data import MachaverseDataSet
from sv3d_models.load_sv3d import load_model
import torch.nn.functional as F
from torch.autograd import Variable

logger = get_logger(__name__, log_level="INFO")


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    # make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # if passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    # ---------------------------------------------- load sv3d ---------------------------------------------------------
    # for mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    sv3d_model = load_model(
        ckpts=args.pretrains_pth,
        pipe_status=args.pipe_status,
        config=args.pretrained_model_configs,
        device="cuda",
        num_frames=args.num_frames,
        num_steps=25,
        verbose=False,
    )
    print_network(sv3d_model, "sv3d_inpainting")
    sv3d_model.to(accelerator.device, dtype=weight_dtype)
    # ------------------------------------------------------------------------------------------------------------------

    # set the optimizer
    optimizer_cls = torch.optim.AdamW

    # check model parameters (save all param ids to a .txt file)
    if not os.path.exists(args.output_model_info_dir): os.makedirs(args.output_model_info_dir)
    txtFile = open(os.path.join(args.output_model_info_dir, "sv3d_params_total.txt"), 'w')

    sv3d_model_params = sv3d_model.state_dict().keys()
    for param in sv3d_model_params:
        txtFile.write(param + "\n")

    txtFile.close()

    #  set the frozen/trainable components
    parameters_list = []

    # sv3d backbone
    for name, para in sv3d_model.named_parameters():
        if (("transformer_blocks" in name)):
        # if "diffusion_model" in name:
            parameters_list.append(para)
            para.requires_grad = True
        else:
            para.requires_grad = False

    # update the optimizer
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # double check model parameters (save frozen/trainable params to different .txt files)
    if accelerator.is_main_process:
        txtFileParamsP1 = open(os.path.join(args.output_model_info_dir, 'sv3d_params_frozen.txt'), 'w')
        txtFileParamsP2 = open(os.path.join(args.output_model_info_dir, 'sv3d_params_trainable.txt'), 'w')
        for name, para in sv3d_model.named_parameters():
            if para.requires_grad is False:
                txtFileParamsP1.write(f'{name}\n')
            else:
                txtFileParamsP2.write(f'{name}\n')
        txtFileParamsP1.close()
        txtFileParamsP2.close()

    # dataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = MachaverseDataSet(imgTrWidth=args.img_width, imgTrHeight=args.img_height,
                                      imgDefectType=args.defect_type, camType=args.camera_type,
                                      numFrames=args.num_frames, azimuths=args.azimuths_rad, polars=args.polars_rad)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # prepare everything with our `accelerator`.
    sv3d_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        sv3d_model, optimizer, lr_scheduler, train_dataloader
    )

    # we need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # we need to initialize the trackers we use, and also store our configuration.
    # the trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("InpaintSV3D", config=vars(args))

    # --------------------------------------------- Training Loop ------------------------------------------------------
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        sv3d_model.train()

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):  # batch: [1, num_frames, 3, img_size, img_size]
            with accelerator.accumulate(sv3d_model):

                # get the video sequences and inpainting masks
                sequence_complete = batch["sequence_complete"].to(weight_dtype).to(accelerator.device)

                # get the image references and the inpainting masks
                reference_complete = batch["reference_complete"].to(weight_dtype).to(accelerator.device)

                object_id = batch["object_id"]
                if global_step % args.check_data_flow_steps == 0:
                    print("Current object id is: " + object_id[0])

                # ----------------- get the input&supervision(target) for training the latent unet ---------------------
                # ----------------------- load the shared sv3d's vae and compute the latents ---------------------------
                sv3d_conditioner_embedders = sv3d_model.conditioner.embedders

                # get the latents of the sequential defect views & the corresponding masks
                sequence_complete_rea = rearrange(sequence_complete, "b f c h w -> (b f) c h w")
                seq_complete_latents = sv3d_conditioner_embedders[1].encoder.encode(sequence_complete_rea)
                seq_complete_latents = rearrange(seq_complete_latents, "(b f) c h w -> b f c h w",
                                               f=args.num_frames)
                seq_complete_latents = seq_complete_latents * args.vae_scaling_factor  # [1, num_frames, 4, imgH // 8, imgW // 8]

                sv3d_latents_given = seq_complete_latents

                sv3d_latents_target = seq_complete_latents
                # ------------------------------------------------------------------------------------------------------

                # ------------- add noise incrementally to the vae-encoder's output (DDPM forward) ---------------------
                # sample noise that we'll add to the latents
                noise = torch.randn_like(sv3d_latents_given)
                bsz = sv3d_latents_given.shape[0]  # batch size

                # sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(sv3d_latents_given.device)

                # add noise to the latents according to the noise magnitude at each timestep
                sigmas = sigmas[:, None, None, None, None]
                noisy_sv3d_latents_given = sv3d_latents_given + noise * sigmas
                in_noisy_sv3d_latents_given = noisy_sv3d_latents_given / ((sigmas**2 + 1) ** 0.5)
                # ------------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------

                # ----------------------------- train the latent unet (DDPM backward) ----------------------------------
                # ------------------------------------ get the conditional inputs  -------------------------------------
                camera_azimuths, camera_polars = batch["camera_azimuths"], batch["camera_polars"]
                for idx in range(args.num_frames):
                    camera_azimuths[idx] = round(float(camera_azimuths[idx][0]), 16)
                    camera_polars[idx] = round(float(camera_polars[idx][0]), 16)

                sv3d_value_dict = {}
                sv3d_value_dict["cond_frames_without_noise"] = reference_complete
                sv3d_value_dict["motion_bucket_id"] = 127
                sv3d_value_dict["fps_id"] = 7
                sv3d_value_dict["cond_aug"] = args.cond_aug
                sv3d_value_dict["cond_frames"] = reference_complete + args.cond_aug * torch.randn_like(reference_complete)
                sv3d_value_dict["polars_rad"] = camera_polars
                sv3d_value_dict["azimuths_rad"] = camera_azimuths

                sv3d_batch = sv3d_get_batch(
                    sv3d_get_unique_embedder_keys_from_conditioner(sv3d_model.conditioner),
                    sv3d_value_dict,
                    [1, args.num_frames],
                    T=args.num_frames,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )

                in_cond = sv3d_model.conditioner.get_conditional_conditioning(
                    sv3d_batch,
                )
                in_cond["vector"] = in_cond["vector"].to(weight_dtype)

                for key_id in ["crossattn", "concat"]:
                    in_cond[key_id] = repeat(in_cond[key_id], "b ... -> b t ...", t=args.num_frames)
                    in_cond[key_id] = rearrange(in_cond[key_id], "b t ... -> (b t) ...", t=args.num_frames)
                # ------------------------------------------------------------------------------------------------------

                # --------------------------------- get the additional inputs ------------------------------------------
                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(1, args.num_frames).to(
                    weight_dtype).to(accelerator.device)
                additional_model_inputs["num_video_frames"] = sv3d_batch["num_video_frames"]
                # ------------------------------------------------------------------------------------------------------

                # ------------------------------- get the InpaintSV3D's prediction -------------------------------------
                # we assume the batch size equals to 1 (currently the bsz has to be 1)
                sv3d_vid_latents_in = in_noisy_sv3d_latents_given[0].to(weight_dtype)
                sv3d_latents_in = torch.cat(
                    [sv3d_vid_latents_in, in_cond["concat"]], dim=1
                )

                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                sv3d_timesteps_in = torch.cat(args.num_frames * [timesteps])

                sv3d_context_in = in_cond["crossattn"]
                sv3d_vector_in = in_cond["vector"]

                sv3d_latents_in = Variable(sv3d_latents_in, requires_grad=True)
                sv3d_timesteps_in = Variable(sv3d_timesteps_in, requires_grad=True)
                sv3d_context_in = Variable(sv3d_context_in, requires_grad=True)
                sv3d_vector_in = Variable(sv3d_vector_in, requires_grad=True)
                additional_model_inputs["image_only_indicator"] = Variable(additional_model_inputs[
                                                                "image_only_indicator"], requires_grad=True)

                sv3d_latents_pred = sv3d_model.model.diffusion_model.forward(
                    x=sv3d_latents_in,
                    timesteps=sv3d_timesteps_in,
                    context=sv3d_context_in,
                    y=sv3d_vector_in,
                    num_video_frames=additional_model_inputs["num_video_frames"],
                    image_only_indicator=additional_model_inputs["image_only_indicator"]
                )
                sv3d_latents_pred = sv3d_latents_pred.unsqueeze(0)
                # ------------------------------------------------------------------------------------------------------

                # ------------------------- recover the latents based on the predicted noise ---------------------------
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_sv3d_latents = sv3d_latents_pred * c_out + c_skip * noisy_sv3d_latents_given
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                # ------------------------------------------------------------------------------------------------------

                # ---------- compute the mse loss between the initial latents and their denoised counterparts ----------
                loss = F.mse_loss(denoised_sv3d_latents.float(), sv3d_latents_target.float(), reduction="none")
                loss = weighing.float() * loss

                loss = loss.mean()

                # gather the losses across all processes for logging (if we use distributed training)
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # ------------------------------------------------------------------------------------------------------

                # -------------------------------------- back propagation ----------------------------------------------
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # ------------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------

            # check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # save the intermediate results
            if global_step % args.checkpointing_steps == 0:
                save_file(sv3d_model.state_dict(), os.path.join(args.output_dir, "ckpts_InpaintSV3D_"
                                                                + format(str(global_step), '0>8s') + ".safetensors"))


            if global_step >= args.max_train_steps:
                break
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------ save checkpoints ------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_file(sv3d_model.state_dict(), os.path.join(args.output_dir, "ckpts_InpaintSV3D_"
                                                        + format(str(global_step), '0>8s') + ".safetensors"))

    accelerator.end_training()
    # ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main(args)