"""Script to fine-tune stable-video-to-3D."""
import random
import os

import cv2
import numpy as np
from PIL import Image
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from configs import args


class MachaverseDataSet(Dataset):
    def __init__(self,
                 imgTrWidth=1024,
                 imgTrHeight=576,
                 imgDefectType="scan_b2f_25p",
                 camType="cam_01",
                 numFrames=25,
                 azimuths=None,
                 polars=None,
                 ):

        videoCompleteIDs, videoDefectIDs, maskInpaintIDs = [], [], []
        categoryIDs = os.listdir(args.videos_complete_dir)
        if ".DS_Store" in categoryIDs: categoryIDs.remove(".DS_Store")
        for ctg_id in categoryIDs:
            objectIDs = os.listdir(os.path.join(args.videos_complete_dir, ctg_id))
            if ".DS_Store" in objectIDs: objectIDs.remove(".DS_Store")
            objectIDs.sort()
            for obj_id in objectIDs:
                vid_complete_pth = os.path.join(args.videos_complete_dir, ctg_id, obj_id, camType,
                                                "multiviews_complete")
                videoCompleteIDs.append(vid_complete_pth)

                vid_defect_pth = os.path.join(args.videos_defect_dir, ctg_id, obj_id, camType,
                                              "multiviews_defect", imgDefectType, "images")
                videoDefectIDs.append(vid_defect_pth)

                msk_inpaint_pth = os.path.join(args.videos_defect_dir, ctg_id, obj_id, camType,
                                              "multiviews_defect", imgDefectType, "masks")
                maskInpaintIDs.append(msk_inpaint_pth)


        self.videos_complete = videoCompleteIDs
        self.videos_defect = videoDefectIDs
        self.masks_inpaint = maskInpaintIDs

        self.image_channels = 3 # rgb
        self.image_width = imgTrWidth
        self.image_height = imgTrHeight
        self.video_length = numFrames
        self.camera_azimuths = azimuths
        self.camera_polars = polars

    def __len__(self):
        return len(self.videos_complete)

    def __getitem__(self, chosen_idx):
        chosen_video_complete_pth = self.videos_complete[chosen_idx]
        chosen_video_defect_pth = self.videos_defect[chosen_idx]
        chosen_mask_inpaint_pth = self.masks_inpaint[chosen_idx]

        data_info = chosen_video_complete_pth.split("/")

        frmIDs = os.listdir(chosen_video_complete_pth)
        if ".DS_Store" in frmIDs: frmIDs.remove(".DS_Store")
        frmIDs.sort()
        frmIDs = frmIDs[:self.video_length]

        # initialize a tensor to store the pixel values
        values_complete_video = torch.empty(
            (self.video_length, self.image_channels, self.image_height, self.image_width))
        values_defect_video = torch.empty(
            (self.video_length, self.image_channels, self.image_height, self.image_width))
        values_inpaint_mask = torch.empty(
            (self.video_length, 1, self.image_height // 8, self.image_width // 8))

        # load and process each frame
        for ii, frm_id in enumerate(frmIDs):

            img_complete_pth = os.path.join(chosen_video_complete_pth, frm_id)
            img_defect_pth = os.path.join(chosen_video_defect_pth, frm_id)
            msk_inpaint_pth = os.path.join(chosen_mask_inpaint_pth, frm_id)

            # load the view representing the complete object
            imgC = cv2.imread(img_complete_pth, flags=cv2.IMREAD_UNCHANGED)
            imgC[:, :, 0] = imgC[:, :, 0] * (imgC[:, :, -1] / 255) + 255 - imgC[:, :, -1]
            imgC[:, :, 1] = imgC[:, :, 1] * (imgC[:, :, -1] / 255) + 255 - imgC[:, :, -1]
            imgC[:, :, 2] = imgC[:, :, 2] * (imgC[:, :, -1] / 255) + 255 - imgC[:, :, -1]
            imgC = imgC[:, :, :3]

            # load the view representing the defect object
            imgD = cv2.imread(img_defect_pth, flags=cv2.IMREAD_UNCHANGED)
            imgD[:, :, 0] = imgD[:, :, 0] * (imgD[:, :, -1] / 255) + 255 - imgD[:, :, -1]
            imgD[:, :, 1] = imgD[:, :, 1] * (imgD[:, :, -1] / 255) + 255 - imgD[:, :, -1]
            imgD[:, :, 2] = imgD[:, :, 2] * (imgD[:, :, -1] / 255) + 255 - imgD[:, :, -1]
            imgD = imgD[:, :, :3]

            # load the inpainting mask corresponding to the defect object
            mskInp = cv2.imread(msk_inpaint_pth, flags=cv2.IMREAD_UNCHANGED)

            # resize the training data
            imgC = cv2.resize(imgC, (self.image_width, self.image_height))
            imgD = cv2.resize(imgD, (self.image_width, self.image_height))
            mskInp = cv2.resize(mskInp, (self.image_width // 8, self.image_height // 8))

            # debug; check if the training data are correctly aligned and transformed
            debug_dir = os.path.join(args.output_dir, "sanity_check")
            if not os.path.exists(debug_dir): os.makedirs(debug_dir)
            if data_info[-4] in ["backpack", "bumbag", "handbag", "suitcase", "hat"]:
                if chosen_idx % 10 == 0 and ii == 5:
                    cv2.imwrite(os.path.join(debug_dir, "complete_obj_" + str(chosen_idx) + ".png"), imgC)
                    cv2.imwrite(os.path.join(debug_dir, "defect_obj_" + str(chosen_idx) + ".png"), imgD)
                    cv2.imwrite(os.path.join(debug_dir, "inpaint_msk_" + str(chosen_idx) + ".png"), mskInp)
            elif data_info[-4] == "shoe":
                if chosen_idx % 10 == 0 and ii == 0:
                    cv2.imwrite(os.path.join(debug_dir, "complete_obj_" + str(chosen_idx) + ".png"), imgC)
                    cv2.imwrite(os.path.join(debug_dir, "defect_obj_" + str(chosen_idx) + ".png"), imgD)
                    cv2.imwrite(os.path.join(debug_dir, "inpaint_msk_" + str(chosen_idx) + ".png"), mskInp)
            else:
                raise ValueError("Check the data!")

            # numpy to tensor
            imgC_tensor = torch.from_numpy(imgC).float()
            imgD_tensor = torch.from_numpy(imgD).float()
            mskInp_tensor = torch.from_numpy(mskInp).float()

            # normalize and export the training data
            imgC_normalized = imgC_tensor / 127.5 - 1
            imgC_normalized = imgC_normalized.permute(2, 0, 1)
            values_complete_video[ii] = imgC_normalized

            imgD_normalized = imgD_tensor / 127.5 - 1
            imgD_normalized = imgD_normalized.permute(2, 0, 1)
            values_defect_video[ii] = imgD_normalized

            mskInp_normalized = mskInp_tensor / 127.5 - 1
            values_inpaint_mask[ii] = mskInp_normalized

        # get the reference image & mask; adjust the sequential order if needed
        if data_info[-4] in ["backpack", "bumbag", "handbag", "suitcase", "hat"]:
            values_defect_ref = values_defect_video[5]
            values_complete_ref = values_complete_video[5]

            # change the sequencial order
            values_complete_video_updated = torch.empty(
                (self.video_length, self.image_channels, self.image_height, self.image_width))
            values_defect_video_updated = torch.empty(
                (self.video_length, self.image_channels, self.image_height, self.image_width))
            values_inpaint_mask_updated = torch.empty(
                (self.video_length, 1, self.image_height // 8, self.image_width // 8))

            values_complete_video_updated[:-5, :, :, :] = values_complete_video[5:, :, :, :]
            values_complete_video_updated[-5:, :, :, :] = values_complete_video[:5, :, :, :]

            values_defect_video_updated[:-5, :, :, :] = values_defect_video[5:, :, :, :]
            values_defect_video_updated[-5:, :, :, :] = values_defect_video[:5, :, :, :]

            values_inpaint_mask_updated[:-5, :, :, :] = values_inpaint_mask[5:, :, :, :]
            values_inpaint_mask_updated[-5:, :, :, :] = values_inpaint_mask[:5, :, :, :]
        elif data_info[-4] == "shoe":
            values_defect_ref =  values_defect_video[0]
            values_complete_ref = values_complete_video[0]

            values_complete_video_updated = values_complete_video
            values_defect_video_updated = values_defect_video
            values_inpaint_mask_updated = values_inpaint_mask
        else:
            raise ValueError("Check the data!")

        return {"sequence_complete": values_complete_video_updated,
                "sequence_defect": values_defect_video_updated,
                "sequence_inpaint_mask": values_inpaint_mask_updated,
                "reference_defect": values_defect_ref,
                "reference_complete": values_complete_ref,
                "object_id": data_info[-3],
                "camera_azimuths": self.camera_azimuths,
                "camera_polars": self.camera_polars
                }