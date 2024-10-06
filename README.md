# I2MVs: MultiView Video Generation from a Single Image
In this repository, we provide:

- [x] A small dataset containing [real-scanned 3D assets](https://omniobject3d.github.io/) across six categories of commonly seen clothing accessories.
- [x] A non-official [SV3D](https://stability.ai/news/introducing-stable-video-3d) training code.
- [ ] A new 3D inpainting method


## The DataSet
We collected 154 real-scanned 3D models from the [OmniObject3D](https://omniobject3d.github.io/) , manually masked varying percentages of the visible mesh faces, and re-rendered orbital videos with [Blender](https://www.blender.org/): 

![hat_003_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/hat_003_cam_01.gif)
![backpack_019_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/backpack_019_cam_01.gif)
![bumbag_004_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/bumbag_004_cam_01.gif)
![handbag_053_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/handbag_053_cam_01.gif)
![shoe_003_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/shoe_003_cam_01.gif)
![suitcase_006_cam_01](https://github.com/Jun-Pu/I2MVs/blob/main/assets/suitcase_006_cam_01.gif)

The whole dataset with official split can be found at [Google Drive]().

## SV3D Re-Implementation

### Training

Please update the [config.py]() before running [train.py]().

### Inference

Run [test.py]().

### Results upon the TestSet

