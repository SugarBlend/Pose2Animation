# 🧠 Project Overview

This research project focuses on **assessing the accuracy of skeleton marking in 3D**, based on the results of **2D pose estimators**. The pipeline includes lifting 2D pose data into 3D space and **transferring the resulting skeleton animations into Blender** for visualization and further analysis.

## ⚙️ Environment Setup

To set up the development environment, run the following script:

```powershell
.\setup_environment.bat
```
> 📌 **Note:** CUDA **12.8** must be installed **before** running the script.
> You can download it from the official NVIDIA archive: [CUDA 12.8 Download](https://developer.nvidia.com/cuda-12-8-1-download-archive)
> Pre-installation of [UV](https://docs.astral.sh/uv/getting-started/installation/) is also required.


## 🎬 Demo 2d: SOTA Pose Estimation with [Sapiens](https://github.com/facebookresearch/sapiens/tree/main)
Сreating preliminary markup:
```powershell
python .\src\label_creator.py --file .\resources\demo.mp4 --detector yolo12m --detections_file .\resources\detections.yml
```

To run the demo using the state-of-the-art pose estimation model **Sapiens**, use the following command:
```powershell
python .\src\demo.py --pose_config .\configs\sapiens_pose\goliath\sapiens_0.3b-210e_goliath-1024x768.py `
--pose_checkpoints .\checkpoints\goliath\sapiens_0.3b\sapiens_0.3b_goliath_best_goliath_AP_573.pth `
--input_file .\resources\demo.mp4 --detections_file .\resources\detections.yml --device cuda:0 --half `
--use_ffmpegcv --output_file .\results --visualize
```
> 📌 **Note:** Make sure to download the model weights from the [Hugging Face repository](https://huggingface.co/facebook/sapiens) before running the demo.
