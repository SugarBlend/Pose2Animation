from argparse import ArgumentParser, Namespace
from colorama import Fore, Style
import cv2
import ffmpegcv
from pathlib import Path
import pickle
import numpy as np
from mmcv.visualization.image import imshow_det_bboxes
from mmengine.config import Config
from mmpose.visualization.fast_visualizer import FastVisualizer
import mimetypes
from itertools import count
import yaml
import torch
from tqdm import tqdm
from typing import Optional, Type, Union, Tuple, List

from src.model.interface import SapiensEnd2End, Backend
from src.visualization.palettes import (
    COCO_KPTS_COLORS,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
)
from src.utils.adapters import GPUArray, pycuda_context_reset, visualizer_adapter


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pose_config",
                        default="../configs/sapiens_pose/goliath/sapiens_0.3b-210e_goliath-1024x768.py",
                        help="Config for pose estimator.")
    parser.add_argument("--pose_checkpoints",
                        default="../weights/goliath/sapiens_0.3b/tensorrt/sapiens_0.3b_goliath_best_goliath_AP_573.plan",
                        # default="../weights/goliath/sapiens_0.3b/torchscript/sapiens_0.3b_goliath_best_goliath_AP_573.pt",
                        # default="../weights/goliath/sapiens_0.3b/onnx/sapiens_0.3b_goliath_best_goliath_AP_573.onnx",
                        help="Checkpoints for pose estimator.")
    parser.add_argument("--input_file", default=r"D:\Datasets\CHI3D\chi3d_train\train\s02\videos\50591643\Grab 1.mp4",
                        help="Path to video file for pose estimation.")
    parser.add_argument("--detections_file", default=r"E:\Projects\SapiensBlender\src\dump\dataset_couple\50591643_Grab 1.yml",
                        help="Path to video file for pose estimation.")
    parser.add_argument("--device", default="cuda:0",
                        help="Execution device in pytorch")
    parser.add_argument("--half", action="store_true", default=False,
                        help="Use half precision for accelerate processing.")
    parser.add_argument("--use_ffmpegcv", action="store_true", default=False,
                        help="Use ffmpegcv for accelerate video decoding.")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Use visualization.")
    parser.add_argument("--delay", type=int, default=1,
                        help="Delay for visualization in msec.")
    parser.add_argument("--output_file", default="results/",
                        help="Folder where will be stored wrote video with labels.")
    args = parser.parse_args()

    args.input_file = Path(args.input_file)
    args.detections_file = Path(args.detections_file)
    args.precision = torch.float16 if args.half else torch.float32
    args.device = torch.device(args.device)

    mimetype, _ = mimetypes.guess_type(args.input_file)
    if not mimetype.startswith("video"):
        raise Exception(f"A video file was expected at the input, but instead received: '{mimetype}'.")

    if not args.detections_file.exists():
        raise Exception("A detection file is required for successful launch. "
                        f"Expected existed file: '{args.detections_file}'.")

    for item in [args.pose_config, args.pose_checkpoints]:
        if not Path(item).exists():
            raise Exception(f"For model reconstruction must be existed file: '{item}'.")

    return args


def setting_multimedia(args: Namespace) -> Tuple[
    Union[cv2.VideoCapture, ffmpegcv.toCUDA], Optional[FastVisualizer], Optional[cv2.VideoWriter], tqdm
]:
    if args.use_ffmpegcv:
        with pycuda_context_reset():
            cap = ffmpegcv.toCUDA(ffmpegcv.VideoCaptureNV(str(args.input_file), pix_fmt="nv12"))
        frames_number = cap.count - 1
    else:
        cap = cv2.VideoCapture(str(args.input_file))
        frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    custom_format = (
        f"{Fore.WHITE}{{desc}}: {{percentage:2.0f}}% |{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, "
        f"{{rate_fmt}}]{Style.RESET_ALL}"
    )
    progress_bar = tqdm(count(), total=frames_number, bar_format=custom_format, position=0, leave=True,
                       ascii=" ▁▂▃▄▅▆▇█", ncols=70)

    visualizer: Optional[FastVisualizer] = None
    if args.visualize:
        if pipeline.model_configuration.num_keypoints == 308:  # noqa: PLR2004
            skeleton_info, pts_colors = GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS
        elif pipeline.model_configuration.num_keypoints == 133:  # noqa: PLR2004
            skeleton_info, pts_colors = COCO_WHOLEBODY_SKELETON_INFO, COCO_WHOLEBODY_KPTS_COLORS
        elif pipeline.model_configuration.num_keypoints == 17:  # noqa: PLR2004
            skeleton_info, pts_colors = COCO_SKELETON_INFO, COCO_KPTS_COLORS
        else:
            raise Exception("Model configure consider unsupported number of keypoints to "
                            f"visualization: {pipeline.model_configuration.num_keypoints}")

        meta_info = visualizer_adapter(skeleton_info, pts_colors)
        visualizer = FastVisualizer(meta_info, radius=3, line_width=1, kpt_thr=0.3)
        cv2.namedWindow("Visualization", cv2.WINDOW_GUI_EXPANDED)

    writer: Optional[cv2.VideoWriter] = None
    if args.output_file:
        ret, img = cap.read()
        if isinstance(img, np.ndarray):
            h, w, c = img.shape
        elif isinstance(img, (torch.Tensor, GPUArray)):
            c, h, w = img.shape
        else:
            raise Exception(f"Unsupported output tensor type: {type(img)}. "
                            f"Expected types: 'np.ndarray', 'torch.Tensor', 'pycuda.gpuarray.GPUArray'.")

        output_file = Path(f"{args.output_file}/{Path(args.pose_config).stem}.mp4")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_file), fourcc, 60, (w, h))

    return cap, visualizer, writer, progress_bar


if __name__ == "__main__":
    args = get_arguments()
    pipeline = SapiensEnd2End(args.pose_checkpoints, args.pose_config, "cuda:0")
    pipeline.init_executor()
    cap, visualizer, writer, progress_bar = setting_multimedia(args)

    with args.detections_file.open(mode="rb") as file:
        frames_detections = yaml.safe_load(file)

    detection_thresh = 0.4
    class_thresh = 0

    frame_joints: List[np.ndarray] = []
    for idx in progress_bar:
        if args.use_ffmpegcv:
            ret, frame = cap.read_torch()
        else:
            ret, frame = cap.read()
        if not ret:
            break

        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).cuda().to(torch.float32).permute(2, 0, 1)

        detections = np.asarray(frames_detections[idx])
        detections = detections[detections[:, 4] > detection_thresh]
        detections = detections[detections[:, 5] == class_thresh]

        if np.prod(detections.shape):
            bboxes = torch.tensor(detections[:, :4], dtype=torch.float32, device=args.device)
            pipeline.preprocess(frame, bboxes)
            output = pipeline.infer(asynchronous=False)[0]
            joints, keypoint_scores = pipeline.postprocess(output)
        else:
            joints, keypoint_scores = [], []
        frame_joints.append(np.concatenate([joints, keypoint_scores], axis=-1))

        if args.visualize or args.output_file:
            img = np.ascontiguousarray(frame.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
            if args.visualize:
                predictions = [Config({"keypoints": joints[i: i + 1], "keypoint_scores": keypoint_scores[i: i + 1]})
                               for i in range(joints.shape[0])]
                imshow_det_bboxes(
                    img, detections[:, :5], labels=detections[:, 5].astype(np.int32), class_names=["person"],
                    bbox_color=(0, 165, 255), text_color="blue", font_scale=0.5, thickness=3, show=False
                )
                for predict in predictions:
                    visualizer.draw_pose(img, predict)
                cv2.imshow("Visualization", img)
                cv2.waitKey(args.delay)

            if args.output_file:
                writer.write(img)

    args.detections_file = Path(args.detections_file)
    num_joints = pipeline.model_configuration.num_keypoints
    output_file = f"{args.detections_file.parent.joinpath(args.detections_file.stem)}_skeleton_{num_joints}.pkl"
    with open(output_file, "wb") as pkl_file:
        pickle.dump(frame_joints, pkl_file)
    cap.release()
