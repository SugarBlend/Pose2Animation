import mimetypes
from argparse import ArgumentParser, Namespace
from itertools import count
from pathlib import Path

import cv2
import ffmpegcv
import numpy as np
import torch
import yaml
from colorama import Fore, Style
from mmpose.apis import init_model as init_pose_estimator
from mmpose.visualization.fast_visualizer import FastVisualizer
from tqdm import tqdm

from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
)
from executor import inference_topdown
from preprocess import PosePreprocessor
from utils import GPUArray, pycuda_context_reset, visualizer_adapter


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pose_config",
                        default="../configs/sapiens_pose/goliath/sapiens_0.3b-210e_goliath-1024x768.py",
                        help="Config for pose estimator.")
    parser.add_argument("--pose_checkpoints",
                        default="../checkpoints/goliath/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_AP_573.pth",
                        help="Checkpoints for pose estimator.")
    parser.add_argument("--input_file", default="../resources/demo.mp4",
                        help="Path to video file for pose estimation.")
    parser.add_argument("--detections_file", default="../resources/detections.yml",
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


if __name__ == "__main__":
    args = get_arguments()

    if args.pose_checkpoints.endswith(".pth"):
        pose_estimator = init_pose_estimator(
            args.pose_config,
            args.pose_checkpoints,
            device=args.device,
        )
        pose_estimator.eval()
        pose_estimator.to(args.device)
        pose_estimator.to(args.precision)
        model = torch.compile(pose_estimator, mode="reduce-overhead", fullgraph=True)
        model.test_cfg.flip_test = False
        model_cfg = model.cfg
    else:
        raise Exception("At now implemented only general PyTorch model format processing.")
    preprocessor = PosePreprocessor(model_cfg.image_size, model.data_preprocessor.mean, model.data_preprocessor.std)

    if args.use_ffmpegcv:
        with pycuda_context_reset():
            cap = ffmpegcv.toCUDA(ffmpegcv.VideoCaptureNV(str(args.input_file), pix_fmt="nv12"))
        frames_number = cap.count - 1
    else:
        cap = cv2.VideoCapture(str(args.input_file))
        frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    if args.visualize:
        if model_cfg.num_keypoints == 308:  # noqa: PLR2004
            skeleton_info, pts_colors = GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS
        elif model_cfg.num_keypoints == 133:  # noqa: PLR2004
            skeleton_info, pts_colors = COCO_WHOLEBODY_SKELETON_INFO, COCO_WHOLEBODY_KPTS_COLORS
        elif model_cfg.num_keypoints == 17:  # noqa: PLR2004
            skeleton_info, pts_colors = COCO_SKELETON_INFO, COCO_KPTS_COLORS
        else:
            raise Exception("Model configure consider unsupported number of keypoints to "
                            f"visualization: {model_cfg.num_keypoints}")

        meta_info = visualizer_adapter(skeleton_info, pts_colors)
        visualizer = FastVisualizer(meta_info, radius=3, line_width=1, kpt_thr=0.3)
        cv2.namedWindow("Visualization", cv2.WINDOW_GUI_EXPANDED)

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

    with args.detections_file.open(mode="rb") as file:
        data = yaml.safe_load(file)

    custom_format = (
        f"{Fore.WHITE}{{desc}}: {{percentage:2.0f}}% |{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, "
        f"{{rate_fmt}}]{Style.RESET_ALL}"
    )
    detection_thresh = 0.4
    class_thresh = 0

    for i in tqdm(count(), total=frames_number, bar_format=custom_format, position=0, leave=True, ascii=" ▁▂▃▄▅▆▇█",
                  ncols=70):
        if args.use_ffmpegcv:
            ret, torch_tensor = cap.read_torch()
            if not ret:
                break
        else:
            ret, img = cap.read()
            if not ret:
                break
            torch_tensor = torch.from_numpy(img).cuda().to(torch.float32).permute(2, 0, 1)

        detections = np.asarray(data[i])
        detections = detections[detections[:, 4] > detection_thresh]
        detections = detections[detections[:, 5] == class_thresh]

        if np.prod(detections.shape):
            bboxes = torch.tensor(detections[:, :4], dtype=torch.float32, device=args.device)
            batched_data, centers, scales = preprocessor(img=torch_tensor, bboxes=bboxes)
            predictions = inference_topdown(model, batched_data.to(args.precision).contiguous())
            for j, prediction in enumerate(predictions):
                prediction.keypoints = ((prediction.keypoints / preprocessor.input_shape) * scales[j] + centers[j] -
                                        0.5 * scales[j])
        else:
            predictions = []

        if args.visualize or args.output_file:
            img = np.ascontiguousarray(torch_tensor.permute(1, 2, 0).to(torch.uint8).cpu().numpy())

            if args.visualize:
                for predict in predictions:
                    visualizer.draw_pose(img, predict)
                cv2.imshow("Visualization", img)
                cv2.waitKey(args.delay)

            if args.output_file:
                writer.write(img)
    cap.release()
