from argparse import ArgumentParser, Namespace
import cv2
from colorama import Fore, Style
from pathlib import Path
import numpy as np
from mmcv.visualization.image import imshow_det_bboxes
from itertools import count
from imutils.video.filevideostream import FileVideoStream
import yaml
import torch
from tqdm import tqdm
from ultralytics import YOLO


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--file", default="../resources/demo.mp4",
                        help="Path to video file.")
    parser.add_argument("--detections_file", default="../resources/detections.yml",
                        help="Path to output file with created detections.")
    parser.add_argument("--detector", default="yolo12x",
                        help="Name of detector which supported by Ultralytics.")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Name of detector which supported by Ultralytics.")
    args = parser.parse_args()

    if not Path(args.file).exists():
        raise Exception(f"Video path '{args.file}' not found.")

    args.detections_file = Path(args.detections_file)
    args.detections_file.parent.mkdir(exist_ok=True, parents=True)

    return args


def create_detections(args: Namespace) -> None:
    cache_path = Path.cwd().joinpath(f"cache/{Path(args.detector).stem}.pt")
    model = YOLO(cache_path)

    cap = FileVideoStream(args.file, queue_size=256)
    cap.start()
    frames_detections = {}

    custom_format = (
        f"{Fore.WHITE}{{desc}}: {{percentage:2.0f}}% |{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, "
        f"{{rate_fmt}}]{Style.RESET_ALL}"
    )
    frame_number = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(count(), total=frame_number, bar_format=custom_format, position=0, ncols=70):
        frame = cap.read()
        if frame is None:
            break

        results = model(frame, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu()
            confs = results.boxes.conf.cpu().unsqueeze(1)
            classes = results.boxes.cls.cpu().unsqueeze(1)
            detections = torch.cat([boxes, confs, classes], dim=1).cpu().numpy()
            detections = detections[detections[:, 5] == 0]

            if args.show:
                imshow_det_bboxes(
                    frame, detections[:, :5], labels=detections[:, 5].astype(np.int32), class_names=["person"],
                    win_name="Detected labels", wait_time=1, bbox_color=(0, 165, 255), text_color="blue",
                    font_scale=0.5, thickness=3
                )

            detections = detections.tolist()
        else:
            detections = []

        frames_detections[i] = detections

    cap.stop()

    with args.detections_file.open("w") as file:
        yaml.dump(frames_detections, file, sort_keys=False)


if __name__ == "__main__":
    create_detections(get_arguments())
