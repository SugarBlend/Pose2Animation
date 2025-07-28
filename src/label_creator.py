import tempfile
from argparse import ArgumentParser, Namespace
from itertools import count
from pathlib import Path

import cv2
import torch
import yaml
from colorama import Fore, Style
from tqdm import tqdm
from ultralytics import YOLO


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--file", default="../resources/demo.mp4",
                        help="Path to video file.")
    parser.add_argument("--detections_file", default="../resources/detections.yml",
                        help="Path to output file with created detections.")
    parser.add_argument("--detector", default="yolo12m",
                        help="Name of detector which supported by Ultralytics.")
    args = parser.parse_args()

    if not Path(args.file).exists():
        raise Exception(f"Video path '{args.file}' not found.")

    args.detections_file = Path(args.detections_file)
    args.detections_file.parent.mkdir(exist_ok=True, parents=True)

    return args


def create_detections(args: Namespace) -> None:
    with tempfile.TemporaryDirectory() as path:
        model = YOLO(f"{path}/{args.detector}.pt")

    cap = cv2.VideoCapture(args.file)
    frames_detections = {}

    custom_format = (
        f"{Fore.WHITE}{{desc}}: {{percentage:2.0f}}% |{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, "
        f"{{rate_fmt}}]{Style.RESET_ALL}"
    )
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(count(), total=frame_number, bar_format=custom_format, position=0, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu()
            confs = results.boxes.conf.cpu().unsqueeze(1)
            classes = results.boxes.cls.cpu().unsqueeze(1)
            detections = torch.cat([boxes, confs, classes], dim=1)
            detections = detections.tolist()
        else:
            detections = []

        frames_detections[i] = detections

    cap.release()

    with args.detections_file.open("w") as file:
        yaml.dump(frames_detections, file, sort_keys=False)


if __name__ == "__main__":
    create_detections(get_arguments())
