from deploy2serve.deployment.core.executors.factory import ExecutorFactory, Backend, BaseExecutor
from pathlib import Path
import numpy as np
from mmengine.config import Config
import torch
from typing import Optional, Tuple, Type, List, Any, Dict

from src.model.preprocess import PosePreprocessor
from src.model.postprocess import udp_decode


class SapiensEnd2End(object):
    def __init__(self, checkpoints_path: str, config: str, device: str) -> None:
        self.checkpoints_path: str = checkpoints_path
        self.device: torch.device = torch.device(device)

        self.backend: Optional[Backend] = None
        self.executor_cls: Optional[Type[BaseExecutor]] = None
        self.executor: Optional[BaseExecutor] = None
        self.batched_data: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.centers: Optional[torch.Tensor] = None

        self.model_cfg = Config.fromfile(config)
        self.input_shape: Tuple[int, int] = self.model_cfg.image_size
        self.preprocessor = PosePreprocessor(self.model_cfg.image_size,
                                             torch.tensor(self.model_cfg.model.data_preprocessor.mean),
                                             torch.tensor(self.model_cfg.model.data_preprocessor.std))

    def _parse_backend_from_file(self) -> None:
        suffix = Path(self.checkpoints_path).suffix
        backend_correspondence = {
            ".pt": Backend.TorchScript,
            ".onnx": Backend.ONNX,
            ".plan": Backend.TensorRT,
        }
        self.backend = backend_correspondence.get(suffix, None)
        if self.backend is None:
            raise Exception("Unsupported file extension or maybe you need to rename extension for one of "
                            f"the available cases: {backend_correspondence.keys()}")

    def _extend_loader_kwargs(self) -> Dict[str, Any]:
        if self.backend == Backend.TensorRT:
            return {"max_batch_size": 8, "log_level": "info"}
        elif self.backend == Backend.TorchScript:
            return {"enable_mixed_precision": True}
        return {}

    @property
    def model_configuration(self):
        return self.model_cfg

    def init_executor(self, **kwargs) -> None:
        self._parse_backend_from_file()
        default_kwargs = {
            "checkpoints_path": self.checkpoints_path,
            "device": self.device
        }
        if not len(kwargs):
            default_kwargs.update(**self._extend_loader_kwargs())
        else:
            default_kwargs.update(**kwargs)

        self.executor_cls: Type[BaseExecutor] = ExecutorFactory.create(self.backend)
        self.executor = self.executor_cls(**default_kwargs)

    def preprocess(self, tensor: torch.Tensor, bboxes: torch.Tensor) -> None:
        self.batched_data, self.centers, self.scales = self.preprocessor(tensor.to(self.device),
                                                                         bboxes.to(self.device))

    def infer(self, **kwargs) -> List[Any]:
        return self.executor.infer(self.batched_data, **kwargs)

    def postprocess(self, heatmaps: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        num_bodies, num_joints = heatmaps.shape[:2]
        joints = np.zeros((num_bodies, num_joints, 2))
        keypoint_scores = np.zeros((num_bodies, num_joints, 1))
        for idx in range(num_bodies):
            joints[idx], scores = udp_decode(
                heatmaps[idx].float().cpu().numpy(),  # type: ignore[attr-defined]
                self.input_shape,
                np.array(self.input_shape) / 4,
            )
            joints[idx] = ((joints[idx] / self.preprocessor.input_shape) * self.scales[idx] + self.centers[idx] -
                           0.5 * self.scales[idx])
            keypoint_scores[idx] = scores.reshape(1, num_joints, 1)
        return joints, keypoint_scores
