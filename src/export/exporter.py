from typing import Any

import mmengine.runner.checkpoint
import tensorrt as trt
import torch
from deploy2serve.deployment.core.exporters.factory import Exporter
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.utils.logger import get_logger
from mmpose.apis import init_model as init_pose_estimator


class SapiensExporter(Exporter):
    def __init__(self, config: ExportConfig) -> None:
        super().__init__(config)
        self.logger = get_logger("onnx")

    def load_checkpoints(self, config_path: str, weights_path: str) -> None:
        def patched_load_checkpoint(filename, map_location=None, logger=None) -> Any:  # noqa: ARG001, ANN001, ANN401
            return torch.load(filename, map_location=map_location, weights_only=False)

        mmengine.runner.checkpoint._load_checkpoint = patched_load_checkpoint  # noqa: SLF001
        model = init_pose_estimator(config_path, weights_path, device=self.config.device)
        model.eval()
        model.to(self.config.device)
        dtype = torch.float16 if self.config.enable_mixed_precision else torch.float32
        model.to(dtype)
        model.test_cfg.flip_test = False
        self.model = model

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition: # type: ignore[attr-defined]
        return network

    def register_onnx_plugins(self) -> None:
        pass
