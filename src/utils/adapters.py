from contextlib import contextmanager, suppress
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pycuda.driver as cuda
import torch
from pycuda.driver import PointerHolderBase
from pycuda.gpuarray import GPUArray


@contextmanager
def pycuda_context_reset() -> Generator[None, Any, None]:
    yield
    with suppress(cuda.Error):
        # Pop active PyCUDA context (created by ffmpegcv)
        cuda.Context.pop()


class Holder(PointerHolderBase):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self) -> int:
        return self.tensor.data_ptr()

    # without an __index__ method, arithmetic calls to the GPUArray backed by this pointer fail
    # not sure why, this needs to return some integer, apparently
    def __index__(self) -> int:
        return self.gpudata


def gpuarray_to_tensor(gpu_array: GPUArray) -> torch.Tensor:
    c, h, w = gpu_array.shape
    torch_tensor = torch.empty((c, h, w), dtype=torch.float32, device="cuda")
    cuda.memcpy_dtod_async(Holder(torch_tensor).gpudata, gpu_array.gpudata, gpu_array.nbytes)
    return torch_tensor.contiguous()


def visualizer_adapter(skeleton_info: Dict[str, Any], pts_colors: List[List[int]]) -> Dict[str, Any]:
    pts_colors = np.asarray(pts_colors, dtype=np.uint8)
    skeleton_links: List[Tuple[int, int]] = []
    skeleton_link_colors: List[np.ndarray] = []

    for info in skeleton_info.values():
        pt1, pt2 = info["link"]
        color = np.asarray(info["color"], dtype=np.uint8)
        skeleton_links.append((pt1, pt2))
        skeleton_link_colors.append(color)

    return {
        "keypoint_id2name": None,
        "keypoint_name2id": None,
        "keypoint_colors": pts_colors,
        "skeleton_links": skeleton_links,
        "skeleton_link_colors": skeleton_link_colors,
    }
