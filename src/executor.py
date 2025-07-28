import torch


def inference_topdown(
        model: torch.nn.Module,
        tensors: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=tensors.dtype):
        feats = model.extract_feat(tensors)
        return model.head.predict(feats, None, test_cfg=model.test_cfg)
