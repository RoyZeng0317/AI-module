from typing import Any

import torch, math

class Module:
    """基礎模型類別，模仿 nn.Module"""
    def __init__(self):
        self._parameters = []

    def parameters(self):
        return self._parameters

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _register_param(self, tensor: torch.Tensor):
        """將 Tensor 註冊為參數(自動設定 requires_grad=True)"""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Parameters must be torch.Tensor")
        tensor.requires_grad_(True)
        self._parameters.append(tensor)
        return tensor
    