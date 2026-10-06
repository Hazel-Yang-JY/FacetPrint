from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


VIEWS = {"bf16", "w8a16", "w4a16", "w8a8", "w4a8"}


def fake_quant_symmetric(
    tensor: torch.Tensor, bits: int, group_size: int | None
) -> torch.Tensor:
    """Quantize-dequantize with a straight-through estimator."""
    qmax = 2 ** (bits - 1) - 1
    original_shape = tensor.shape
    if group_size and tensor.shape[-1] >= group_size and tensor.shape[-1] % group_size == 0:
        grouped = tensor.reshape(*tensor.shape[:-1], -1, group_size)
    else:
        grouped = tensor.reshape(*tensor.shape[:-1], 1, tensor.shape[-1])
    scale = grouped.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-7) / qmax
    qdq = torch.round(grouped / scale).clamp(-qmax, qmax) * scale
    qdq = qdq.reshape(original_shape)
    return tensor + (qdq - tensor).detach()


class FacetPrintLinear(nn.Module):
    """LoRA linear shared by all FacetPrint numerical training views."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, group_size: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.group_size = group_size
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.view = "bf16"
        self._cached_view: str | None = None
        self._cached_base_weight: torch.Tensor | None = None

    def set_view(self, view: str) -> None:
        if view not in VIEWS:
            raise ValueError(f"Unknown FacetPrint view: {view}")
        if view != self.view:
            self._cached_view = None
            self._cached_base_weight = None
        self.view = view

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight_bits = 8 if self.view in {"w8a16", "w8a8"} else 4 if self.view in {"w4a16", "w4a8"} else None
        if self.view in {"w8a8", "w4a8"}:
            inputs = fake_quant_symmetric(inputs, 8, None)
        if weight_bits is None:
            base_weight = self.base.weight
            lora_a = self.lora_a
            lora_b = self.lora_b
        else:
            if self._cached_view != self.view or self._cached_base_weight is None:
                with torch.no_grad():
                    self._cached_base_weight = fake_quant_symmetric(
                        self.base.weight, weight_bits, self.group_size
                    ).detach()
                self._cached_view = self.view
            base_weight = self._cached_base_weight
            lora_a = fake_quant_symmetric(self.lora_a, weight_bits, self.group_size)
            lora_b = fake_quant_symmetric(self.lora_b, weight_bits, None)
        base_output = F.linear(inputs, base_weight, self.base.bias)
        low_rank_output = F.linear(F.linear(inputs, lora_a), lora_b) * self.scale
        return base_output + low_rank_output

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            dtype=self.base.weight.dtype,
            device=self.base.weight.device,
        )
        with torch.no_grad():
            merged.weight.copy_(
                self.base.weight + (self.lora_b @ self.lora_a) * self.scale
            )
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def _resolve_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    group_size: int,
) -> int:
    targets = set(target_modules)
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets
    ]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, module in replacements:
        parent, child = _resolve_parent(model, name)
        setattr(parent, child, FacetPrintLinear(module, rank, alpha, group_size))
    if not replacements:
        raise RuntimeError("No target linear modules were found")
    return len(replacements)


def set_view(model: nn.Module, view: str) -> None:
    if view not in VIEWS:
        raise ValueError(f"Unknown FacetPrint view: {view}")
    for module in model.modules():
        if isinstance(module, FacetPrintLinear):
            module.set_view(view)


def merge_lora(model: nn.Module) -> int:
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, FacetPrintLinear)
    ]
    for name, module in replacements:
        parent, child = _resolve_parent(model, name)
        setattr(parent, child, module.merged_linear())
    return len(replacements)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
