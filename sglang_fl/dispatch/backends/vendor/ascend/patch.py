"""Vendor monkey-patches on sglang internals for Ascend / NPU — entrypoint.

These replace direct edits to sglang source that were previously required on
Huawei NPU:
  - scheduler_pp: PP send/recv ordering + stream syncs (HCCL deadlock fix)
  - attention_registry: defer CUDA-only linear-attn backend imports on NPU
  - qwen_vl_processor: transformers-version-compatible Qwen-VL preprocess
"""

import logging

from .patches.attention_registry import patch_attn_backend_wrapper
from .patches.minimax_m3_image import (
    patch_minimax_m3_image_preprocess_device,
    patch_minimax_m3_vision_rope_compile,
)
from .patches.minimax_m3_kv_pool import patch_minimax_m3_kv_pool
from .patches.minimax_m3_moe import (
    patch_minimax_m3_moe_activation,
    patch_minimax_m3_w8a8_moe_offsets,
    patch_minimax_m3_w8a8_quant_targets,
)
from .patches.qwen_vl_processor import patch_qwen_vl_processor
from .patches.scheduler_pp import (
    patch_pp_launch_batch_sync,
    patch_pp_send_recv_order,
)

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_ascend_patches() -> None:
    """Apply all Ascend/NPU-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_pp_send_recv_order()
    patch_pp_launch_batch_sync()
    patch_attn_backend_wrapper()
    patch_qwen_vl_processor()
    patch_minimax_m3_moe_activation()
    patch_minimax_m3_w8a8_moe_offsets()
    patch_minimax_m3_w8a8_quant_targets()
    patch_minimax_m3_image_preprocess_device()
    patch_minimax_m3_vision_rope_compile()
    patch_minimax_m3_kv_pool()

apply_ascend_patches()
