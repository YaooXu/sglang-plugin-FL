# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Keep the current MiniMax-M3.1 image processor on CPU on Ascend.

SGLang's NPU preprocessing hook recognizes ``MiniMaxVLProcessor``. The
checkpoint's Transformers processor is named ``MiniMaxM3VLProcessor`` instead,
so SGLang sends its unpatched 10-dimensional patch extraction to the NPU,
which supports at most eight dimensions. The upstream processor succeeds on
CPU; its output is transferred to the model device by the normal MM path.
"""

import hashlib
import json
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def patch_minimax_m3_image_preprocess_device() -> None:
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    original = BaseMultimodalProcessor._fast_image_processor_device
    if getattr(original, "_sglang_fl_minimax_m3_image_cpu_patch", False):
        return

    @wraps(original)
    def patched(self, processor):
        if type(processor).__name__ == "MiniMaxM3VLProcessor":
            if not getattr(self, "_sglang_fl_minimax_m3_image_cpu_logged", False):
                logger.info("Using CPU preprocessing for MiniMaxM3VLProcessor on NPU")
                self._sglang_fl_minimax_m3_image_cpu_logged = True
            return "cpu"
        return original(self, processor)

    patched._sglang_fl_minimax_m3_image_cpu_patch = True
    BaseMultimodalProcessor._fast_image_processor_device = patched


def patch_minimax_m3_vision_rope_compile() -> None:
    """Use eager MiniMax vision RoPE with FlagTree's Triton on Ascend.

    The upstream function is torch.compile-decorated. Torch NPU's Inductor
    cache imports ``triton_key``, which FlagTree's Triton module does not
    provide. The undecorated function performs the same RoPE operations and
    can run in the normal vision forward pass without Inductor.
    """
    from sglang.srt.models import minimax_vl_common

    compiled = minimax_vl_common._minimax_rope_applier
    eager = getattr(compiled, "__wrapped__", None)
    if eager is None:
        return
    if getattr(eager, "_sglang_fl_minimax_m3_vision_rope_patch", False):
        return

    eager._sglang_fl_minimax_m3_vision_rope_patch = True
    minimax_vl_common._minimax_rope_applier = eager
    logger.info("Using eager MiniMax vision RoPE on Ascend")


def patch_minimax_m3_image_token_boundaries() -> None:
    """Preserve the HF image framing when expanding native token prompts.

    The generic retokenization avoidance path repeats only the image token.
    MiniMax's HF processor also inserts a start/end token around every image;
    without those tokens adjacent images collapse into a single MM span.
    Keep the caller's text tokens verbatim and restore only missing framing.
    """
    from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor

    original = MiniMaxM3VLProcessor._expand_input_ids
    if getattr(original, "_sglang_fl_minimax_m3_image_boundaries_patch", False):
        return

    @wraps(original)
    def patched(self, original_ids, counts, placeholder_token_id):
        if placeholder_token_id != self.IM_TOKEN_ID:
            return original(original_ids, counts, placeholder_token_id)
        # Retain the upstream validation (including placeholder/count mismatch).
        original(original_ids, counts, placeholder_token_id)
        rebuilt = []
        image_index = 0
        for index, token_id in enumerate(original_ids):
            if token_id != placeholder_token_id:
                rebuilt.append(token_id)
                continue
            if index == 0 or original_ids[index - 1] != self.IM_START_TOKEN_ID:
                rebuilt.append(self.IM_START_TOKEN_ID)
            rebuilt.extend([placeholder_token_id] * counts[image_index])
            if (
                index + 1 == len(original_ids)
                or original_ids[index + 1] != self.IM_END_TOKEN_ID
            ):
                rebuilt.append(self.IM_END_TOKEN_ID)
            image_index += 1
        return rebuilt

    patched._sglang_fl_minimax_m3_image_boundaries_patch = True
    MiniMaxM3VLProcessor._expand_input_ids = patched
    logger.info("Preserving MiniMax image boundaries in native token prompts")


def patch_minimax_m3_image_grid_cache_key() -> None:
    """Scope raw image cache identities to the grid used by vision RoPE.

    Equal flattened patches can describe different spatial grids. Feature-only
    hashes alias these images even when their placeholder counts match.
    """
    import torch

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalInputFormat,
    )
    from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor

    original = MiniMaxM3VLProcessor._postprocess_mm_items_before_transport
    if getattr(original, "_sglang_fl_minimax_m3_grid_cache_patch", False):
        return
    state_key = "_sglang_fl_minimax_m3_grid_cache"

    @wraps(original)
    def patched(self, mm_items, *, images):
        items = original(self, mm_items, images=images)
        for item in items:
            feature = item.feature
            grid = item.model_specific_data.get("image_grid_thw")
            if (
                item.modality != Modality.IMAGE
                or item.format != MultimodalInputFormat.NORMAL
                or item.precomputed_embeddings is not None
                or not isinstance(feature, torch.Tensor)
                or not feature.is_cpu
                or not isinstance(grid, torch.Tensor)
                or not grid.is_cpu
                or grid.ndim != 2
                or grid.shape[1] != 3
            ):
                continue
            item.set_pad_value()
            if item.hash is None:
                continue
            previous = item.model_specific_data.get(state_key)
            base_hash = (
                previous["base"]
                if isinstance(previous, dict) and item.hash == previous.get("scoped")
                else item.hash
            )
            payload = {
                "version": "minimax-m3-image-grid-v1",
                "feature_hash": base_hash,
                "feature_dtype": str(feature.dtype),
                "feature_shape": list(feature.shape),
                "image_grid_thw": grid.tolist(),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).digest()
            scoped_hash = int.from_bytes(digest[:8], "big")
            item.set_hash(scoped_hash)
            item.model_specific_data[state_key] = {
                "base": base_hash,
                "scoped": scoped_hash,
            }
        return items

    patched._sglang_fl_minimax_m3_grid_cache_patch = True
    MiniMaxM3VLProcessor._postprocess_mm_items_before_transport = patched
    logger.info("Including MiniMax image grids in multimodal cache identities")
