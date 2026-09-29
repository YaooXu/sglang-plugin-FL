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
