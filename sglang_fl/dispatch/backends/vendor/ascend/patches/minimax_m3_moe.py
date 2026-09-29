# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve MiniMax-M3's SwiGLU formula in the unquantized Ascend MoE runner.

The generic non-DeepEP runner picks ``NPUSwigluStepAndMul`` for ``silu`` with a
clamp limit. That activation omits MiniMax's sigmoid alpha and ``up + 1``. The
existing grouped OAI kernel handles non-interleaved ``[gate | up]`` weights and
is already used by the runner's DeepEP branch with these same parameters.
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def patch_minimax_m3_moe_activation() -> None:
    """Patch only the MiniMax-shaped, unquantized Ascend expert activation."""
    from sglang.srt.hardware_backend.npu.moe.activation import (
        NPUSwigluDeepEPKernel,
        NPUSwigluStepAndMul,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUUnquantMoEMethod,
    )
    from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore

    original_init = AscendRunnerCore.__init__
    if getattr(original_init, "_sglang_fl_minimax_m3_activation_patch", False):
        return

    patched_count = 0

    @wraps(original_init)
    def patched_init(self, config):
        nonlocal patched_count
        original_init(self, config)

        layer = config.layer
        if (
            config.activation != "silu"
            or not config.is_gated
            or config.gate_up_interleaved
            or config.gemm1_alpha != 1.702
            or config.gemm1_clamp_limit != 7
            or config.gemm1_beta != 1.0
            or layer is None
            or not getattr(layer, "layer_name", "").endswith(".experts")
            or not isinstance(layer.w2_kernel, NPUUnquantMoEMethod)
            or not isinstance(self.activation, NPUSwigluStepAndMul)
        ):
            return

        # The runner forwards group_list/group_list_type for this activation
        # class, so the kernel processes the same routed rows as the GMMs.
        self.activation = NPUSwigluDeepEPKernel(
            need_quant=False,
            alpha=config.gemm1_alpha,
            limit=config.gemm1_clamp_limit,
        )
        patched_count += 1
        if patched_count in (1, 57):
            logger.info(
                "Applied MiniMax-M3 Ascend MoE SwiGLU activation patch to %d runner(s)",
                patched_count,
            )

    patched_init._sglang_fl_minimax_m3_activation_patch = True
    AscendRunnerCore.__init__ = patched_init
