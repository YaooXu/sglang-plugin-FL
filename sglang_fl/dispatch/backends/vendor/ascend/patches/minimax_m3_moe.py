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

"""Preserve MiniMax-M3's SwiGLU formula in the Ascend MoE runner.

The generic non-DeepEP runner picks ``NPUSwigluStepAndMul`` for ``silu`` with a
clamp limit. That activation omits MiniMax's sigmoid alpha and ``up + 1``. The
existing grouped OAI kernel handles non-interleaved ``[gate | up]`` weights and
is already used by the runner's DeepEP branch with these same parameters. The
same formula must be applied before requantizing M3.1's W8A8 experts.
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def patch_minimax_m3_moe_activation() -> None:
    """Patch MiniMax-shaped BF16 and W8A8 Ascend expert activations."""
    from sglang.srt.hardware_backend.npu.moe.activation import (
        NPUSwigluDeepEPKernel,
        NPUSwigluQuant,
        NPUSwigluStepAndMul,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUUnquantMoEMethod,
        NPUW8A8Int8MoEMethod,
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
        kernel = getattr(layer, "w2_kernel", None)
        is_w8a8 = isinstance(kernel, NPUW8A8Int8MoEMethod)
        if (
            config.activation != "silu"
            or not config.is_gated
            or config.gate_up_interleaved
            or config.gemm1_alpha != 1.702
            or config.gemm1_clamp_limit != 7
            or config.gemm1_beta != 1.0
            or layer is None
            or not getattr(layer, "layer_name", "").endswith(".experts")
            or not (
                (is_w8a8 and isinstance(self.activation, NPUSwigluQuant))
                or (
                    isinstance(kernel, NPUUnquantMoEMethod)
                    and isinstance(self.activation, NPUSwigluStepAndMul)
                )
            )
        ):
            return

        # The runner forwards group_list/group_list_type for this activation
        # class, so the kernel processes the same routed rows as the GMMs.
        self.activation = NPUSwigluDeepEPKernel(
            need_quant=is_w8a8,
            alpha=config.gemm1_alpha,
            limit=config.gemm1_clamp_limit,
        )
        patched_count += 1
        if patched_count in (1, 57):
            logger.info(
                "Applied MiniMax Ascend MoE SwiGLU activation patch to %d runner(s) (W8A8=%s)",
                patched_count,
                is_w8a8,
            )

    patched_init._sglang_fl_minimax_m3_activation_patch = True
    AscendRunnerCore.__init__ = patched_init


def patch_minimax_m3_w8a8_moe_offsets() -> None:
    """Supply absent offsets as None for symmetric compressed-tensors INT8 MoE.

    The current SGLang scheme creates only weights and scales, then reads both
    offset attributes unconditionally while building AscendQuantInfo.
    """
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8_moe import (
        NPUCompressedTensorsW8A8Int8DynamicMoE,
    )

    original_create_weights = NPUCompressedTensorsW8A8Int8DynamicMoE.create_weights
    if getattr(original_create_weights, "_sglang_fl_minimax_m3_offsets_patch", False):
        return

    @wraps(original_create_weights)
    def patched_create_weights(self, layer, *args, **kwargs):
        result = original_create_weights(self, layer, *args, **kwargs)
        for name in ("w13_weight_offset", "w2_weight_offset"):
            if not hasattr(layer, name):
                setattr(layer, name, None)
        return result

    patched_create_weights._sglang_fl_minimax_m3_offsets_patch = True
    NPUCompressedTensorsW8A8Int8DynamicMoE.create_weights = patched_create_weights


def patch_minimax_m3_w8a8_quant_targets() -> None:
    """Map the checkpoint's separate projections to SGLang's fused layers.

    M3.1's compressed-tensors config lists each q/k/v and gate/up projection,
    but does not include ``packed_modules_mapping``. The index projection has
    only q/k, so it must be matched before the shorter ``qkv_proj`` suffix.
    """
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    original_from_config = CompressedTensorsConfig.from_config.__func__
    if getattr(original_from_config, "_sglang_fl_minimax_m3_targets_patch", False):
        return

    @classmethod
    @wraps(original_from_config)
    def patched_from_config(cls, config):
        targets = (
            target
            for group in config.get("config_groups", {}).values()
            for target in group.get("targets", ())
        )
        if config.get("quant_method") == "compressed-tensors" and any(
            ".self_attn.index_q_proj" in target for target in targets
        ):
            config = dict(config)
            packed = {
                "index_qkv_proj": ["index_q_proj", "index_k_proj"],
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                "gate_up_proj": ["gate_proj", "up_proj"],
            }
            packed.update(config.get("packed_modules_mapping") or {})
            config["packed_modules_mapping"] = packed
            logger.info("Installed MiniMax-M3.1 W8A8 fused projection target mapping")
        return original_from_config(cls, config)

    patched_from_config.__func__._sglang_fl_minimax_m3_targets_patch = True
    CompressedTensorsConfig.from_config = patched_from_config
