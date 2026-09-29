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

"""Route MiniMax-M3 through its Ascend sparse KV pool on the OOT platform."""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def patch_minimax_m3_kv_pool() -> None:
    """Preserve SGLang's MiniMax-specific pool behind the OOT platform branch."""
    from sglang.srt.configs.model_config import is_minimax_sparse
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.runtime_context import get_exec

    original_build = KVCacheConfigurator._build_oot_mha_kv_pool
    if getattr(original_build, "_sglang_fl_minimax_m3_kv_pool_patch", False):
        return

    @wraps(original_build)
    def patched_build(self, *, max_total_num_tokens: int):
        device_type = str(self.device).split(":", 1)[0]
        if (
            device_type == "npu"
            and get_exec().kernel.attention_backend == "ascend"
            and is_minimax_sparse(self.model_config.hf_config)
        ):
            logger.info("Using Ascend MiniMax-M3 sparse KV pool on OOT platform")
            return self._build_ascend_minimax_sparse_kv_pool(
                max_total_num_tokens=max_total_num_tokens
            )
        return original_build(self, max_total_num_tokens=max_total_num_tokens)

    patched_build._sglang_fl_minimax_m3_kv_pool_patch = True
    KVCacheConfigurator._build_oot_mha_kv_pool = patched_build
