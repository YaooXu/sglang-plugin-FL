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

"""Avoid dynamic boolean index_put in the Ascend PyTorch sampler.

The native fallback uses boolean tensor assignment for its three probability
filters. NPU watchdog traces can stop in the dynamic index_put dispatch. Use
shape-preserving masked_fill_ on NPU while retaining the upstream filter order,
seeded sampling, and optional sampling-mask outputs.
"""

import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def patch_ascend_torch_sampling() -> None:
    from sglang.srt.layers import sampler

    original = sampler.top_k_top_p_min_p_sampling_from_probs_torch
    if getattr(original, "_sglang_fl_npu_masked_sampling", False):
        return
    if "return_filtered_probs" not in inspect.signature(original).parameters:
        logger.warning("Skipping Ascend sampling patch: unsupported sampler signature")
        return

    @wraps(original)
    def patched(
        probs,
        top_ks,
        top_ps,
        min_ps,
        need_min_p_sampling,
        sampling_seed,
        positions,
        *,
        return_filtered_probs=False,
    ):
        if probs.device.type != "npu":
            return original(
                probs,
                top_ks,
                top_ps,
                min_ps,
                need_min_p_sampling,
                sampling_seed,
                positions,
                return_filtered_probs=return_filtered_probs,
            )

        torch = sampler.torch
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
        # Preserve upstream ordering: cumsum precedes the top-k filter.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_k_mask = torch.arange(probs.shape[-1], device=probs.device).view(
            1, -1
        ) >= top_ks.view(-1, 1)
        probs_sort.masked_fill_(top_k_mask, 0.0)
        probs_sort.masked_fill_((probs_sum - probs_sort) > top_ps.view(-1, 1), 0.0)
        if need_min_p_sampling:
            assert sampling_seed is None, (
                "With sampling seed, multinomial_with_seed will provide wrong results"
            )
            min_p_thresholds = probs_sort[:, 0] * min_ps
            probs_sort.masked_fill_(probs_sort < min_p_thresholds.view(-1, 1), 0.0)

        if sampling_seed is None:
            sampled_index = torch.multinomial(probs_sort, num_samples=1)
        else:
            logprobs = probs_sort.to(torch.float64, copy=return_filtered_probs)
            if not return_filtered_probs:
                del probs_sort
            logprobs.log_()
            sampled_index = sampler.multinomial_with_seed(
                logprobs, sampling_seed, positions
            )

        if return_filtered_probs:
            selected_weight = torch.gather(probs_sort, 1, sampled_index).view(-1)
        probs_idx = probs_idx.to(torch.int32)
        batch_next_token_ids = torch.gather(probs_idx, 1, sampled_index).view(-1)
        if return_filtered_probs:
            return batch_next_token_ids, probs_sort, probs_idx, selected_weight
        return batch_next_token_ids

    patched._sglang_fl_npu_masked_sampling = True
    sampler.top_k_top_p_min_p_sampling_from_probs_torch = patched
    logger.info("Installed Ascend PyTorch sampling masked_fill patch")
