"""Opt-in graph/sampling completion diagnostics for Ascend.

Author: liyan <yan.l@pku.edu.cn>
"""

import functools
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)
_state = {}
_lock = threading.Lock()
_heartbeat_started = False


def _heartbeat():
    while True:
        time.sleep(5)
        with _lock:
            state = dict(_state)
        state["phase_age_s"] = round(time.monotonic() - state.pop("since"), 3)
        logger.warning("FL_NPU_STAGE %s", state)


def _stage(phase, **metadata):
    global _heartbeat_started
    with _lock:
        _state.update(metadata, phase=phase, since=time.monotonic())
        if not _heartbeat_started:
            threading.Thread(target=_heartbeat, daemon=True).start()
            _heartbeat_started = True


def patch_ascend_graph_stages():
    """Preserve execution order; fence only when diagnostics are enabled."""
    if os.environ.get("SGLANG_FL_ASCEND_STAGE_DIAGNOSTICS") != "1":
        return

    import torch
    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )
    from sglang.srt.layers import sampler

    if getattr(NPUCudaGraphBackend, "_sglang_fl_stage_diagnostics", False):
        return
    original_replay = NPUCudaGraphBackend.replay_with_input_update
    original_sync = sampler.Sampler._sync_token_ids_across_tp

    @functools.wraps(original_replay)
    def replay(self, shape_key, *args, **kwargs):
        step = getattr(self, "_sglang_fl_diagnostic_step", 0) + 1
        self._sglang_fl_diagnostic_step = step
        _stage(
            "graph_submit",
            rank=self._device_id,
            graph_step=step,
            last_graph_bucket=repr(shape_key),
            current_forward_graph=True,
            graph_stream=str(torch.npu.current_stream()),
        )
        result = original_replay(self, shape_key, *args, **kwargs)
        stream = torch.npu.current_stream()
        _stage("graph_wait", graph_stream=str(stream))
        stream.synchronize()
        _stage("graph_done")
        return result

    @functools.wraps(original_sync)
    def sync_tokens(self, batch_next_token_ids, sampling_info):
        if batch_next_token_ids.device.type != "npu":
            return original_sync(self, batch_next_token_ids, sampling_info)
        step = getattr(self, "_sglang_fl_diagnostic_step", 0) + 1
        self._sglang_fl_diagnostic_step = step
        with _lock:
            graph_step = _state.get("graph_step", 0)
        used_graph = graph_step != getattr(self, "_sglang_fl_graph_seen", 0)
        self._sglang_fl_graph_seen = graph_step
        stream = torch.npu.current_stream()
        _stage(
            "sampler_compute_wait",
            rank=batch_next_token_ids.device.index,
            sampler_step=step,
            token_shape=tuple(batch_next_token_ids.shape),
            token_dtype=str(batch_next_token_ids.dtype),
            current_forward_graph=used_graph,
            sampler_stream=str(stream),
        )
        stream.synchronize()
        _stage("token_sync_call")
        result = original_sync(self, batch_next_token_ids, sampling_info)
        _stage("token_sync_wait")
        stream.synchronize()
        _stage("token_sync_done")
        return result

    NPUCudaGraphBackend.replay_with_input_update = replay
    NPUCudaGraphBackend._sglang_fl_stage_diagnostics = True
    sampler.Sampler._sync_token_ids_across_tp = sync_tokens
    logger.warning(
        "Enabled FL NPU stage diagnostics; SYNC_TOKEN_IDS_ACROSS_TP=%s",
        sampler.SYNC_TOKEN_IDS_ACROSS_TP,
    )
