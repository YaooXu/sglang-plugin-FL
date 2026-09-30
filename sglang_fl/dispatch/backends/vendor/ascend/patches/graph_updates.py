"""Serialize Ascend graph updates and replay when explicitly requested.

Author: liyan <yan.l@pku.edu.cn>
"""

import functools
import logging
import os

logger = logging.getLogger(__name__)


def patch_ascend_graph_updates():
    """Keep task updates off the replay stream and propagate update failures."""
    if os.environ.get("SGLANG_FL_ASCEND_SERIAL_GRAPH_UPDATE") != "1":
        return

    import numpy as np
    import torch
    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )

    if getattr(NPUCudaGraphBackend, "_sglang_fl_serial_graph_update", False):
        return
    original = NPUCudaGraphBackend.replay_with_input_update
    diagnostics = os.environ.get("SGLANG_FL_ASCEND_STAGE_DIAGNOSTICS") == "1"

    def stage(phase, **metadata):
        if diagnostics:
            from .graph_stages import _stage

            _stage(phase, **metadata)

    @functools.wraps(original)
    def replay(
        self,
        shape_key,
        seq_lens,
        attr_name=None,
        attr_type=None,
        cpu_update_input=None,
    ):
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph = self._graphs[shape_key]
        mode = getattr(graph, "graph_dispatch_mode", None)
        update_stream = getattr(mode, "update_stream", None)
        if update_stream is None or not callable(
            getattr(update_stream, "synchronize", None)
        ):
            raise RuntimeError(
                "Serial NPU graph updates require the TorchNPU graph dispatch "
                "update_stream interface (verified on torch_npu 2.10.0)."
            )
        metadata = {}
        if diagnostics:
            lengths = seq_lens
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.tolist() if lengths.device.type == "cpu" else None
            if isinstance(lengths, (list, tuple)) and lengths:
                metadata = {
                    "seq_lens": list(lengths),
                    "seq_lens_min": min(lengths),
                    "seq_lens_max": max(lengths),
                }
        with torch.npu.device(self._device_id):
            # Retire the preceding replay before updating its recorded tasks.
            replay_stream = torch.npu.current_stream()
            stage("graph_previous_wait", **metadata)
            replay_stream.synchronize()
            stage("graph_update_call")
            graph.update(cpu_update_input=cpu_update_input)
            stage("graph_update_wait", update_stream=str(update_stream))
            update_stream.synchronize()
            stage("graph_replay_submit")
            graph.replay()
            stage("graph_wait")
            replay_stream.synchronize()
            stage("graph_done")
        return self._outputs[shape_key]

    NPUCudaGraphBackend.replay_with_input_update = replay
    NPUCudaGraphBackend._sglang_fl_serial_graph_update = True
    logger.warning("Enabled FL serial NPU graph update and replay completion")
