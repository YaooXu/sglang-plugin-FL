"""Implement the SGLang OOT profiler hooks for Ascend.

Author: liyan <yan.l@pku.edu.cn>
"""

import functools


def patch_ascend_profiler_activity():
    from sglang_fl.platform import PlatformFL

    if getattr(PlatformFL, "_sglang_fl_ascend_profiler", False):
        return
    original_name = PlatformFL.get_torch_profiler_activity_str
    original_activity = PlatformFL.get_torch_profiler_activity

    @functools.wraps(original_name)
    def activity_name(self):
        if self._device_type == "npu":
            # SGLang maps this existing torch activity and its GPU request alias
            # to NPU in SchedulerProfilerManager on Ascend.
            return "CUDA"
        return original_name(self)

    @functools.wraps(original_activity)
    def activity(self):
        if self._device_type == "npu":
            import torch_npu

            return torch_npu.profiler.ProfilerActivity.NPU
        return original_activity(self)

    PlatformFL.get_torch_profiler_activity_str = activity_name
    PlatformFL.get_torch_profiler_activity = activity
    PlatformFL._sglang_fl_ascend_profiler = True
