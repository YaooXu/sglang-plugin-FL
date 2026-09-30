"""Opt-in HCCL AIV execution for the SGLang TP communication domain.

Author: liyan <yan.l@pku.edu.cn>
"""

import functools
import logging
import os

logger = logging.getLogger(__name__)


def patch_ascend_tp_hccl_options():
    if os.environ.get("SGLANG_FL_ASCEND_TP_AIV") != "1":
        return

    import torch_npu
    from sglang.srt.distributed import parallel_state

    original = parallel_state.get_torch_distributed_pg_options
    if getattr(original, "_sglang_fl_tp_aiv", False):
        return

    @functools.wraps(original)
    def options(group_name=None):
        result = original(group_name)
        if group_name == "tp" and parallel_state._is_npu:
            if result is None:
                result = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            # CANN 9.0 HcclOpExpansionMode: 3 selects AIV for this domain.
            config = dict(result.hccl_config)
            config["hccl_op_expansion_mode"] = 3
            result.hccl_config = config
            logger.warning("Requested HCCL AIV execution for TP domain")
        return result

    options._sglang_fl_tp_aiv = True
    parallel_state.get_torch_distributed_pg_options = options
