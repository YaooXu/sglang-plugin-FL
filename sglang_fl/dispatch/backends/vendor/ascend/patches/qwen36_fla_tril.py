# Copyright 2026 FlagOS Contributors
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

"""Drop the single-segment ``cu_seqlens`` from ``solve_tril`` (SGLANG_FL_FLA_TRIL_SINGLE=1).

Measured on the installed ``sgl_kernel_npu`` FLA kernels, with identical inputs
and identical program counts (T=16384, H=24, BT=64, NT16=14, NT64=256):

    kernel                                        IS_VARLEN=False   IS_VARLEN=True
    solve_tril_16x16_kernel_paral_v3                    392.8 us        2500.7 us
    merge_16x16_to_64x64_inverse_kernel_reorder         1492.8 us      453282.6 us
    total                                               1885.6 us       455783.3 us

i.e. the merge kernel's VARLEN branch is **304x slower** in the same process,
same tensors, same grid, and the outputs are bitwise identical.  The host side
is not to blame: ``prepare_chunk_indices`` costs 5.6 us.

The GDN prefill path does pass it (``gdn_triton.GDNKernelDispatcher.extend`` ->
``chunk_gated_delta_rule(..., cu_seqlens=query_start_loc)``), so this patch
supplies the missing condition: when ``cu_seqlens`` describes exactly one
segment ``[0, T]`` covering the whole ``A`` tensor, the VARLEN and non-VARLEN
code paths are provably the same computation (verified ``torch.equal`` on
T=2048/8192/16384), so ``cu_seqlens=None`` is semantically identical and picks
the fast branch.

Multi-segment / genuinely padded batches are left completely untouched.
"""

from __future__ import annotations

import functools
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MARK = "_sglang_fl_fla_tril_single"
_ENV = "SGLANG_FL_FLA_TRIL_SINGLE"
# TRIL_ALL additionally drops cu_seqlens at the orchestrator so kkt /
# recompute_w_u / fwd_h / chunk_o / local_cumsum all take the fast branch.
_ENV_ALL = "SGLANG_FL_FLA_TRIL_ALL"
_STATS = {"calls": 0, "dropped": 0, "cache_hits": 0}
_CACHE: dict = {}


def _enabled() -> bool:
    return os.getenv(_ENV, "0") == "1"


def _all_enabled() -> bool:
    return os.getenv(_ENV_ALL, "0") == "1"


def _single_full_segment(cu_seqlens, total: int) -> bool:
    """True iff cu_seqlens is exactly one segment [0, total]."""
    try:
        if cu_seqlens is None or cu_seqlens.numel() != 2:
            return False
        key = (cu_seqlens.data_ptr(), cu_seqlens.numel(), total)
        hit = _CACHE.get(key)
        if hit is not None:
            _STATS["cache_hits"] += 1
            return hit
        lo, hi = cu_seqlens.detach().to("cpu").tolist()
        ok = (int(lo) == 0 and int(hi) == int(total))
        _CACHE[key] = ok
        return ok
    except Exception:  # noqa: BLE001 - never break the model
        return False


def _fwd_wrapper(original):
    """Drop a single-full-segment cu_seqlens before it fans out to 5 kernels."""

    def wrapped(q, k, v, g, beta, scale, initial_state, output_final_state,
                cu_seqlens=None):
        if not _all_enabled():
            return original(q, k, v, g, beta, scale, initial_state,
                            output_final_state, cu_seqlens)
        _STATS["calls"] += 1
        if cu_seqlens is not None and _single_full_segment(cu_seqlens, q.shape[1]):
            _STATS["dropped"] += 1
            cu_seqlens = None
        return original(q, k, v, g, beta, scale, initial_state,
                        output_final_state, cu_seqlens)

    setattr(wrapped, _MARK, True)
    return wrapped


@triton.jit(do_not_specialize=["T"])
def _wu_novarlen_fix_kernel(
    k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Byte-for-byte the installed kernel plus the missing ``i_t = i_t_o``.

    The installed ``recompute_w_u_fwd_kernel_npu_kernel`` omits ``i_t`` in its
    non-VARLEN branch and therefore cannot compile when ``cu_seqlens is None``
    (``NameError('i_t is not defined')``).  Same math, same grid ``(NT, B)``,
    same inner ``for i_bh in range(H)`` loop, same dtypes.
    """
    T_max = T
    i_t_o, _ = tl.program_id(0), tl.program_id(1)
    for i_bh in range(H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_o * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_o * 2 + 1).to(tl.int32),
            )
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int32),
                tl.load(cu_seqlens + i_n + 1).to(tl.int32),
            )
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            i_t = i_t_o

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt * 1
        b_A = tl.load(ptr_A, mask=mask_t[:, None], other=0.0).to(tl.float32)

        b_g = tl.exp(
            tl.load(g + bos + i_h * T_max + global_offs_t, mask=mask_t, other=0.0)
        ).to(tl.float32)
        b_beta = tl.load(
            beta + bos + i_h * T_max + global_offs_t, mask=mask_t, other=0.0
        ).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = mask_t[:, None] & (offs_v < V)
            b_v = tl.load(
                v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v,
                mask=mask_v, other=0.0,
            ).to(tl.float32)
            b_u = tl.dot(b_A, b_v * b_beta[:, None], allow_tf32=False)
            tl.store(
                u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v,
                b_u.to(u.dtype.element_ty), mask=mask_v,
            )

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = mask_t[:, None] & (offs_k < K)
            b_k = tl.load(
                k + (bos * Hg + i_h // (H // Hg)) * K
                + offs_t_2d * (Hg * K) + offs_k,
                mask=mask_k, other=0.0,
            ).to(tl.float32)
            b_w = tl.dot(b_A, b_k * b_beta[:, None] * b_g[:, None])
            tl.store(
                w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k,
                b_w.to(w.dtype.element_ty), mask=mask_k,
            )


def _wu_fixed(original):
    def wrapped(k, v, beta, g_cumsum, A, cu_seqlens, **kwargs):
        # Only the non-VARLEN call needs the bug-fixed copy: the installed
        # kernel cannot compile with cu_seqlens=None.  Multi-sequence batches
        # (cu_seqlens is not None) keep the installed kernel, which is what the
        # 1K/4K cases use -- replacing it there measured -2%..-5%.
        if cu_seqlens is not None:
            return original(k, v, beta, g_cumsum, A, cu_seqlens)
        try:
            from sgl_kernel_npu.fla.utils import prepare_chunk_indices

            B, T, Hg, K, V = *k.shape, v.shape[-1]
            H = v.shape[-2]
            BT = A.shape[-1]
            ci = None
            u = torch.empty_like(v)
            w = k.new_empty(B, T, H, K)
            beta_t = beta.transpose(1, 2).contiguous()
            g_t = g_cumsum.transpose(1, 2).contiguous()
            _wu_novarlen_fix_kernel[(triton.cdiv(T, BT), B)](
                k=k, v=v, beta=beta_t, w=w, u=u, A=A, g=g_t,
                cu_seqlens=cu_seqlens, chunk_indices=ci, T=T,
                H=H, Hg=Hg, K=K, V=V, BT=BT, BK=128, BV=128,
                IS_VARLEN=cu_seqlens is not None,
                num_warps=4, num_stages=3,
            )
            return w, u
        except Exception:  # noqa: BLE001
            logger.exception("FLA fixed recompute_w_u failed; falling back")
            return original(k, v, beta, g_cumsum, A, cu_seqlens)

    setattr(wrapped, _MARK, True)
    return wrapped


def _wrapper(original):
    def wrapped(A, cu_seqlens=None, output_dtype=torch.float, **kwargs):
        if not _enabled():
            return original(A, cu_seqlens=cu_seqlens, output_dtype=output_dtype, **kwargs)
        _STATS["calls"] += 1
        if cu_seqlens is not None and _single_full_segment(cu_seqlens, A.shape[1]):
            _STATS["dropped"] += 1
            cu_seqlens = None
        return original(A, cu_seqlens=cu_seqlens, output_dtype=output_dtype, **kwargs)

    setattr(wrapped, _MARK, True)
    return wrapped


def patch_qwen36_fla_tril_single() -> bool:
    if not (_enabled() or _all_enabled()):
        return False
    try:
        from sgl_kernel_npu.fla import chunk as fla_chunk
        from sgl_kernel_npu.fla import solve_tril as fla_tril
    except ImportError:
        logger.warning("Skipping FLA tril single-segment patch: module unavailable")
        return False

    pairs = [
        (fla_chunk, "solve_tril"),
        (fla_tril, "solve_tril_npu"),
    ]
    if _all_enabled():
        # One shot at the orchestrator drops cu_seqlens for *all* downstream
        # kernels; it needs the bug-fixed recompute_w_u below to compile.
        pairs.append((fla_chunk, "chunk_gated_delta_rule_fwd"))
    for mod, attr in pairs:
        cur = getattr(mod, attr, None)
        if cur is None or getattr(cur, _MARK, False):
            continue
        if attr == "chunk_gated_delta_rule_fwd":
            setattr(mod, attr, _fwd_wrapper(cur))
        else:
            setattr(mod, attr, _wrapper(cur))

    if _all_enabled():
        try:
            from sgl_kernel_npu.fla import wy_fast as fla_wy

            cur = getattr(fla_chunk, "recompute_w_u_fwd", None)
            if cur is not None and not getattr(cur, _MARK, False):
                fixed = _wu_fixed(cur)
                setattr(fla_chunk, "recompute_w_u_fwd", fixed)
                setattr(fla_wy, "recompute_w_u_fwd_npu", fixed)
        except ImportError:
            pass

    import atexit

    atexit.register(
        lambda: logger.info(
            "FLA_TRIL_SINGLE_STATS calls=%d dropped=%d cache_hits=%d",
            _STATS["calls"],
            _STATS["dropped"],
            _STATS["cache_hits"],
        )
    )
    logger.info(
        "Installed FLA single-segment fast path: TRIL_SINGLE=%s TRIL_ALL=%s",
        _enabled(),
        _all_enabled(),
    )
    return True
