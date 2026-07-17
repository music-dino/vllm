# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import os
import warnings

import torch

from vllm.platforms import current_platform


def _install_sdp_matmul_tracers():
    """TEMP DIAGNOSTIC: trace who resets SDP/matmul state during collection."""
    import sys
    import traceback

    b = torch.backends

    def _trace(label, predicate):
        def emit(*args):
            if predicate(args):
                print(
                    "\n=== TORCH_MUTATION_TRACE: %s%r ===" % (label, args),
                    file=sys.stderr,
                )
                traceback.print_stack(file=sys.stderr)
                print("=== END TORCH_MUTATION_TRACE ===", file=sys.stderr)

        return emit

    # Patch the C-LEVEL setters so we catch callers that bypass the Python
    # wrappers (torch.backends.cuda.enable_*). These are what actually mutate
    # the at::Context flags.
    c_targets = {
        "_set_sdp_use_flash": lambda a: bool(a and a[0]),
        "_set_sdp_use_mem_efficient": lambda a: bool(a and a[0]),
        "_set_float32_matmul_precision": lambda a: bool(a and a[0] == "highest"),
        "_set_cublas_allow_tf32": lambda a: bool(a and a[0] is False),
    }
    for cname, pred in c_targets.items():
        orig = getattr(torch._C, cname, None)
        if orig is None:
            continue
        emit = _trace("torch._C." + cname, pred)

        def make(o, e):
            def wrapped(*a, **k):
                e(*a)
                return o(*a, **k)

            return wrapped

        setattr(torch._C, cname, make(orig, emit))

    # Trace the FIRST lazy HIP/CUDA init (suspected flag-reset trigger).
    orig_lazy = torch.cuda._lazy_init
    _fired = {"done": False}

    def lazy(*a, **k):
        if not _fired["done"] and not torch.cuda.is_initialized():
            _fired["done"] = True
            print("\n=== TORCH_LAZY_INIT_TRACE (first HIP init) ===", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
            print("=== END TORCH_LAZY_INIT_TRACE ===", file=sys.stderr)
        return orig_lazy(*a, **k)

    torch.cuda._lazy_init = lazy


def pytest_configure(config):
    """Early ROCm configuration that must happen before test collection."""
    if not current_platform.is_rocm():
        return

    _install_sdp_matmul_tracers()

    # Disable skinny GEMM on ROCm to avoid non-deterministic results
    # from atomic reductions in wvSplitKrc kernel.
    # See: https://github.com/vllm-project/vllm/pull/33493#issuecomment-3906083975
    os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
    warnings.warn(
        "ROCm: Set VLLM_ROCM_USE_SKINNY_GEMM=0 to avoid non-deterministic "
        "results from skinny GEMM atomic reductions",
        UserWarning,
        stacklevel=1,
    )


def pytest_collection_finish(session):
    """TEMP DIAGNOSTIC: dump global torch numeric state after all imports."""
    import sys

    b = torch.backends
    state = {
        "cuda_initialized": torch.cuda.is_initialized(),
        "flash_sdp": b.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": b.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": b.cuda.math_sdp_enabled(),
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": b.cuda.matmul.allow_tf32,
    }
    print("\n=== TORCH_STATE_PROBE (collected=%d) ===" % len(session.items),
          file=sys.stderr)
    for k, v in state.items():
        print("TORCH_STATE_PROBE %s = %r" % (k, v), file=sys.stderr)
    print("=== END TORCH_STATE_PROBE ===", file=sys.stderr)


def pytest_sessionstart(session):
    """Configure ROCm-specific settings before test session starts."""
    if not current_platform.is_rocm():
        return

    # Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    # TODO: Remove once ROCm SDP accuracy issues are resolved on HuggingFace
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("high")
    warnings.warn(
        "ROCm: Disabled flash_sdp and mem_efficient_sdp, enabled math_sdp "
        "to avoid HuggingFace Transformers accuracy issues",
        UserWarning,
        stacklevel=1,
    )
