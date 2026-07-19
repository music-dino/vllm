# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import os
import warnings

import torch

from vllm.platforms import current_platform


def pytest_configure(config):
    """Early ROCm configuration that must happen before test collection.

    Both the skinny-GEMM env var and the SDP settings are applied here
    (not in ``pytest_sessionstart``). ``pytest_sessionstart`` only fires for
    *initial* conftests on the CLI arg path; when the suite is collected via a
    parent directory (e.g. ``pytest models/language``), this conftest is loaded
    during collection recursion, after ``pytest_sessionstart`` has already run,
    so a sessionstart hook here would be silently skipped. ``pytest_configure``
    runs for late-loaded conftests too.
    """
    if not current_platform.is_rocm():
        return

    # Disable skinny GEMM on ROCm to avoid non-deterministic results
    # from atomic reductions in wvSplitKrc kernel.
    # See: https://github.com/vllm-project/vllm/pull/33493#issuecomment-3906083975
    os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"

    # Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    # TODO: Remove once ROCm SDP accuracy issues are resolved on HuggingFace
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    warnings.warn(
        "ROCm: Set VLLM_ROCM_USE_SKINNY_GEMM=0 and disabled flash/mem-efficient "
        "SDP (enabled math SDP) to avoid non-deterministic and HuggingFace "
        "Transformers accuracy issues",
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
