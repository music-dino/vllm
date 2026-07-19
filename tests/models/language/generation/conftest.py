# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.fixture(scope="package", autouse=True)
def rocm_sdp_backend():
    """Disable Flash/MemEfficient SDP on ROCm for the generation test package.

    HF Transformers has accuracy issues with the flash and mem-efficient SDP
    backends on ROCm, so the math backend is forced instead.
    See https://github.com/vllm-project/vllm/issues/30167

    Package-scoped because the SDP backend is process-global torch state shared
    by every test here; set once for the package and restored afterwards. It is
    an autouse fixture rather than a session/config hook so that it runs
    regardless of how the suite is invoked -- fixture discovery is path-based,
    unlike ``pytest_sessionstart`` which is silently skipped for conftests
    loaded during parent-directory collection -- and, because fixture search is
    upward-only, it cannot leak into sibling test directories.
    """
    if not current_platform.is_rocm():
        yield
        return

    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math = torch.backends.cuda.math_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(prev_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_mem)
        torch.backends.cuda.enable_math_sdp(prev_math)


@pytest.fixture(autouse=True)
def rocm_skinny_gemm_env(monkeypatch):
    """Disable skinny GEMM on ROCm to avoid non-deterministic results from the
    wvSplitKrc kernel's atomic reductions.
    See https://github.com/vllm-project/vllm/pull/33493#issuecomment-3906083975

    Set per test via ``monkeypatch`` (function-scoped) so it is applied before
    each test's vLLM subprocess launches and automatically restored afterwards.
    """
    if current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "0")
    yield
