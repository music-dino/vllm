# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest

from vllm import LLM
from vllm.utils.mem_constants import GiB_bytes

from ..utils import create_new_process_for_each_test
from .registry import (
    _TRANSFORMERS_BACKEND_MODELS,
    AUTO_EXAMPLE_MODELS,
    HF_EXAMPLE_MODELS,
    HfExampleModels,
)
from .utils import dummy_hf_overrides

# This minimal list of model architectures is smaller than the total list of
# supported models. The intention is that in the "typical" regression testing
# scenario, we only test initializing these models. This subset was chosen
# to include representative examples of model varieties/workloads (conditional
# generation, sequence classification, causal LM, ranking, chat, reward model,
# multimodal, geospatial, voice, embedding, MTP)
MINIMAL_MODEL_ARCH_LIST = [
    "LlavaForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "BertForSequenceClassification",
    "Gemma3nForCausalLM",
    "JinaVLForRanking",
    "InternVLChatModel",
    "InternLM2ForRewardModel",
    "TransformersMultiModalForCausalLM",
    "PrithviGeoSpatialMAE",
    "UltravoxModel",
    "DeepSeekMTPModel",
    "XLMRobertaModel",
]

# This list is the complement of the minimal list above. The intention is that
# this list of models is only tested in a "special case" i.e. most PRs should
# not test these models
OTHER_MODEL_ARCH_LIST = set(HF_EXAMPLE_MODELS.get_supported_archs()) - set(
    MINIMAL_MODEL_ARCH_LIST
)


@create_new_process_for_each_test()
def can_initialize(
    model_arch: str, monkeypatch: pytest.MonkeyPatch, EXAMPLE_MODELS: HfExampleModels
):
    """Run each test in a separate process for isolation.

    Memory profiling and model warmup/cudagraph capture (and the model forward
    passes they require) are skipped by setting
    ``VLLM_TEST_KV_CACHE_MEMORY_BYTES``, which the engine core reads to use a
    fixed KV cache memory budget. Using an env var (rather than monkeypatching
    ``EngineCore._initialize_kv_caches``) makes this work whether the engine
    core is forked or spawned -- e.g. on ROCm/XPU where spawn is forced.
    """

    model_info = EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    hf_overrides_fn = partial(
        dummy_hf_overrides,
        model_arch=model_arch,
        exist_overrides=model_info.hf_overrides,
        use_original_num_layers=getattr(model_info, "use_original_num_layers", False),
    )

    if model_arch == "MoonshotKimiaForCausalLM":
        pytest.skip(
            "Kimi-Audio requires SpeechToTextConfig "
            "which is not configured in test environment"
        )

    if model_arch in ("PrithviGeoSpatialMAE", "Terratorch"):
        import importlib.util

        if importlib.util.find_spec("terratorch") is None:
            pytest.skip(
                "terratorch is not installed; "
                "temporarily skipped while PyPI has `lightning` quarantined "
                "(see #41376)"
            )

    if model_arch in ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]:
        from vllm.platforms import current_platform

        capability = current_platform.get_device_capability()
        if capability and capability.major < 9:
            pytest.skip(
                f"DeepseekV32 requires Hopper (9.0+) or Blackwell (10.0+) "
                f"for FLASHMLA_SPARSE backend. Current device has compute "
                f"capability {capability.major}.{capability.minor}"
            )

    with monkeypatch.context() as m:
        # Skip memory profiling (and the model forward pass it requires) by
        # giving the engine core a fixed KV cache memory budget via env var.
        # Env-var based so it survives engine-core spawn (e.g. on ROCm/XPU).
        m.setenv("VLLM_TEST_KV_CACHE_MEMORY_BYTES", str(10 * GiB_bytes))

        # FIXME: A hack to bypass FA3 assertion because our CI's L4 GPU
        # has cc==8.9 which hasn't supported FA3 yet. Remove this hack when
        # L4 supports FA3.
        # Step1ForCausalLM requires TRITON_ATTN for use_alibi_sqrt support.
        attention_config = (
            {"backend": "TRITON_ATTN"}
            if model_arch in ("GptOssForCausalLM", "Step1ForCausalLM")
            else None
        )
        if model_arch == "WhisperForConditionalGeneration":
            m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        kwargs = {}
        if not model_info.enable_prefix_caching:
            kwargs["enable_prefix_caching"] = False

        LLM(
            model_info.default,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            revision=model_info.revision,
            enforce_eager=model_info.enforce_eager,
            skip_tokenizer_init=model_info.require_embed_inputs,
            enable_prompt_embeds=model_info.require_embed_inputs,
            enable_mm_embeds=model_info.require_embed_inputs,
            dtype=model_info.dtype,
            speculative_config={
                "model": model_info.speculative_model,
                "method": model_info.speculative_method,
                "num_speculative_tokens": 1,
            }
            if model_info.speculative_model
            else None,
            trust_remote_code=model_info.trust_remote_code,
            max_model_len=model_info.max_model_len,
            max_num_batched_tokens=model_info.max_num_batched_tokens,
            # these tests seem to produce leftover memory
            gpu_memory_utilization=0.80,
            load_format="dummy",
            model_impl="transformers"
            if model_arch in _TRANSFORMERS_BACKEND_MODELS
            else "vllm",
            hf_overrides=hf_overrides_fn,
            max_num_seqs=model_info.max_num_seqs,
            attention_config=attention_config,
            **kwargs,
        )


@pytest.mark.parametrize("model_arch", MINIMAL_MODEL_ARCH_LIST)
def test_can_initialize_small_subset(model_arch: str, monkeypatch: pytest.MonkeyPatch):
    """Test initializing small subset of supported models"""
    can_initialize(model_arch, monkeypatch, HF_EXAMPLE_MODELS)


@pytest.mark.parametrize("model_arch", OTHER_MODEL_ARCH_LIST)
def test_can_initialize_large_subset(model_arch: str, monkeypatch: pytest.MonkeyPatch):
    """Test initializing large subset of supported models

    This test covers the complement of the tests covered in the "small subset"
    test.
    """
    can_initialize(model_arch, monkeypatch, HF_EXAMPLE_MODELS)


@pytest.mark.parametrize("model_arch", AUTO_EXAMPLE_MODELS.get_supported_archs())
def test_implicit_converted_models(model_arch: str, monkeypatch: pytest.MonkeyPatch):
    can_initialize(model_arch, monkeypatch, AUTO_EXAMPLE_MODELS)
