# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM preset: expand options={"vllm": True} into per-flag defaults.

Used by torchdynamo.compile and torchdynamo.vllm.compile_hooks to apply
vLLM-specific defaults when the caller opts into the preset. Lives in the
vllm/ subpackage so that the generic torchdynamo backend stays free of
vLLM-specific knowledge.
"""

import os
from typing import Optional, Any


def bool_opt(options, key: str, default: bool) -> bool:
    """Resolve a boolean plugin option with vLLM-preset fallback.

    Priority: options[key] > vLLM preset (if active) > default.
    Strings \"false\"/\"0\" are treated as False.

    Used by vllm/ code that needs preset-aware resolution. Generic
    torchdynamo callers should inline a plain
    ``bool(options and options.get(key, default))`` instead — they do not
    need the preset lookup.
    """
    if options is not None and key in options:
        v = options[key]
    else:
        if is_vllm_preset(options) and has_preset_flag(key):
            v = preset_flag(key)
        else:
            return default
    return bool(v) and str(v).lower() not in ("false", "0")

# Per-flag defaults expanded from options["vllm"]=True. Caller-supplied flags
# take priority over these (see _bool_opt).
_PRESET_FLAGS = {
    "unbind_affinity": True,
    "paged_attention": True,
    "pa_translate": True,
    "no_fallback": True,
    "fc_decompress": True,
    "dynamic_shapes": False,
}

# OV CPU-config defaults expanded from options["vllm"]=True. Caller-supplied
# config keys win.
_PRESET_CONFIG = {
    "KV_CACHE_PRECISION": "bf16",
    "INFERENCE_PRECISION_HINT": "bf16",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32,
}

# Env escapes for the config defaults above. Without these the preset wins over
# the env var: merge_preset_config() runs setdefault() before
# compile_hooks.apply_kv_cache_config_defaults() ever reads OV_KV_CACHE_PRECISION,
# so the env var is silently ignored. That matters on AVX2-only CPUs, where the
# PagedAttention kernel accepts f32 only (executor_pa.cpp:2558) and the bf16
# default makes the backend unusable.
_PRESET_CONFIG_ENV = {
    "KV_CACHE_PRECISION": "OV_KV_CACHE_PRECISION",
    "INFERENCE_PRECISION_HINT": "OV_INFERENCE_PRECISION_HINT",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "OV_DYNAMIC_QUANTIZATION_GROUP_SIZE",
}


def _preset_config_value(key: str) -> Any:
    """Preset default for `key`, overridden by its env var when one is set."""
    default = _PRESET_CONFIG[key]
    raw = os.environ.get(_PRESET_CONFIG_ENV.get(key, ""))
    if not raw:
        return default
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    return raw


def is_vllm_preset(options) -> bool:
    """True iff options["vllm"] is set to a truthy value."""
    if options is None or "vllm" not in options:
        return False
    v = options["vllm"]
    return bool(v) and str(v).lower() not in ("false", "0")


def preset_flag(key: str):
    """Return the preset value for `key`, or None if `key` is not in the preset."""
    return _PRESET_FLAGS.get(key)


def has_preset_flag(key: str) -> bool:
    return key in _PRESET_FLAGS


def merge_preset_config(base: Optional[dict]) -> dict:
    """Return a dict with the preset OV-config defaults filled in. Caller-supplied
    entries in `base` take priority."""
    out = dict(base or {})
    for k in _PRESET_CONFIG:
        out.setdefault(k, _preset_config_value(k))
    return out


def config_with_vllm_defaults(options):
    """Return options["config"] (or a fresh dict), merged with the vLLM
    preset OV-config defaults when options["vllm"] is set. Caller-supplied
    config keys take priority. Returns the unchanged config when the vLLM
    preset is not active.
    """
    from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_config
    base = dict(_get_config(options) or {})
    if is_vllm_preset(options):
        return merge_preset_config(base)
    return base
