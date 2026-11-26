# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify a GPU compile leaves the KV-cache Parameters untyped, so the GPU plugin types them from KV_CACHE_PRECISION.

A bf16 model's bf16 KV Parameters otherwise get a Convert, and the GPU PagedAttention op is built with a dynamic shape.
"""

import pytest

import openvino as ov
import openvino.opset13 as op

from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks


def _model(et):
    key = op.parameter([-1, -1, -1, -1], et, name="__pa__model.layers.0.self_attn.attn__key_cache")
    value = op.parameter([-1, -1, -1, -1], et, name="__pa__model.layers.0.self_attn.attn__value_cache")
    x = op.parameter([-1, 8], ov.Type.f32, name="x")
    return ov.Model([op.result(key), op.result(value), op.result(x)], [key, value, x])


@pytest.mark.precommit
@pytest.mark.parametrize("device", ["GPU", "GPU.0"])
@pytest.mark.parametrize("et", [ov.Type.bf16, ov.Type.f16])
def test_gpu_kv_cache_left_to_plugin(device, et):
    model, config = _model(et), {}
    compile_hooks.apply_post_config(config, device, {"vllm": True}, model)

    kv = [p for p in model.inputs if compile_hooks._is_kv_cache_port(p)]
    assert len(kv) == 2 and all(p.get_element_type() == ov.Type.dynamic for p in kv)
    assert model.input("x").get_element_type() == ov.Type.f32
    # KV_CACHE_PRECISION stays the user's, or the GPU plugin's compressed default.
    assert config == {}


@pytest.mark.precommit
def test_gpu_untouched_without_vllm_preset():
    model = _model(ov.Type.bf16)
    compile_hooks.apply_post_config({}, "GPU", {}, model)
    assert all(p.get_element_type() == ov.Type.bf16 for p in model.inputs if compile_hooks._is_kv_cache_port(p))
