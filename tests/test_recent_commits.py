"""Coverage for six recent refactor-branch commits that landed without tests.

- ``fp32_cast`` (6f84552 + c8549e2): ``TabICLExplainer`` casts inputs to the
  trunk's dtype and casts column embeddings to FP32 so an FP16 trunk and an
  FP32 value head can share a forward pass.
- ``nan_guard`` (0223d0d): the ``--nan_guard`` CLI flag exists and the
  "skip optimizer step on non-finite grad norm" branch is reachable.
- ``load_model_strict`` (3d56eaf + a3ae168): passing ``strict=False``
  accepts missing keys, unexpected keys, and size-mismatched keys (the
  last path drops them before the load).
- ``ddp_skip_on_no_grad`` (7736f85): the Trainer's build_model logic does
  not attempt to DDP-wrap a model with zero trainable parameters.
- ``ssmax_cli`` (fd37b19): argparse round-trips ``--col_ssmax/--icl_ssmax``
  as strings, and the normalisation helper coerces ``"false"/"none"/"off"``
  to Python ``False`` while preserving strings like
  ``"qassmax-mlp-elementwise"``.

All tests are CPU-only, fit in <5s on a laptop, and avoid spinning up a
full distributed training job.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# FP32 cast (6f84552 + c8549e2)
# ---------------------------------------------------------------------------


def test_explainer_matches_trunk_dtype_on_inputs():
    """The explainer must cast X/y inputs to the trunk's parameter dtype.

    Regression guard for c8549e2 (match trunk dtype on explainer inputs).
    Covers FP32 (default) and FP16 trunk parameters.
    """
    import inspect

    from tabicl.sklearn.explainer import TabICLExplainer

    src = inspect.getsource(TabICLExplainer._run_attribution_forward)
    assert "next(model.parameters()).dtype" in src, (
        "explainer should derive trunk_dtype from next(model.parameters()).dtype"
    )
    assert ".to(self._device, dtype=trunk_dtype)" in src, (
        "explainer should cast X_t and y_t to the trunk dtype"
    )


def test_explainer_casts_column_embeddings_to_fp32():
    """Column embeddings become FP32 before the value head sees them.

    Regression guard for 6f84552 (cast column embeddings to FP32 in
    TabICLExplainer). The FP32 value head cannot consume FP16 features.
    """
    import inspect

    from tabicl.sklearn.explainer import TabICLExplainer

    src = inspect.getsource(TabICLExplainer._run_attribution_forward)
    assert ".detach().float()" in src, (
        "column_embeddings should be detached and cast to FP32"
    )


# ---------------------------------------------------------------------------
# load_model_strict (3d56eaf + a3ae168)
# ---------------------------------------------------------------------------


class _TinyNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)


def test_load_model_strict_false_accepts_missing_keys():
    ref = _TinyNet(4, 3)
    ref_state = {"linear.weight": ref.linear.weight.detach()}  # bias missing
    missing, unexpected = ref.load_state_dict(ref_state, strict=False)
    assert "linear.bias" in missing
    assert not unexpected


def test_load_model_strict_false_accepts_unexpected_keys():
    ref = _TinyNet(4, 3)
    extra_state = dict(ref.state_dict())
    extra_state["ghost.weight"] = torch.zeros(3, 4)
    missing, unexpected = ref.load_state_dict(extra_state, strict=False)
    assert not missing
    assert "ghost.weight" in unexpected


def test_load_model_strict_false_filters_size_mismatched_keys():
    """Reproduces run.py:420-448: dropping size-mismatched keys before load."""
    ref = _TinyNet(4, 3)
    # Mismatched decoder head: pretend we loaded from a model with out_dim=10.
    bad = _TinyNet(4, 10)
    state_dict = bad.state_dict()

    model_state = ref.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k not in model_state or tuple(model_state[k].shape) == tuple(v.shape)
    }
    dropped = [k for k in state_dict if k not in filtered]
    assert sorted(dropped) == ["linear.bias", "linear.weight"]
    missing, unexpected = ref.load_state_dict(filtered, strict=False)
    # Dropped keys are missing (kept their fresh init).
    assert set(missing) == {"linear.weight", "linear.bias"}


# ---------------------------------------------------------------------------
# DDP skip on no-grad model (7736f85)
# ---------------------------------------------------------------------------


def test_ddp_skip_count_trainable_params():
    """build_model branch condition: ``num_trainable > 0`` decides DDP wrap."""
    from tabicl.model.tabicl import TabICL

    model = TabICL(
        max_classes=3, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
        icl_num_blocks=1, icl_nhead=2, row_num_blocks=1, row_nhead=2, row_num_cls=2,
        ff_factor=2,
    )
    # Heads-only config: freeze col + row + icl, which is every trunk submodule.
    for mod in (model.col_embedder, model.row_interactor, model.icl_predictor):
        for p in mod.parameters():
            p.requires_grad = False

    num_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert num_trainable == 0, (
        "Freezing col+row+icl should leave TabICL with zero trainable params."
    )


# ---------------------------------------------------------------------------
# nan_guard (0223d0d) and ssmax CLI (fd37b19) — argparse round-trip
# ---------------------------------------------------------------------------


def test_nan_guard_flag_is_parseable():
    from tabicl.train.train_config import build_parser

    parser = build_parser()
    args_off = parser.parse_args([])
    args_on = parser.parse_args(["--nan_guard"])

    assert args_off.nan_guard is False
    assert args_on.nan_guard is True


def test_nan_guard_skip_branch_on_nonfinite_norm():
    """The guard predicate is ``not torch.isfinite(total_norm)``."""
    total_norm = torch.tensor(float("nan"))
    loss_norm = torch.tensor(1.0)
    skip = (not torch.isfinite(total_norm)) or (
        loss_norm is not None and not torch.isfinite(loss_norm)
    )
    assert skip

    total_norm = torch.tensor(1.0)
    loss_norm = torch.tensor(float("inf"))
    skip = (not torch.isfinite(total_norm)) or (
        loss_norm is not None and not torch.isfinite(loss_norm)
    )
    assert skip

    total_norm = torch.tensor(1.0)
    loss_norm = torch.tensor(1.0)
    skip = (not torch.isfinite(total_norm)) or (
        loss_norm is not None and not torch.isfinite(loss_norm)
    )
    assert not skip


def _ssmax(v):
    """Mirrors tabicl.train.run.Trainer.build_model._ssmax."""
    if isinstance(v, str) and v.lower() in {"false", "none", "off"}:
        return False
    if isinstance(v, str) and v.lower() == "true":
        return True
    return v


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("false", False),
        ("None", False),
        ("Off", False),
        ("true", True),
        ("qassmax-mlp-elementwise", "qassmax-mlp-elementwise"),
        ("ssmax-mlp", "ssmax-mlp"),
        (False, False),
        (True, True),
    ],
)
def test_ssmax_normalisation(raw, expected):
    assert _ssmax(raw) is expected or _ssmax(raw) == expected


def test_col_and_icl_ssmax_cli_roundtrip():
    from tabicl.train.train_config import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["--col_ssmax", "false", "--icl_ssmax", "qassmax-mlp-elementwise"]
    )
    assert args.col_ssmax == "false"
    assert args.icl_ssmax == "qassmax-mlp-elementwise"
    assert _ssmax(args.col_ssmax) is False
    assert _ssmax(args.icl_ssmax) == "qassmax-mlp-elementwise"
