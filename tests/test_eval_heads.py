"""Phase 6e — end-to-end smoke for the eval_heads CLI.

Builds a tiny TabICL trunk + three freshly-initialised heads, saves a
checkpoint in the format the Phase 5-extended ``Trainer.save_checkpoint``
writes, runs the eval script on a 2-batch LiNGAM-only stream, and
checks the output CSV has the expected columns plus a trailing ``mean``
row.

The metrics produced here are not meaningful (untrained heads), but the
pipeline is — the test exercises:

- Checkpoint round-trip (``load_checkpoint`` + fresh ``state_dict``).
- End-to-end forward through the trunk with ``return_column_embeddings``.
- Per-batch metric computation including Head C per-triple evaluation.
- CSV writer's trailing-mean aggregation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from tabicl.eval.eval_heads import main as eval_main
from tabicl.model.heads import (
    ConditionalHead,
    InterventionalHead,
    ObservationalHead,
)
from tabicl.model.tabicl import TabICL


EMBED_DIM = 32


def _build_tiny_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    trunk_config = {
        "max_classes": 10,
        "embed_dim": EMBED_DIM,
        "col_num_blocks": 2,
        "col_nhead": 4,
        "col_num_inds": 8,
        "col_feature_group": False,
        "icl_num_blocks": 2,
        "icl_nhead": 4,
        "row_num_blocks": 2,
        "row_nhead": 4,
        "row_num_cls": 2,
        "ff_factor": 2,
    }
    trunk = TabICL(**trunk_config)
    head_a = ObservationalHead(embed_dim=EMBED_DIM)
    head_i = InterventionalHead(embed_dim=EMBED_DIM)
    head_c = ConditionalHead(embed_dim=EMBED_DIM)

    checkpoint = {
        "config": trunk_config,
        "state_dict": trunk.state_dict(),
        "heads": {
            "observational": head_a.state_dict(),
            "interventional": head_i.state_dict(),
            "conditional": head_c.state_dict(),
            "config": {"embed_dim": EMBED_DIM, "hidden_dim": None},
        },
    }
    torch.save(checkpoint, path)


def test_eval_heads_end_to_end(tmp_path: Path):
    ckpt = tmp_path / "phase6e_smoke.ckpt"
    out_csv = tmp_path / "heads_eval.csv"
    _build_tiny_checkpoint(ckpt)

    rc = eval_main(
        [
            "--checkpoint_path", str(ckpt),
            "--out", str(out_csv),
            "--n_batches", "2",
            "--batch_size", "4",
            "--batch_size_per_gp", "2",
            "--min_features", "4",
            "--max_features", "6",
            "--min_seq_len", "64",
            "--max_seq_len", "128",
            "--prior_type", "lingam_scm",
            "--seed", "0",
            "--device", "cpu",
        ]
    )
    assert rc == 0
    assert out_csv.exists()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))

    # Two per-batch rows plus the trailing mean row.
    assert len(rows) == 3
    assert rows[-1]["batch"] == "mean"
    # Expected columns exist.
    for col in ("spearman_A", "pearson_A", "top1_A", "mse_C", "mae_C", "frac_identifiable"):
        assert col in rows[0], f"missing column {col}"

    # The LiNGAM-only prior has is_identifiable=True for every sample.
    assert float(rows[0]["frac_identifiable"]) == pytest.approx(1.0)


def test_eval_heads_rejects_checkpoint_without_heads(tmp_path: Path):
    from tabicl.eval.eval_heads import load_checkpoint

    bad_ckpt = tmp_path / "no_heads.ckpt"
    torch.save({"config": {"embed_dim": EMBED_DIM}, "state_dict": {}}, bad_ckpt)

    with pytest.raises(KeyError, match="heads"):
        load_checkpoint(bad_ckpt, device="cpu")
