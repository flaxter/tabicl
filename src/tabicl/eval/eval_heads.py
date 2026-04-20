"""Phase 6e — evaluate attribution heads against Phase 3 ground-truth labels.

Loads a checkpoint that contains both the TabICL trunk weights and the
three attribution-head state dicts (written by the Phase 5-extended
``Trainer.save_checkpoint``), runs the Phase 3 synthetic sampler for
``n_batches`` batches, and reports Spearman/Pearson/top-k-recall for
each head plus Head C MSE/MAE, one CSV row per batch and a trailing
``mean`` row.

Usage
-----
    python -m tabicl.eval.eval_heads \
        --checkpoint_path path/to/head_finetune.ckpt \
        --out results/heads_eval.csv \
        --n_batches 50 --batch_size 32 \
        --min_features 5 --max_features 50 \
        --min_seq_len 256 --max_seq_len 1024 \
        --seed 0 --device cpu

Output CSV columns:
    batch, n_datasets, frac_identifiable,
    spearman_A, pearson_A, top1_A, top3_A, top5_A,
    spearman_I, pearson_I, top1_I, top3_I, top5_I,
    mse_C, mae_C
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from tabicl.eval.metrics import HeadMetrics, aggregate_metrics
from tabicl.model.heads import (
    ConditionalHead,
    InterventionalHead,
    ObservationalHead,
)
from tabicl.model.tabicl import TabICL
from tabicl.prior.dataset import PriorDataset


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedHeads:
    trunk: TabICL
    head_a: ObservationalHead
    head_i: InterventionalHead
    head_c: ConditionalHead


def load_checkpoint(path: str | Path, device: str | torch.device) -> LoadedHeads:
    """Load trunk + three heads from a checkpoint written by the Phase 4 Trainer.

    Requires ``checkpoint["config"]`` (trunk config dict),
    ``checkpoint["state_dict"]`` (trunk weights), and
    ``checkpoint["heads"]`` with entries ``observational``,
    ``interventional``, ``conditional``, ``config``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    for key in ("config", "state_dict", "heads"):
        if key not in checkpoint:
            raise KeyError(
                f"checkpoint at {path} missing required key '{key}'. Expected a "
                f"Phase 4 Trainer checkpoint with multi_task_enabled=True."
            )

    trunk = TabICL(**checkpoint["config"])
    trunk.load_state_dict(checkpoint["state_dict"])
    trunk.to(device)
    # Keep the trunk in ``train()`` mode for eval: the inference path
    # asserts all tables share a class count, which is false for the
    # heterogeneous Phase 3 prior stream. We wrap every forward in
    # ``torch.no_grad()`` so there are no gradient or dropout effects.
    trunk.train()

    heads_blob = checkpoint["heads"]
    head_cfg = heads_blob.get("config", {})
    embed_dim = head_cfg.get("embed_dim") or checkpoint["config"].get("embed_dim")
    hidden_dim = head_cfg.get("hidden_dim")
    if embed_dim is None:
        raise ValueError(
            f"checkpoint at {path} does not expose embed_dim in either "
            f"`heads.config` or `config`."
        )

    head_a = ObservationalHead(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device).eval()
    head_i = InterventionalHead(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device).eval()
    head_c = ConditionalHead(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device).eval()
    head_a.load_state_dict(heads_blob["observational"])
    head_i.load_state_dict(heads_blob["interventional"])
    head_c.load_state_dict(heads_blob["conditional"])

    return LoadedHeads(trunk=trunk, head_a=head_a, head_i=head_i, head_c=head_c)


# ---------------------------------------------------------------------------
# Per-batch evaluation
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    batch: int
    n_datasets: int
    frac_identifiable: float
    metrics_A: HeadMetrics
    metrics_I: HeadMetrics
    mse_C: float
    mae_C: float


def evaluate_batch(
    batch,
    heads: LoadedHeads,
    batch_idx: int,
    device: str | torch.device,
) -> BatchResult:
    """Run the heads on one nine-tuple batch from :class:`PriorDataset`."""
    X, y, d, seq_lens, train_sizes, o_star, i_star, is_identifiable, c_triples = batch

    X = X.to(device)
    d = d.to(device)
    # One train_size per batch (validate_micro_batch semantics); use the
    # minimum so every dataset has at least that many training rows.
    train_size = int(train_sizes.min().item())
    y_train = y[:, :train_size].to(device)

    with torch.no_grad():
        _, col_emb = heads.trunk(
            X, y_train=y_train, d=d, return_column_embeddings=True
        )

        preds_A = heads.head_a(col_emb).cpu().numpy()  # (B, H)
        preds_I = heads.head_i(col_emb).cpu().numpy()  # (B, H)

    targets_A = o_star.cpu().numpy()
    targets_I = i_star.cpu().numpy()

    # Gate Head I predictions on the identifiability mask — non-identifiable
    # samples have all-NaN targets and should not contribute to aggregation.
    is_id = is_identifiable.cpu().numpy().astype(bool)
    preds_I_gated = preds_I.copy()
    preds_I_gated[~is_id] = np.nan
    targets_I_gated = targets_I.copy()
    targets_I_gated[~is_id] = np.nan

    metrics_A = aggregate_metrics(preds_A, targets_A)
    metrics_I = aggregate_metrics(preds_I_gated, targets_I_gated)

    mse_C, mae_C = _head_c_errors(heads.head_c, col_emb, c_triples)

    return BatchResult(
        batch=batch_idx,
        n_datasets=int(X.shape[0]),
        frac_identifiable=float(is_id.mean()),
        metrics_A=metrics_A,
        metrics_I=metrics_I,
        mse_C=mse_C,
        mae_C=mae_C,
    )


def _head_c_errors(
    head_c: ConditionalHead,
    col_emb: torch.Tensor,
    c_triples,
) -> tuple[float, float]:
    """Per-dataset Head C MSE and MAE, averaged across datasets.

    ``c_triples`` is a list of length ``B``; each element is a list of
    ``(i, S_mask, c_star)`` tuples produced by Phase 3. Features past
    each dataset's active count carry NaN in the embeddings' position
    dimension via the NaN-padded mask; we trust Phase 3 to only emit
    triples for active features.
    """
    B, H, _ = col_emb.shape
    device = col_emb.device
    per_dataset_errors = []

    for b, triples in enumerate(c_triples):
        if not triples:
            continue
        c_pred = []
        c_true = []
        for (i, s_mask, c_star) in triples:
            mask = torch.as_tensor(s_mask, dtype=torch.bool, device=device)
            if mask.numel() != H:
                # PriorDataset pads masks to max_features; slice to this dataset's p.
                mask = mask[:H] if mask.numel() > H else torch.cat(
                    [mask, torch.zeros(H - mask.numel(), dtype=torch.bool, device=device)]
                )
            with torch.no_grad():
                scores = head_c(col_emb[b:b + 1], mask.unsqueeze(0)).squeeze(0)
            c_pred.append(float(scores[int(i)].item()))
            c_true.append(float(c_star))

        if not c_pred:
            continue
        diff = np.array(c_pred) - np.array(c_true)
        per_dataset_errors.append((float((diff ** 2).mean()), float(np.abs(diff).mean())))

    if not per_dataset_errors:
        return float("nan"), float("nan")
    mses, maes = zip(*per_dataset_errors)
    return float(np.mean(mses)), float(np.mean(maes))


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


_CSV_COLS = [
    "batch", "n_datasets", "frac_identifiable",
    "spearman_A", "pearson_A", "top1_A", "top3_A", "top5_A", "n_valid_A",
    "spearman_I", "pearson_I", "top1_I", "top3_I", "top5_I", "n_valid_I",
    "mse_C", "mae_C",
]


def _result_to_row(res: BatchResult) -> dict:
    return {
        "batch": res.batch,
        "n_datasets": res.n_datasets,
        "frac_identifiable": res.frac_identifiable,
        "spearman_A": res.metrics_A.spearman,
        "pearson_A": res.metrics_A.pearson,
        "top1_A": res.metrics_A.top1,
        "top3_A": res.metrics_A.top3,
        "top5_A": res.metrics_A.top5,
        "n_valid_A": res.metrics_A.n_valid,
        "spearman_I": res.metrics_I.spearman,
        "pearson_I": res.metrics_I.pearson,
        "top1_I": res.metrics_I.top1,
        "top3_I": res.metrics_I.top3,
        "top5_I": res.metrics_I.top5,
        "n_valid_I": res.metrics_I.n_valid,
        "mse_C": res.mse_C,
        "mae_C": res.mae_C,
    }


def write_csv(out_path: str | Path, rows: List[dict]) -> None:
    """Write per-batch rows plus a trailing mean row."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write; did the eval loop produce any batches?")

    mean_row = {"batch": "mean", "n_datasets": sum(r["n_datasets"] for r in rows)}
    for col in _CSV_COLS[2:]:
        values = [r[col] for r in rows if np.isfinite(r[col])]
        mean_row[col] = float(np.mean(values)) if values else float("nan")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow(mean_row)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--batch_size_per_gp", type=int, default=4)
    p.add_argument("--min_features", type=int, default=5)
    p.add_argument("--max_features", type=int, default=50)
    p.add_argument("--max_classes", type=int, default=10)
    p.add_argument("--min_seq_len", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--min_train_size", type=float, default=0.1)
    p.add_argument("--max_train_size", type=float, default=0.9)
    p.add_argument("--prior_type", default="mix_scm_identifiable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    heads = load_checkpoint(args.checkpoint_path, device=device)

    dataset = PriorDataset(
        batch_size=args.batch_size,
        batch_size_per_gp=args.batch_size_per_gp,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        min_train_size=args.min_train_size,
        max_train_size=args.max_train_size,
        prior_type=args.prior_type,
        device=str(device),
        n_jobs=1,
    )

    rows: List[dict] = []
    for b in range(args.n_batches):
        batch = dataset.get_batch(return_labels=True)
        res = evaluate_batch(batch, heads, batch_idx=b, device=device)
        rows.append(_result_to_row(res))
        print(
            f"[{b + 1}/{args.n_batches}] spearman_A={res.metrics_A.spearman:.3f} "
            f"spearman_I={res.metrics_I.spearman:.3f} mse_C={res.mse_C:.4f}"
        )

    write_csv(args.out, rows)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
