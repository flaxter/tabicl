"""scikit-learn predictive-oracle API.

Ships :class:`TabICLExplainer`, a thin wrapper around a fitted
:class:`TabICLClassifier` or :class:`TabICLRegressor` that exposes the
conditional predictive value oracle as sklearn-compatible attributes and
methods:

- :meth:`conditional_predictive_values(S)` — length-``p`` RMS vector of
  ``sqrt(Delta_{i|S})`` for features not in ``S``. Entries inside ``S``
  are ``NaN``; entries for features dropped by ``UniqueFeatureFilter``
  are also ``NaN``.
- :attr:`predictive_sufficiency_` — cached ``conditional_predictive_values([])``:
  how useful each feature is **on its own**.
- :attr:`predictive_necessity_` — batched leave-one-out: entry ``i``
  estimates ``sqrt(Delta_{i|[p]\\{i}})`` — how much ``X_i`` still adds once
  every other feature is known.
- :meth:`greedy_predictive_path(k)` — greedy acquisition path by repeatedly
  adding the feature with highest predicted value.
- :meth:`conditional_value_graph(threshold)` — pairwise thresholded graph
  of ``Delta_{i|{j}}`` edges. **Not a causal graph.**

Attribution is computed from a **single forward pass** of the trunk with
``return_column_embeddings=True`` on one canonical (no-shuffle,
first-norm-method) view of the data. The ensemble path used by
``predict``/``predict_proba`` is untouched: prediction continues to go
through the base estimator's 8-member ensemble.

The value head is not present in upstream TabICL v1/v2 checkpoints. The
caller must supply it either:

1. Directly as an ``nn.Module`` instance via ``value_head=...``.
2. From a checkpoint produced by training via
   ``heads_checkpoint_path="..."``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tabicl.model.heads import ConditionalPredictiveValueHead


class TabICLExplainer(BaseEstimator):
    """Wrap a fitted TabICL estimator with the conditional predictive value API.

    Parameters
    ----------
    base_estimator : TabICLClassifier or TabICLRegressor
        An instance of the base sklearn-compatible TabICL estimator.
    value_head : ConditionalPredictiveValueHead, optional
        The trained value head. Mutually exclusive with
        ``heads_checkpoint_path``.
    heads_checkpoint_path : str or Path, optional
        Path to a ``torch.save``'d dict with the format:

        .. code-block:: python

            {
                "heads": {
                    "value": state_dict,
                    "config": {"embed_dim": int, "hidden_dim": int | None},
                },
                ...
            }

        The explainer reconstructs the value head from ``config`` and
        loads its state dict.
    device : str, torch.device, or None
        Device for attribution-pass compute. If ``None``, inherits from
        ``base_estimator.device_`` (after its ``fit``).
    verbose : bool
        If True, print progress messages during ``fit``.

    Attributes
    ----------
    base_estimator_ : fitted base estimator
    predictive_sufficiency_ : ndarray of shape (n_features_in_,)
        RMS ``sqrt(Delta_{i|empty}) = sqrt(V({i}))``.
    predictive_necessity_ : ndarray of shape (n_features_in_,)
        RMS ``sqrt(Delta_{i|[p]\\{i}})``.
    n_features_in_ : int
    """

    def __init__(
        self,
        base_estimator,
        value_head: Optional[ConditionalPredictiveValueHead] = None,
        heads_checkpoint_path: Optional[str | Path] = None,
        device: Optional[str | torch.device] = None,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.value_head = value_head
        self.heads_checkpoint_path = heads_checkpoint_path
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "TabICLExplainer":
        """Fit the base estimator, cache column embeddings, compute endpoints."""
        self.base_estimator_ = self.base_estimator.fit(X, y)
        self.n_features_in_ = int(self.base_estimator_.n_features_in_)

        self._load_or_check_head()
        self._resolve_device()
        self._run_attribution_forward()
        self._compute_endpoints()
        return self

    def predict(self, X) -> np.ndarray:
        """Delegate to ``base_estimator.predict``."""
        check_is_fitted(self, "base_estimator_")
        return self.base_estimator_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Delegate to ``base_estimator.predict_proba`` (classifier only)."""
        check_is_fitted(self, "base_estimator_")
        if not hasattr(self.base_estimator_, "predict_proba"):
            raise AttributeError(
                "predict_proba is only available when the base estimator is a "
                "TabICLClassifier. Use predict() for regression."
            )
        return self.base_estimator_.predict_proba(X)

    # ------------------------------------------------------------------
    # Public oracle queries
    # ------------------------------------------------------------------

    def conditional_predictive_values(self, S: Sequence[int]) -> np.ndarray:
        """RMS predictive value at information state ``S``.

        Returns an array of shape ``(n_features_in_,)`` where entry ``i``
        estimates ``sqrt(Delta_{i|S})`` — the outcome-scale reduction in
        Bayes squared-error risk from revealing ``X_i`` once ``X_S`` is
        known. Entries inside ``S`` are ``NaN`` (the query is undefined);
        entries for features dropped by ``UniqueFeatureFilter`` are also
        ``NaN``. Predictive, not causal.
        """
        check_is_fitted(self, "base_estimator_")
        S_list = [int(i) for i in S]
        if any(not (0 <= i < self.n_features_in_) for i in S_list):
            raise ValueError(
                f"S must contain indices in [0, n_features_in_={self.n_features_in_}); got {S_list}"
            )

        mask_filtered = self._filtered_cond_mask(S_list)
        cond_mask = torch.as_tensor(
            mask_filtered, dtype=torch.bool, device=self._device
        ).unsqueeze(0)

        with torch.no_grad():
            scores_filtered = self.value_head_(
                self._column_embeddings, cond_mask
            )  # (1, H_filtered)

        out = self._inflate(scores_filtered.squeeze(0).cpu().numpy().astype(np.float64))
        for i in S_list:
            out[i] = np.nan
        return out

    def greedy_predictive_path(
        self, k: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """Greedy acquisition path by repeated oracle queries.

        Starts at ``S_0 = empty`` and at each step selects
        ``i_t = argmax_{i not in S_t} Delta_{i|S_t}`` (in RMS units),
        then sets ``S_{t+1} = S_t union {i_t}``.

        Parameters
        ----------
        k : int, optional
            Number of features to acquire. ``None`` runs until every
            non-constant feature is selected.

        Returns
        -------
        path : list[int]
            Selected feature indices in acquisition order (original-space).
        gains : list[float]
            Predicted RMS gain at each step.
        """
        check_is_fitted(self, "base_estimator_")
        keep_mask = self.base_estimator_.ensemble_generator_.unique_filter_.features_to_keep_
        valid = set(int(i) for i in np.flatnonzero(keep_mask))
        budget = len(valid) if k is None else min(int(k), len(valid))

        path: List[int] = []
        gains: List[float] = []
        S: List[int] = []
        for _ in range(budget):
            scores = self.conditional_predictive_values(S)
            scores_masked = scores.copy()
            # Only consider features that are in `valid` and not yet in S.
            candidates = valid - set(S)
            not_candidate = np.setdiff1d(
                np.arange(self.n_features_in_), np.fromiter(candidates, dtype=int)
            )
            scores_masked[not_candidate] = -np.inf
            if not np.isfinite(scores_masked).any():
                break
            i_best = int(np.nanargmax(scores_masked))
            path.append(i_best)
            gains.append(float(scores[i_best]))
            S.append(i_best)
        return path, gains

    def conditional_value_graph(self, threshold: float = 0.1) -> np.ndarray:
        """Boolean ``(p, p)`` adjacency matrix from pairwise singleton queries.

        Edge ``(i, j)`` is set iff ``sqrt(Delta_{j|{i}}) > threshold``. The
        diagonal is ``False``.

        .. warning::

           **Not a causal graph.** Edges encode conditional predictive
           value — feature ``j`` adds predictive signal beyond ``{X_i}``.
           They do not encode causal mechanisms or direct-cause edges.
        """
        check_is_fitted(self, "base_estimator_")
        p = self.n_features_in_
        rows = [self.conditional_predictive_values([i]) for i in range(p)]
        scores = np.stack(rows, axis=0)  # (p, p)
        with np.errstate(invalid="ignore"):
            graph = scores > threshold
        np.fill_diagonal(graph, False)
        return graph

    # ------------------------------------------------------------------
    # sklearn boilerplate
    # ------------------------------------------------------------------

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_or_check_head(self) -> None:
        if self.value_head is not None and self.heads_checkpoint_path is not None:
            raise ValueError(
                "Specify at most one of `value_head` and `heads_checkpoint_path`."
            )
        if self.value_head is None and self.heads_checkpoint_path is None:
            raise ValueError(
                "No value head supplied. Pass `value_head=...` with a "
                "ConditionalPredictiveValueHead module, or "
                "`heads_checkpoint_path=...` pointing to a training "
                "checkpoint that persisted the head state dict."
            )

        if self.value_head is not None:
            if not isinstance(self.value_head, nn.Module):
                raise TypeError(
                    f"value_head must be an nn.Module, got {type(self.value_head).__name__}"
                )
            self.value_head_ = self.value_head
            return

        self.value_head_ = _load_value_head_from_checkpoint(self.heads_checkpoint_path)

    def _resolve_device(self) -> None:
        base_device = getattr(self.base_estimator_, "device_", None)
        if self.device is not None:
            self._device = torch.device(self.device)
        elif base_device is not None:
            self._device = base_device
        else:
            self._device = torch.device("cpu")

        self.value_head_.to(self._device)
        self.value_head_.eval()

    def _run_attribution_forward(self) -> None:
        """Single canonical forward pass producing per-column embeddings."""
        model = self.base_estimator_.model_
        eg = self.base_estimator_.ensemble_generator_

        norm_method = next(iter(eg.preprocessors_))
        preprocessor = eg.preprocessors_[norm_method]

        X_train_filtered = np.asarray(preprocessor.X_transformed_, dtype=np.float32)
        y_train_np = np.asarray(eg.y_, dtype=np.float32)

        n_train, h_filtered = X_train_filtered.shape

        X_dummy_test = np.zeros((1, h_filtered), dtype=np.float32)
        X_cat = np.concatenate([X_train_filtered, X_dummy_test], axis=0)[None, ...]

        # Force the whole trunk to fp32 for the attribution forward and
        # disable AMP on every InferenceManager so no sub-forward promotes
        # back to fp16. The v1/v2 checkpoints download in fp16, and the
        # each submodule (col_embedder/row_interactor/icl_predictor) owns
        # its own InferenceManager that wraps forwards in
        # `torch.autocast("cuda")` when use_amp is True (the default for
        # non-small data). Without this belt-and-suspenders cast we see
        # "RuntimeError: expected scalar type Half but found Float" inside
        # row_interactor's LayerNorm because col_embedder's inference
        # manager re-promotes to fp16 even though model.float() cast the
        # params, and the row_interactor return_features=True path
        # bypasses its own InferenceManager and calls LayerNorm directly.
        #
        # 1. Force every submodule's params+buffers to fp32.
        for m in model.modules():
            m.float()
        # 2. Disable AMP on every InferenceManager so no nested forward
        #    auto-promotes back to fp16.
        for m in model.modules():
            if hasattr(m, "inference_mgr"):
                try:
                    m.inference_mgr.use_amp = False
                except AttributeError:
                    pass
        X_t = torch.from_numpy(X_cat).float().to(self._device)
        y_t = torch.from_numpy(y_train_np[None, ...]).float().to(self._device)

        inference_config = getattr(self.base_estimator_, "inference_config_", None)

        was_training = model.training
        model.eval()
        try:
            # The base estimator's InferenceManager wraps each forward in
            # `torch.autocast(device_type="cuda")` when use_amp is on (the
            # default for non-small data). Autocast promotes ops to fp16
            # regardless of the params we cast to fp32 above, so downstream
            # LayerNorms can hit "expected Half but found Float" when
            # normalising already-Half intermediate tensors against fp32
            # weights. Explicitly disable autocast inside the attribution
            # forward so the whole pipeline stays fp32.
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                _, column_embeddings = model(
                    X_t,
                    y_train=y_t,
                    return_column_embeddings=True,
                    inference_config=inference_config,
                )
        finally:
            if was_training:
                model.train()

        assert column_embeddings.shape == (1, h_filtered, model.embed_dim), (
            f"Unexpected column-embedding shape {tuple(column_embeddings.shape)}; "
            f"expected (1, {h_filtered}, {model.embed_dim})"
        )
        self._column_embeddings = column_embeddings.detach().float()

    def _compute_endpoints(self) -> None:
        """Cache predictive_sufficiency_ and predictive_necessity_."""
        self.predictive_sufficiency_ = self.conditional_predictive_values([])

        p = self.n_features_in_
        keep = self.base_estimator_.ensemble_generator_.unique_filter_.features_to_keep_
        valid_idx = np.flatnonzero(keep)
        h_filtered = int(keep.sum())

        # Leave-one-out for every valid filtered feature, batched through
        # the head. One row per valid feature.
        mask_rows = np.ones((h_filtered, h_filtered), dtype=bool)
        np.fill_diagonal(mask_rows, False)
        cond_mask = torch.as_tensor(mask_rows, dtype=torch.bool, device=self._device)
        emb_batch = self._column_embeddings.expand(h_filtered, -1, -1)

        with torch.no_grad():
            scores = self.value_head_(emb_batch, cond_mask)  # (h_filtered, h_filtered)

        diag = scores.diagonal(dim1=0, dim2=1).cpu().numpy().astype(np.float64)
        # Note: diag[j] = head output at position j when mask = LOO-of-j
        # = predicted sqrt(Delta_{j | all-but-j}) = predictive necessity.

        out = np.full(p, np.nan, dtype=np.float64)
        out[valid_idx] = diag
        self.predictive_necessity_ = out

    def _inflate(self, filtered_scores: np.ndarray) -> np.ndarray:
        """Expand filtered-feature scores to original ordering with NaN padding."""
        mask = self.base_estimator_.ensemble_generator_.unique_filter_.features_to_keep_
        out = np.full(self.n_features_in_, np.nan, dtype=np.float64)
        out[mask] = filtered_scores
        return out

    def _filtered_cond_mask(self, S: Sequence[int]) -> np.ndarray:
        """Map an original-space conditioning set ``S`` to the filtered mask."""
        keep = self.base_estimator_.ensemble_generator_.unique_filter_.features_to_keep_
        new_index = np.full(self.n_features_in_, -1, dtype=np.int64)
        new_index[keep] = np.arange(int(keep.sum()))

        mask = np.zeros(int(keep.sum()), dtype=bool)
        for i in S:
            j = int(new_index[int(i)])
            if j >= 0:
                mask[j] = True
            # Features dropped by the unique filter are skipped — they
            # contribute zero to e_S.
        return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_value_head_from_checkpoint(path: str | Path) -> ConditionalPredictiveValueHead:
    """Reconstruct the value head from a training checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"heads_checkpoint_path does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "heads" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {path} has no 'heads' entry. Was it produced "
            f"by training with the predictive-value head enabled? Upstream "
            f"TabICL checkpoints do not contain attribution heads."
        )

    heads_blob = checkpoint["heads"]
    config = heads_blob.get("config", {})
    embed_dim = config.get("embed_dim")
    hidden_dim = config.get("hidden_dim")
    if embed_dim is None:
        raise ValueError(
            f"Checkpoint at {path} is missing heads.config.embed_dim; "
            f"cannot reconstruct the value head."
        )

    head = ConditionalPredictiveValueHead(embed_dim=embed_dim, hidden_dim=hidden_dim)
    head.load_state_dict(heads_blob["value"])
    return head
