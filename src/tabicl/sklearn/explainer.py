"""Phase 5 — scikit-learn attribution API.

Ships :class:`TabICLExplainer`, a thin wrapper around a fitted
:class:`TabICLClassifier` or :class:`TabICLRegressor` that populates the
per-feature attribution attributes specified in PLAN §Phase 5:

- ``observational_relevance_``    — Head A output per original feature
- ``interventional_effects_``     — Head I output per original feature
- ``identifiability_scope_``      — string documenting Head I caveats
- ``marginal_conditional_contributions(S)`` — Head C at conditioning set ``S``
- ``conditional_relevance_graph(threshold)``  — ``(p, p)`` boolean adjacency

Attribution is computed from a **single forward pass** of the trunk with
``return_column_embeddings=True`` on one canonical (no-shuffle,
first-norm-method) view of the data. The ensemble path used by
``predict``/``predict_proba`` is untouched: prediction continues to go
through the base estimator's 8-member ensemble.

The three attribution heads are not present in upstream TabICL v1/v2
checkpoints. Phase 5 requires the caller to supply them either:

1. Directly as ``nn.Module`` instances via ``heads={...}``.
2. From a checkpoint produced by Phase 4 training via
   ``heads_checkpoint_path="..."``.

Attribution *quality* is out of scope for Phase 5 — trained head weights
come from Phase 4, and their evaluation lives in Phase 6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Mapping, Any

import numpy as np
import torch
from torch import nn

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tabicl.model.heads import (
    ObservationalHead,
    InterventionalHead,
    ConditionalHead,
)


_IDENTIFIABILITY_DISCLAIMER = (
    "Head I (interventional_effects_) is trained on the identifiable "
    "structural subfamily (LiNGAM, ANM, tree-path) from the Phase 3 "
    "sampler. At inference time the true data-generating process is "
    "unknown; the returned scores are posterior means under the training "
    "prior conditioned on the observed context. Interpret them as causal "
    "do(X_i) effects only if the dataset's DGP plausibly falls inside "
    "that identifiable family. See notes/PLAN.md section 'What this "
    "method does and does not fix' for the honest scoping table."
)


class TabICLExplainer(BaseEstimator):
    """Wrap a fitted TabICL estimator with the Phase 5 attribution API.

    This is the scikit-learn surface that exposes the three attribution
    heads trained during Phase 4 (observational :math:`o^*_i`,
    interventional :math:`\\iota^*_i`, conditional :math:`c^*_{i \\mid S}`)
    without altering the base estimator's prediction contract. Prediction
    continues to go through ``base_estimator_``'s 8-member ensemble;
    attribution is computed once on a single canonical view of the data
    from a single forward pass with ``return_column_embeddings=True``.

    .. warning::

       :meth:`conditional_relevance_graph` returns a thresholded
       pairwise-Head-C adjacency matrix. It is **not** a causal DAG
       and must not be interpreted as one. See the method's docstring
       for the full caveat.

    .. warning::

       :attr:`interventional_effects_` is meaningful only within the
       Head I identifiability scope (LiNGAM, ANM, tree-SCM). Always
       read :attr:`identifiability_scope_` before citing these scores
       as causal effects.

    Parameters
    ----------
    base_estimator : TabICLClassifier or TabICLRegressor
        An instance of the base sklearn-compatible TabICL estimator.
        Can be fitted or unfitted; :meth:`fit` delegates to the base
        estimator's ``fit``.

    heads : dict[str, nn.Module], optional
        Pre-built attribution head modules, keyed by
        ``'observational'``, ``'interventional'``, ``'conditional'``.
        Mutually exclusive with ``heads_checkpoint_path``.

    heads_checkpoint_path : str or Path, optional
        Path to a ``torch.save``'d dict with the format produced by
        Phase-4 ``Trainer.save_checkpoint`` when multi-task training is
        enabled:

        .. code-block:: python

            {
                "heads": {
                    "observational": state_dict,
                    "interventional": state_dict,
                    "conditional":   state_dict,
                    "config": {"embed_dim": int, "hidden_dim": int | None},
                },
                ...
            }

        The explainer reconstructs the three head modules from ``config``
        and loads the state dicts.

    device : str, torch.device, or None, default=None
        Device for attribution-pass compute. If ``None``, inherits from
        ``base_estimator.device_`` (after its ``fit``).

    verbose : bool, default=False
        If True, print progress messages during ``fit``.

    Attributes
    ----------
    base_estimator_ : fitted base estimator
        The fitted base estimator; exact object passed in at ``__init__``.
    observational_relevance_ : ndarray of shape (n_features_in_,)
        Head A scores per original input feature. ``NaN`` for features
        dropped by ``UniqueFeatureFilter`` (single-unique-value columns).
    interventional_effects_ : ndarray of shape (n_features_in_,)
        Head I scores per original input feature. Meaningful only within
        the Head I identifiability scope; see
        :attr:`identifiability_scope_`.
    identifiability_scope_ : str
        Human-readable caveat documenting Head I's identifiability scope.
        Present as an attribute so users are prompted to read it.
    n_features_in_ : int
        Number of input features at ``fit`` time (pre-unique-filter).
    """

    def __init__(
        self,
        base_estimator,
        heads: Optional[Mapping[str, nn.Module]] = None,
        heads_checkpoint_path: Optional[str | Path] = None,
        device: Optional[str | torch.device] = None,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.heads = heads
        self.heads_checkpoint_path = heads_checkpoint_path
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "TabICLExplainer":
        """Fit the base estimator, then compute attributions in one pass.

        Steps
        -----
        1. Delegate to ``base_estimator.fit(X, y)`` — this loads the
           pretrained trunk, fits the ensemble generator, and (optionally)
           builds the KV cache.
        2. Resolve the three attribution heads from ``self.heads`` or
           ``self.heads_checkpoint_path``, and move them to the target
           device in ``eval()`` mode.
        3. Run a single canonical forward pass with
           ``return_column_embeddings=True`` to obtain per-feature
           embeddings of shape ``(1, H_filtered, embed_dim)``.
        4. Apply Head A and Head I and store the inflated scores.
        5. Cache the embeddings for on-demand Head C queries.
        """
        self.base_estimator_ = self.base_estimator.fit(X, y)
        self.n_features_in_ = int(self.base_estimator_.n_features_in_)

        self._load_or_check_heads()
        self._resolve_device()
        self._run_attribution_forward()
        self._apply_static_heads()

        self.identifiability_scope_ = _IDENTIFIABILITY_DISCLAIMER
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
    # Attribution queries (Head C)
    # ------------------------------------------------------------------

    def marginal_conditional_contributions(
        self, S: Sequence[int]
    ) -> np.ndarray:
        """Head C scores for every feature given a conditioning set ``S``.

        Parameters
        ----------
        S : sequence of int
            Feature indices (in the **original** input space, pre-filter)
            that are treated as already revealed. Must be a subset of
            ``range(n_features_in_)``.

        Returns
        -------
        ndarray of shape (n_features_in_,)
            Per-feature Head C scores. Entries inside ``S`` are ``NaN``
            because asking "what does X_j add given X_j is known" is
            ill-posed. Entries for features dropped by
            ``UniqueFeatureFilter`` are also ``NaN``.
        """
        check_is_fitted(self, "base_estimator_")
        S = list(S)
        if any(not (0 <= int(i) < self.n_features_in_) for i in S):
            raise ValueError(
                f"S must contain indices in [0, n_features_in_={self.n_features_in_}); got {S}"
            )

        mask_filtered = self._filtered_cond_mask(S)
        cond_mask = torch.as_tensor(mask_filtered, dtype=torch.bool, device=self._device).unsqueeze(0)

        with torch.no_grad():
            scores_filtered = self._head_c(self._column_embeddings, cond_mask)  # (1, H_filtered)

        out = self._inflate(scores_filtered.squeeze(0).cpu().numpy().astype(np.float64))
        # NaN out entries inside S (in original space)
        for i in S:
            out[int(i)] = np.nan
        return out

    def conditional_relevance_graph(
        self, threshold: float = 0.1
    ) -> np.ndarray:
        """Boolean ``(p, p)`` adjacency matrix thresholding pairwise Head C.

        For every feature pair ``(i, j)`` with ``i != j``, asks Head C
        "what does ``X_j`` add given ``X_i`` is known" and marks the edge
        ``(i, j)`` as ``True`` iff that score exceeds ``threshold``.

        .. warning::

           **This is not a causal DAG.** Edges here encode conditional
           observational relevance: feature ``j`` adds predictive signal
           beyond ``{X_i}``. They do not encode causal mechanisms,
           direct-cause edges, or Markov equivalence. Using this graph
           as a substitute for causal discovery will produce incorrect
           conclusions. For causal readings use
           :attr:`interventional_effects_`, subject to the
           :attr:`identifiability_scope_` caveat, or pipe a
           causal-discovery front-end.

        Parameters
        ----------
        threshold : float, default=0.1
            Threshold on the Head C score. Edges below the threshold are
            set to ``False``. Use ``+inf`` for an empty graph, ``-inf``
            for a complete graph (minus the diagonal).

        Returns
        -------
        ndarray of shape (n_features_in_, n_features_in_), dtype=bool
            Directed adjacency matrix. **Not** a causal DAG — each edge
            indicates conditional observational relevance, not a causal
            mechanism. See
            ``learning-to-explain/notes/PLAN.md`` §Phase 5 key decision
            #17 for the rationale behind the ``conditional_relevance_graph``
            name (renamed from the old ``causal_attention_graph`` to
            prevent this exact confusion).
        """
        check_is_fitted(self, "base_estimator_")
        p = self.n_features_in_
        rows = []
        for i in range(p):
            rows.append(self.marginal_conditional_contributions([i]))
        scores = np.stack(rows, axis=0)  # shape (p, p), NaNs at scores[i, i] and filtered positions
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

    def _load_or_check_heads(self) -> None:
        """Resolve the three attribution heads into ``self._head_{a,i,c}``."""
        if self.heads is not None and self.heads_checkpoint_path is not None:
            raise ValueError(
                "Specify at most one of `heads` and `heads_checkpoint_path`."
            )
        if self.heads is None and self.heads_checkpoint_path is None:
            raise ValueError(
                "No attribution heads supplied. Pass `heads={...}` with "
                "ObservationalHead/InterventionalHead/ConditionalHead "
                "modules, or `heads_checkpoint_path=...` pointing to a "
                "Phase-4 training checkpoint that persisted the head "
                "state dicts. Phase 5 is the sklearn surface; training "
                "the heads is Phase 4's job (see notes/PLAN.md)."
            )

        if self.heads is not None:
            heads = _validate_heads_dict(self.heads)
            self._head_a = heads["observational"]
            self._head_i = heads["interventional"]
            self._head_c = heads["conditional"]
            return

        self._head_a, self._head_i, self._head_c = _load_heads_from_checkpoint(
            self.heads_checkpoint_path
        )

    def _resolve_device(self) -> None:
        base_device = getattr(self.base_estimator_, "device_", None)
        if self.device is not None:
            self._device = torch.device(self.device)
        elif base_device is not None:
            self._device = base_device
        else:
            self._device = torch.device("cpu")

        for head in (self._head_a, self._head_i, self._head_c):
            head.to(self._device)
            head.eval()

    def _run_attribution_forward(self) -> None:
        """Do one canonical forward pass with per-column embeddings."""
        model = self.base_estimator_.model_
        eg = self.base_estimator_.ensemble_generator_

        # Canonical view: first fitted normalization method, identity shuffle.
        norm_method = next(iter(eg.preprocessors_))
        preprocessor = eg.preprocessors_[norm_method]

        X_train_filtered = np.asarray(preprocessor.X_transformed_, dtype=np.float32)  # (n_train, H_filtered)
        y_train_np = np.asarray(eg.y_, dtype=np.float32)                                # (n_train,)

        n_train, h_filtered = X_train_filtered.shape

        # One dummy test row — column embeddings are mean-pooled over
        # [:train_size] so the test row has no effect on the embeddings.
        X_dummy_test = np.zeros((1, h_filtered), dtype=np.float32)
        X_cat = np.concatenate([X_train_filtered, X_dummy_test], axis=0)[None, ...]  # (1, T, H)

        X_t = torch.from_numpy(X_cat).to(self._device)
        y_t = torch.from_numpy(y_train_np[None, ...]).to(self._device)

        inference_config = getattr(self.base_estimator_, "inference_config_", None)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
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
        self._column_embeddings = column_embeddings.detach()  # (1, H_filtered, E)

    def _apply_static_heads(self) -> None:
        """Run Head A + Head I once and store inflated scores."""
        with torch.no_grad():
            obs = self._head_a(self._column_embeddings).squeeze(0).cpu().numpy().astype(np.float64)
            interv = self._head_i(self._column_embeddings).squeeze(0).cpu().numpy().astype(np.float64)

        self.observational_relevance_ = self._inflate(obs)
        self.interventional_effects_ = self._inflate(interv)

    def _inflate(self, filtered_scores: np.ndarray) -> np.ndarray:
        """Expand filtered-feature scores to original feature ordering with NaN padding."""
        mask = self.base_estimator_.ensemble_generator_.unique_filter_.features_to_keep_
        out = np.full(self.n_features_in_, np.nan, dtype=np.float64)
        out[mask] = filtered_scores
        return out

    def _filtered_cond_mask(self, S: Sequence[int]) -> np.ndarray:
        """Map an original-space conditioning set ``S`` to the filtered mask."""
        keep = self.base_estimator_.ensemble_generator_.unique_filter_.features_to_keep_
        # Old-index -> new-index only for kept features
        new_index = np.full(self.n_features_in_, -1, dtype=np.int64)
        new_index[keep] = np.arange(int(keep.sum()))

        mask = np.zeros(int(keep.sum()), dtype=bool)
        for i in S:
            j = int(new_index[int(i)])
            if j >= 0:
                mask[j] = True
            # Features dropped by the unique filter silently skip — they contribute 0 to e_S
        return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_heads_dict(heads: Mapping[str, Any]) -> dict[str, nn.Module]:
    required = {"observational", "interventional", "conditional"}
    missing = required - set(heads.keys())
    if missing:
        raise ValueError(
            f"`heads` must contain keys {sorted(required)}; missing {sorted(missing)}"
        )

    expected_types = {
        "observational": ObservationalHead,
        "interventional": InterventionalHead,
        "conditional": ConditionalHead,
    }
    out: dict[str, nn.Module] = {}
    for key, cls in expected_types.items():
        mod = heads[key]
        if not isinstance(mod, nn.Module):
            raise TypeError(
                f"heads['{key}'] must be an nn.Module (ideally {cls.__name__}); "
                f"got {type(mod).__name__}"
            )
        out[key] = mod
    return out


def _load_heads_from_checkpoint(
    path: str | Path,
) -> tuple[ObservationalHead, InterventionalHead, ConditionalHead]:
    """Reconstruct the three heads from a Phase-4 checkpoint's ``heads`` dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"heads_checkpoint_path does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "heads" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {path} has no 'heads' entry. Was it produced "
            f"by Phase-4 training with multi_task_enabled=True? Upstream "
            f"TabICL checkpoints do not contain attribution heads."
        )

    heads_blob = checkpoint["heads"]
    config = heads_blob.get("config", {})
    embed_dim = config.get("embed_dim")
    hidden_dim = config.get("hidden_dim")
    if embed_dim is None:
        raise ValueError(
            f"Checkpoint at {path} is missing heads.config.embed_dim; "
            f"cannot reconstruct attribution heads."
        )

    head_a = ObservationalHead(embed_dim=embed_dim, hidden_dim=hidden_dim)
    head_i = InterventionalHead(embed_dim=embed_dim, hidden_dim=hidden_dim)
    head_c = ConditionalHead(embed_dim=embed_dim, hidden_dim=hidden_dim)

    head_a.load_state_dict(heads_blob["observational"])
    head_i.load_state_dict(heads_blob["interventional"])
    head_c.load_state_dict(heads_blob["conditional"])
    return head_a, head_i, head_c
