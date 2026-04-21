"""CPU-only end-to-end demo of the Phase 5 attribution API.

Exercises ``TabICLExplainer`` with randomly-initialised attribution heads
on a toy classification dataset and prints the five public attribution
surfaces:

1. ``observational_relevance_``          (Head A)
2. ``interventional_effects_``           (Head I)
3. ``identifiability_scope_``            (caveat string)
4. ``marginal_conditional_contributions(S)``  (Head C query)
5. ``conditional_relevance_graph(threshold)`` (Head C graph)

The heads here are freshly-initialised -- **the numbers are not
meaningful attributions**. The point is to prove the API surface works
end-to-end on a laptop without needing GPU access, HF-Hub access, or a
Phase-4 checkpoint.

To keep the demo CPU-only and offline, we monkey-patch the base
estimator's ``_load_model`` to install a tiny in-memory TabICL trunk
(same trick the Phase 5 unit tests use).

Run
---

    uv run python notebooks/explainer_demo.py
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_classification

from tabicl import TabICLClassifier, TabICLExplainer
from tabicl.model.heads import (
    ConditionalHead,
    InterventionalHead,
    ObservationalHead,
)
from tabicl.model.tabicl import TabICL


EMBED_DIM = 32
MAX_CLASSES = 10
SEED = 0


def _build_tiny_tabicl() -> TabICL:
    """A ~100k-parameter TabICL trunk we can run on a laptop CPU in seconds."""
    torch.manual_seed(SEED)
    return TabICL(
        max_classes=MAX_CLASSES,
        embed_dim=EMBED_DIM,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=8,
        icl_num_blocks=2,
        icl_nhead=4,
        row_num_blocks=2,
        row_nhead=4,
        row_num_cls=2,
        ff_factor=2,
    )


def _install_tiny_trunk(estimator: TabICLClassifier) -> None:
    """Replace ``_load_model`` so ``fit`` doesn't touch HF Hub."""
    tiny = _build_tiny_tabicl()

    def _fake_load_model(self=estimator):
        self.model_path_ = None
        self.model_ = tiny
        self.model_config_ = {"max_classes": MAX_CLASSES, "embed_dim": EMBED_DIM}
        self.model_.eval()

    estimator._load_model = _fake_load_model


def _fresh_heads() -> dict:
    torch.manual_seed(SEED)
    return {
        "observational": ObservationalHead(embed_dim=EMBED_DIM),
        "interventional": InterventionalHead(embed_dim=EMBED_DIM),
        "conditional": ConditionalHead(embed_dim=EMBED_DIM),
    }


def _format_row(label: str, values: np.ndarray) -> str:
    with np.printoptions(precision=3, suppress=True, nanstr="  nan"):
        return f"{label:>38s}: {values}"


def main() -> None:
    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=5, n_redundant=0,
        n_classes=2, random_state=SEED,
    )
    X = X.astype(np.float32)
    X_train, X_test = X[:60], X[60:]
    y_train = y[:60]

    # Introduce a constant feature (column 2) so the demo also shows the
    # UniqueFeatureFilter -> NaN inflation path.
    X_train = X_train.copy()
    X_train[:, 2] = 3.14
    X_test = X_test.copy()
    X_test[:, 2] = 3.14

    clf = TabICLClassifier(n_estimators=2, random_state=SEED, verbose=False)
    _install_tiny_trunk(clf)

    explainer = TabICLExplainer(base_estimator=clf, heads=_fresh_heads())
    explainer.fit(X_train, y_train)

    print("Phase 5 attribution API -- toy-data demo")
    print("=" * 60)
    print(f"n_features_in_: {explainer.n_features_in_}  (column 2 is constant)")
    print()

    print("Static attributes (populated at fit):")
    print(_format_row("observational_relevance_", explainer.observational_relevance_))
    print(_format_row("interventional_effects_", explainer.interventional_effects_))
    print()

    print("identifiability_scope_ (caveat string):")
    print("  " + explainer.identifiability_scope_.replace("\n", "\n  "))
    print()

    print("Head C -- marginal_conditional_contributions(S):")
    for S in ([], [0], [0, 1]):
        scores = explainer.marginal_conditional_contributions(S)
        print(_format_row(f"  S={S}", scores))
    print()

    print("Head C -- conditional_relevance_graph (not a causal DAG):")
    graph = explainer.conditional_relevance_graph(threshold=0.0)
    print("  threshold=0.0 ->")
    print("  " + str(graph).replace("\n", "\n  "))
    print()

    # Prediction still delegates to the 8-member ensemble of the base estimator.
    y_pred = explainer.predict(X_test)
    proba = explainer.predict_proba(X_test)
    print(f"predict(X_test[:5])        : {y_pred[:5]}")
    print(f"predict_proba(X_test[:1])  : {proba[:1]}")

    print()
    print("Reminder: these heads are randomly-initialised. The API is "
          "live; the numbers become meaningful only after Phase 4 (or "
          "the Phase 6e head-only fine-tune) produces trained weights.")


if __name__ == "__main__":
    main()
