from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import N_FEATURES, build_features, reciprocal_rank, zscore  # noqa: E402
from models import Reranker, listwise_loss  # noqa: E402
from train import recall_at_k, tile_edges  # noqa: E402


def toy_scores() -> dict[str, np.ndarray]:
    return {"dense_max": np.array([[0.9, 0.5, 0.5, 0.1]]),
            "dense_mean": np.array([[0.4, 0.4, 0.2, 0.0]]),
            "bm25_max": np.array([[0.0, 3.0, 1.0, 2.0]])}


def test_features_shape_and_values() -> None:
    deg = np.array([1.0, -1.0, 0.0, 0.0])
    x = build_features(toy_scores(), deg)
    assert x.shape == (1, 4, N_FEATURES) and x.dtype == np.float32
    np.testing.assert_allclose(x[0, :, 0].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(x[0, :, 0].std(), 1, atol=1e-5)
    # dense tie between clauses 1 and 2 broken by lower index
    np.testing.assert_allclose(x[0, :, 3], [1, 1 / 2, 1 / 3, 1 / 4])
    np.testing.assert_allclose(x[0, :, 4], [1 / 4, 1, 1 / 3, 1 / 2])
    np.testing.assert_allclose(x[0, :, 5], [1, 1, 1, 1])          # 4 clauses < top-5
    np.testing.assert_allclose(x[0, :, 6], deg)


def test_zscore_constant_row_is_zero() -> None:
    assert np.all(zscore(np.ones((2, 5))) == 0)


def test_reciprocal_rank_top5_flag() -> None:
    rr = reciprocal_rank(np.arange(8, dtype=float)[None, ::-1])
    assert ((rr >= 1 / 5).sum()) == 5


def test_listwise_loss_matches_formula() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    gold = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    z = math.log(math.exp(2) + 1 + math.exp(-1))
    expected = ((-(2 - z) - (0 - z)) / 2 + math.log(3)) / 2
    assert float(listwise_loss(logits, gold)) == pytest.approx(expected, rel=1e-6)


def test_listwise_loss_rejects_empty_gold() -> None:
    with pytest.raises(ValueError):
        listwise_loss(torch.zeros(1, 3), torch.zeros(1, 3))


def toy_graph() -> tuple[torch.Tensor, torch.Tensor]:
    # 4 nodes, relation 0: 0->1, 1->2 ; relation 1: 3->0, 2->3
    ei = torch.tensor([[0, 1, 3, 2], [1, 2, 0, 3]])
    et = torch.tensor([0, 0, 1, 1])
    return ei, et


@pytest.mark.parametrize("kind", ["mlp", "rgcn", "sage", "gat"])
def test_forward_pass_toy(kind: str) -> None:
    torch.manual_seed(0)
    ei, et = toy_graph()
    m = Reranker(kind, N_FEATURES, hidden=8, num_relations=2).eval()
    out = m(torch.randn(4, N_FEATURES), ei, et)
    assert out.shape == (4,) and torch.isfinite(out).all()


def test_rgcn_uses_edge_types_and_mlp_ignores_edges() -> None:
    torch.manual_seed(0)
    ei, et = toy_graph()
    x = torch.randn(4, N_FEATURES)
    rgcn = Reranker("rgcn", N_FEATURES, hidden=8, num_relations=2).eval()
    assert not torch.allclose(rgcn(x, ei, et), rgcn(x, ei, 1 - et))
    mlp = Reranker("mlp", N_FEATURES, hidden=8).eval()
    assert torch.allclose(mlp(x, ei, et), mlp(x, ei[:, :0], et[:0]))


def test_disjoint_union_equals_separate_graphs() -> None:
    torch.manual_seed(0)
    ei, et = toy_graph()
    m = Reranker("rgcn", N_FEATURES, hidden=8, num_relations=2).eval()
    xa, xb = torch.randn(4, N_FEATURES), torch.randn(4, N_FEATURES)
    tei, tet = tile_edges(ei.numpy(), et.numpy(), 4, 2)
    joint = m(torch.cat([xa, xb]), tei, tet)
    assert torch.allclose(joint[:4], m(xa, ei, et), atol=1e-6)
    assert torch.allclose(joint[4:], m(xb, ei, et), atol=1e-6)


def test_recall_ties_by_index() -> None:
    s = np.array([[1.0, 1.0, 0.0]])
    assert recall_at_k(s, np.array([[0.0, 1.0, 0.0]]), 1) == 0.0
    assert recall_at_k(s, np.array([[1.0, 0.0, 0.0]]), 1) == 1.0
