"""Training loop: disjoint-union batching, early stopping, cached predictions.

Each query is its own copy of the 82-node clause graph with query-specific
features; a batch of B queries is the disjoint union of B copies. Early
stopping watches recall@5 on an inner validation fold of real benchmark queries
(never the test fold).
Single-threaded CPU with fixed seeds is reproducible, so deterministic-algorithms
mode is left off (it doubled step time in the prototype).
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from gnn_common import CACHE_DIR, config_hash
from models import Reranker, listwise_loss

CACHE_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    kind: str
    relation_set: str
    hidden: int = 32
    layers: int = 2
    dropout: float = 0.2
    lr: float = 5e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 15
    gat_heads: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tile_edges(edge_index: np.ndarray, edge_type: np.ndarray, n_nodes: int, copies: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges of `copies` disjoint copies, copy k's nodes offset by k * n_nodes (copy order preserved)."""
    e = edge_index.shape[1]
    offs = np.repeat(np.arange(copies) * n_nodes, e)
    ei = np.tile(edge_index, copies) + offs[None, :]
    return torch.from_numpy(ei), torch.from_numpy(np.tile(edge_type, copies))


def gold_mask(golds: list[list[int]], n_nodes: int) -> np.ndarray:
    m = np.zeros((len(golds), n_nodes), dtype=np.float32)
    for i, g in enumerate(golds):
        m[i, g] = 1.0
    return m


def recall_at_k(scores: np.ndarray, gold: np.ndarray, k: int = 5) -> float:
    """Mean recall@k; ties broken by lower index, matching graph_expand.rank."""
    order = np.lexsort((np.broadcast_to(np.arange(scores.shape[1]), scores.shape), -scores), axis=1)[:, :k]
    hits = np.take_along_axis(gold, order, axis=1).sum(axis=1)
    return float((hits / gold.sum(axis=1)).mean())


class Batcher:
    def __init__(self, edge_index: np.ndarray, edge_type: np.ndarray, n_nodes: int, max_batch: int) -> None:
        self.n, self.e = n_nodes, edge_index.shape[1]
        self.ei, self.et = tile_edges(edge_index, edge_type, n_nodes, max_batch)

    def edges(self, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ei[:, : b * self.e], self.et[: b * self.e]


def predict(model: Reranker, x: np.ndarray, batcher: Batcher, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.from_numpy(x[s:s + batch])
            b = xb.shape[0]
            if b > batcher.ei.shape[1] // max(batcher.e, 1):
                raise ValueError("batch larger than the tiled edge buffer")
            ei, et = batcher.edges(b)
            out.append(model(xb.reshape(-1, xb.shape[-1]), ei, et).reshape(b, -1).numpy())
    return np.concatenate(out) if out else np.zeros((0, x.shape[1]))


def fingerprint(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(a.shape).encode())
    return h.hexdigest()[:16]


def train_and_predict(cfg: TrainConfig, seed: int, x_train: np.ndarray, g_train: np.ndarray,
                      x_val: np.ndarray, g_val: np.ndarray, x_eval: np.ndarray,
                      edge_index: np.ndarray, edge_type: np.ndarray, num_relations: int,
                      tag: str = "") -> dict[str, Any]:
    """Train one model, early-stopped on validation recall@5; return eval-set scores.

    Results are cached under results/gnn_milestone/cache/ keyed by the config,
    seed and a fingerprint of every input array.
    """
    key = config_hash({"cfg": asdict(cfg), "seed": seed, "v": CACHE_VERSION, "nrel": num_relations,
                       "data": fingerprint(x_train, g_train, x_val, g_val, x_eval, edge_index, edge_type)})
    path = CACHE_DIR / f"run_{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        rec["eval_scores"] = np.array(rec["eval_scores"])
        rec["cached"] = True
        return rec

    torch.set_num_threads(1)
    set_seed(seed)
    n_nodes, n_feat = x_train.shape[1], x_train.shape[2]
    max_batch = max(cfg.batch_size, 256)
    batcher = Batcher(edge_index, edge_type, n_nodes, max_batch)
    model = Reranker(cfg.kind, n_feat, cfg.hidden, cfg.layers, num_relations, cfg.dropout, cfg.gat_heads)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    xt, gt = torch.from_numpy(x_train), torch.from_numpy(g_train)

    best_r5, best_epoch, best_state, bad, history = -1.0, -1, None, 0, []
    t0 = time.time()
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        losses = []
        for s in range(0, len(perm), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            b = len(idx)
            ei, et = batcher.edges(b)
            logits = model(xt[idx].reshape(-1, n_feat), ei, et).reshape(b, n_nodes)
            loss = listwise_loss(logits, gt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        r5 = recall_at_k(predict(model, x_val, batcher), g_val, 5)
        history.append((epoch, float(np.mean(losses)), r5))
        if r5 > best_r5:
            best_r5, best_epoch, bad = r5, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    eval_scores = predict(model, x_eval, batcher)
    rec = {"tag": tag, "cfg": asdict(cfg), "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
           "val_recall@5": best_r5, "seconds": round(time.time() - t0, 1), "history": history,
           "eval_scores": eval_scores.round(6).tolist()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    rec["eval_scores"] = eval_scores
    rec["cached"] = False
    return rec
