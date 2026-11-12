"""Clause rerankers: one shared trunk and head, four choices of propagation layer.

    x -> Linear(F, 32) + ReLU = h0
      -> layer 1 (+ ReLU, dropout) = h1
      -> layer 2 (+ ReLU, dropout) = h2
      -> head(concat[x, h0, h1, h2]) = one score per clause

Layer kinds:
    rgcn  RGCNConv, one weight matrix per relation, mean aggregation, root weight
          (Schlichtkrull et al. 2018): h_v' = W_0 h_v + sum_r mean_{u in N_r(v)} W_r h_u
    sage  SAGEConv (mean), edge types ignored
    gat   GATConv (4 heads averaged, self-loops), edge types ignored
    mlp   Linear: same depth and width, no message passing (control)
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, RGCNConv, SAGEConv

MODEL_KINDS = ("mlp", "rgcn", "sage", "gat")


class Reranker(nn.Module):
    def __init__(self, kind: str, in_dim: int, hidden: int = 32, layers: int = 2,
                 num_relations: int = 1, dropout: float = 0.2, gat_heads: int = 4) -> None:
        super().__init__()
        if kind not in MODEL_KINDS:
            raise ValueError(f"Unknown model kind {kind!r}")
        self.kind = kind
        self.inp = nn.Linear(in_dim, hidden)
        convs: list[nn.Module] = []
        for _ in range(layers):
            if kind == "rgcn":
                convs.append(RGCNConv(hidden, hidden, num_relations=num_relations, aggr="mean"))
            elif kind == "sage":
                convs.append(SAGEConv(hidden, hidden, aggr="mean"))
            elif kind == "gat":
                convs.append(GATConv(hidden, hidden, heads=gat_heads, concat=False))
            else:
                convs.append(nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(convs)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(in_dim + hidden * (layers + 1), hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """x: [total_nodes, F] (disjoint union of query graphs). Returns [total_nodes] scores."""
        h = torch.relu(self.inp(x))
        hs = [x, h]
        for conv in self.convs:
            if self.kind == "rgcn":
                h = conv(h, edge_index, edge_type)
            elif self.kind == "mlp":
                h = conv(h)
            else:
                h = conv(h, edge_index)
            h = self.drop(torch.relu(h))
            hs.append(h)
        return self.head(torch.cat(hs, dim=-1)).squeeze(-1)


def listwise_loss(logits: Tensor, gold: Tensor) -> Tensor:
    """Multi-positive listwise softmax, averaged over queries.

    logits, gold: [n_queries, n_clauses]; gold is a 0/1 mask with >= 1 positive per row.
    L(q) = -(1/|G_q|) sum_{g in G_q} log softmax(s_q)_g
    """
    logp = torch.log_softmax(logits, dim=-1)
    n_pos = gold.sum(dim=-1)
    if (n_pos < 1).any():
        raise ValueError("Every query needs at least one gold clause")
    return (-(logp * gold).sum(dim=-1) / n_pos).mean()
