"""
graph_builder.py
Builds a synthetic financial transaction graph.
- Nodes  = accounts (with account-level features)
- Edges  = transactions (with edge features)
- Fraud patterns injected: rings, funnels, new-node explosions
"""

import numpy as np
import networkx as nx
import json
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)


def build_transaction_graph(
    n_accounts: int = 500,
    n_legit_txns: int = 3000,
    n_fraud_rings: int = 8,
    ring_size_range: tuple = (4, 8),
):
    G = nx.DiGraph()

    # ── Add account nodes ─────────────────────────────────────────────────────
    for i in range(n_accounts):
        G.add_node(i, **{
            "avg_txn_amount": float(np.random.lognormal(8, 1.5)),
            "account_age_days": int(np.random.randint(30, 3650)),
            "degree_in": 0,
            "degree_out": 0,
            "fraud_neighbor_ratio": 0.0,
            "velocity_24h": int(np.random.poisson(2)),
            "is_fraud": 0,
        })

    edges = []

    # ── Legitimate transactions ───────────────────────────────────────────────
    for _ in range(n_legit_txns):
        src = random.randint(0, n_accounts - 1)
        dst = random.randint(0, n_accounts - 1)
        if src == dst:
            continue
        edges.append({
            "src": src, "dst": dst,
            "amount": float(np.random.lognormal(8, 1.2)),
            "time_gap_min": float(np.random.exponential(60)),
            "same_device": int(np.random.random() < 0.15),
            "same_ip": int(np.random.random() < 0.10),
            "is_fraud_edge": 0,
        })

    # ── Fraud pattern 1: Circular rings ──────────────────────────────────────
    fraud_nodes = set()
    fraud_edges_list = []

    for _ in range(n_fraud_rings):
        ring_size = random.randint(*ring_size_range)
        ring_accounts = random.sample(range(n_accounts), ring_size)
        for i, node in enumerate(ring_accounts):
            G.nodes[node]["is_fraud"] = 1
            fraud_nodes.add(node)
        for i in range(ring_size):
            src = ring_accounts[i]
            dst = ring_accounts[(i + 1) % ring_size]
            e = {
                "src": src, "dst": dst,
                "amount": float(np.random.uniform(9000, 25000)),
                "time_gap_min": float(np.random.uniform(0.5, 5)),
                "same_device": 1,
                "same_ip": int(np.random.random() < 0.7),
                "is_fraud_edge": 1,
            }
            edges.append(e)
            fraud_edges_list.append(e)

    # ── Fraud pattern 2: Funnels (many → one) ────────────────────────────────
    for _ in range(n_fraud_rings // 2):
        n_sources = random.randint(5, 12)
        sink = random.randint(0, n_accounts - 1)
        sources = random.sample([x for x in range(n_accounts) if x != sink], n_sources)
        G.nodes[sink]["is_fraud"] = 1
        fraud_nodes.add(sink)
        for src in sources:
            G.nodes[src]["is_fraud"] = 1
            fraud_nodes.add(src)
            e = {
                "src": src, "dst": sink,
                "amount": float(np.random.uniform(500, 3000)),
                "time_gap_min": float(np.random.uniform(1, 10)),
                "same_device": int(np.random.random() < 0.5),
                "same_ip": int(np.random.random() < 0.4),
                "is_fraud_edge": 1,
            }
            edges.append(e)
            fraud_edges_list.append(e)

    # ── Fraud pattern 3: New node explosions ─────────────────────────────────
    for _ in range(n_fraud_rings):
        new_acct = random.randint(0, n_accounts - 1)
        G.nodes[new_acct]["account_age_days"] = random.randint(1, 7)
        G.nodes[new_acct]["is_fraud"] = 1
        fraud_nodes.add(new_acct)
        n_rapid = random.randint(8, 15)
        targets = random.sample([x for x in range(n_accounts) if x != new_acct], n_rapid)
        for dst in targets:
            e = {
                "src": new_acct, "dst": dst,
                "amount": float(np.random.uniform(100, 5000)),
                "time_gap_min": float(np.random.uniform(0.1, 2)),
                "same_device": 1,
                "same_ip": 1,
                "is_fraud_edge": 1,
            }
            edges.append(e)
            fraud_edges_list.append(e)

    # ── Add edges to graph ────────────────────────────────────────────────────
    for e in edges:
        G.add_edge(e["src"], e["dst"],
                   amount=e["amount"],
                   time_gap_min=e["time_gap_min"],
                   same_device=e["same_device"],
                   same_ip=e["same_ip"],
                   is_fraud_edge=e["is_fraud_edge"])

    # ── Update degree features ────────────────────────────────────────────────
    for node in G.nodes():
        G.nodes[node]["degree_in"]  = G.in_degree(node)
        G.nodes[node]["degree_out"] = G.out_degree(node)
        neighbors = list(G.predecessors(node)) + list(G.successors(node))
        if neighbors:
            fraud_nbrs = sum(G.nodes[n]["is_fraud"] for n in neighbors)
            G.nodes[node]["fraud_neighbor_ratio"] = round(fraud_nbrs / len(neighbors), 4)

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_fraud_nodes": len(fraud_nodes),
        "n_fraud_edges": len(fraud_edges_list),
        "fraud_node_pct": round(len(fraud_nodes) / G.number_of_nodes() * 100, 1),
        "fraud_edge_pct": round(len(fraud_edges_list) / max(G.number_of_edges(), 1) * 100, 1),
    }
    return G, stats


NODE_FEATURES = [
    "avg_txn_amount", "account_age_days", "degree_in",
    "degree_out", "fraud_neighbor_ratio", "velocity_24h"
]


def graph_to_arrays(G):
    """Convert NetworkX graph to numpy arrays for the GAT model."""
    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    X = np.zeros((N, len(NODE_FEATURES)), dtype=np.float32)
    y = np.zeros(N, dtype=np.int32)

    for i, node in enumerate(nodes):
        attrs = G.nodes[node]
        for j, feat in enumerate(NODE_FEATURES):
            X[i, j] = float(attrs.get(feat, 0))
        y[i] = int(attrs.get("is_fraud", 0))

    # Normalise features
    means = X.mean(axis=0) + 1e-8
    stds  = X.std(axis=0)  + 1e-8
    X_norm = (X - means) / stds

    # Adjacency list
    edge_index = []
    edge_attr  = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_idx[u], node_idx[v]])
        edge_attr.append([
            data.get("amount", 0) / 10000,
            data.get("time_gap_min", 0) / 60,
            float(data.get("same_device", 0)),
            float(data.get("same_ip", 0)),
        ])

    edge_index = np.array(edge_index, dtype=np.int32) if edge_index else np.zeros((0, 2), dtype=np.int32)
    edge_attr  = np.array(edge_attr,  dtype=np.float32) if edge_attr  else np.zeros((0, 4), dtype=np.float32)

    return X_norm, y, edge_index, edge_attr, node_idx, means, stds


if __name__ == "__main__":
    G, stats = build_transaction_graph()
    print("Graph built:", stats)
    X, y, ei, ea, _, _, _ = graph_to_arrays(G)
    print(f"Node features: {X.shape}, Labels: {y.shape}, Edges: {ei.shape}")
    print(f"Fraud nodes: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")
