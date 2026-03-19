"""
main.py  —  FraudGraph FastAPI server
Endpoints:
  GET  /              → dashboard UI
  GET  /health        → service status
  POST /analyze       → run full graph fraud analysis
  GET  /graph-stats   → current graph statistics
  GET  /top-fraud     → top suspicious accounts
"""

import json
import pickle
import joblib
import numpy as np
import networkx as nx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from graph_builder import build_transaction_graph, graph_to_arrays, NODE_FEATURES
from gat_model import FraudGAT
from explainer import detect_fraud_rings, explain_account

app = FastAPI(title="FraudGraph", version="1.0.0")

MODEL_DIR = Path("model")

# ── Load artifacts at startup ─────────────────────────────────────────────────
print("Loading FraudGraph model artifacts...")
gat = FraudGAT.load(str(MODEL_DIR / "gat_weights.json"))
clf = joblib.load(MODEL_DIR / "gbm_classifier.pkl")
means = np.load(MODEL_DIR / "norm_means.npy")
stds  = np.load(MODEL_DIR / "norm_stds.npy")
metrics = json.load(open(MODEL_DIR / "metrics.json"))

with open(MODEL_DIR / "graph.pkl", "rb") as f:
    G: nx.DiGraph = pickle.load(f)

print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def run_inference(G: nx.DiGraph):
    """Run GAT + GBM inference on the full graph."""
    X, y, edge_index, _, node_idx, _, _ = graph_to_arrays(G)
    _, gat_emb = gat.forward(X, edge_index)
    X_aug = np.hstack([X, gat_emb])
    probs = clf.predict_proba(X_aug)[:, 1]
    nodes = sorted(G.nodes())
    fraud_scores = {int(nodes[i]): round(float(probs[i]), 4) for i in range(len(nodes))}
    return fraud_scores, X, y, edge_index


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "GAT + GradientBoosting",
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "model_roc_auc": metrics["roc_auc"],
    }


@app.get("/graph-stats")
async def graph_stats():
    fraud_nodes = [n for n, d in G.nodes(data=True) if d.get("is_fraud") == 1]
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "fraud_nodes": len(fraud_nodes),
        "fraud_node_pct": round(len(fraud_nodes) / G.number_of_nodes() * 100, 1),
        "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "model_roc_auc": metrics["roc_auc"],
    }


@app.post("/analyze")
async def analyze():
    """Full graph fraud analysis — runs GAT inference + ring detection + NIM explanations."""
    fraud_scores, X, y, edge_index = run_inference(G)

    # Top suspicious accounts
    sorted_accounts = sorted(fraud_scores.items(), key=lambda x: x[1], reverse=True)
    top_suspicious = []
    for account_id, score in sorted_accounts[:15]:
        if score < 0.3:
            break
        node = G.nodes.get(account_id, {})
        neighbors = list(G.predecessors(account_id)) + list(G.successors(account_id))
        fraud_nbrs = [n for n in neighbors if fraud_scores.get(n, 0) >= 0.5]

        top_suspicious.append({
            "account_id": account_id,
            "fraud_score": score,
            "verdict": "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW",
            "account_age_days": int(node.get("account_age_days", 0)),
            "degree_in": int(node.get("degree_in", 0)),
            "degree_out": int(node.get("degree_out", 0)),
            "fraud_neighbor_ratio": round(float(node.get("fraud_neighbor_ratio", 0)), 3),
            "velocity_24h": int(node.get("velocity_24h", 0)),
            "is_ground_truth_fraud": int(node.get("is_fraud", 0)),
            "explanation": explain_account(
                account_id, G, score,
                [n for n in neighbors if fraud_scores.get(n, 0) >= 0.4]
            )
        })

    # Fraud rings
    rings = detect_fraud_rings(G, fraud_scores, threshold=0.45)

    # Summary stats
    high_risk   = sum(1 for s in fraud_scores.values() if s >= 0.7)
    medium_risk = sum(1 for s in fraud_scores.values() if 0.4 <= s < 0.7)
    low_risk    = sum(1 for s in fraud_scores.values() if s < 0.4)

    return JSONResponse({
        "summary": {
            "total_accounts": G.number_of_nodes(),
            "total_transactions": G.number_of_edges(),
            "high_risk_accounts":   high_risk,
            "medium_risk_accounts": medium_risk,
            "low_risk_accounts":    low_risk,
            "fraud_rings_detected": len(rings),
            "model_roc_auc":        metrics["roc_auc"],
        },
        "top_suspicious_accounts": top_suspicious,
        "fraud_rings": rings[:5],
    })


@app.get("/top-fraud")
async def top_fraud(threshold: float = 0.5, limit: int = 20):
    fraud_scores, _, _, _ = run_inference(G)
    results = [
        {"account_id": k, "fraud_score": v,
         "verdict": "HIGH" if v >= 0.7 else "MEDIUM"}
        for k, v in sorted(fraud_scores.items(), key=lambda x: x[1], reverse=True)
        if v >= threshold
    ]
    return {"accounts": results[:limit], "total_flagged": len(results)}
