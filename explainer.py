"""
explainer.py
Uses NVIDIA NIM API to generate plain-English explanations
of detected fraud rings and suspicious accounts.
Falls back to rule-based explanation if NIM key not set.
"""

import os
import json
import requests
import networkx as nx
import numpy as np


NIM_API_KEY = os.getenv("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", "")
NIM_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL   = "meta/llama-3.1-8b-instruct"


def _call_nim(prompt: str, max_tokens: int = 400) -> str:
    if not NIM_API_KEY:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {NIM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": NIM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial fraud analyst at a major bank. "
                        "Explain detected fraud patterns clearly and concisely for a compliance team. "
                        "Be specific about the suspicious signals. Keep responses under 150 words."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        resp = requests.post(NIM_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None


def _rule_based_explanation(account_id: int, G: nx.DiGraph, fraud_score: float, risk_signals: list) -> str:
    node = G.nodes.get(account_id, {})
    age  = node.get("account_age_days", 0)
    deg_out = node.get("degree_out", 0)
    deg_in  = node.get("degree_in",  0)
    fnr     = node.get("fraud_neighbor_ratio", 0)

    lines = [f"Account {account_id} flagged with {fraud_score*100:.0f}% fraud probability."]

    if age < 30:
        lines.append(f"Account is only {age} days old — typical of mule accounts opened for a single fraud operation.")
    if deg_out > 8:
        lines.append(f"Sent transactions to {deg_out} different accounts — consistent with a money dispersion pattern.")
    if deg_in > 8:
        lines.append(f"Received funds from {deg_in} accounts — consistent with a funnel/aggregation pattern.")
    if fnr > 0.5:
        lines.append(f"{fnr*100:.0f}% of this account's transaction partners are also flagged as suspicious.")
    for sig in risk_signals:
        lines.append(sig)

    return " ".join(lines)


def explain_fraud_ring(ring_nodes: list, G: nx.DiGraph, fraud_scores: dict) -> str:
    """Generate explanation for a detected fraud ring."""
    if len(ring_nodes) < 2:
        return "Isolated suspicious account — insufficient ring structure for pattern analysis."

    ring_amounts = []
    time_gaps    = []
    same_device_count = 0

    for i in range(len(ring_nodes)):
        src = ring_nodes[i]
        dst = ring_nodes[(i + 1) % len(ring_nodes)]
        if G.has_edge(src, dst):
            data = G.edges[src, dst]
            ring_amounts.append(data.get("amount", 0))
            time_gaps.append(data.get("time_gap_min", 0))
            same_device_count += int(data.get("same_device", 0))

    avg_amount = np.mean(ring_amounts) if ring_amounts else 0
    avg_gap    = np.mean(time_gaps)    if time_gaps    else 0
    avg_score  = np.mean([fraud_scores.get(n, 0) for n in ring_nodes])

    prompt = f"""
Detected fraud ring with {len(ring_nodes)} accounts.
- Average transaction amount: ₹{avg_amount:,.0f}
- Average time between transactions: {avg_gap:.1f} minutes
- Transactions using same device: {same_device_count}/{max(len(ring_nodes)-1, 1)}
- Average fraud probability across ring members: {avg_score*100:.0f}%
- Ring account IDs: {ring_nodes}

Explain this pattern in 2-3 sentences for a compliance team. 
What type of fraud does this represent and what makes it suspicious?
"""

    nim_response = _call_nim(prompt)
    if nim_response:
        return nim_response

    # Fallback
    return (
        f"Circular money movement detected among {len(ring_nodes)} accounts. "
        f"Average transaction of ₹{avg_amount:,.0f} with {avg_gap:.1f}-minute gaps "
        f"({'same device used' if same_device_count > 0 else 'different devices'}) "
        f"suggests coordinated layering — funds cycling through accounts to obscure origin. "
        f"All members flagged with {avg_score*100:.0f}% average fraud probability."
    )


def explain_account(account_id: int, G: nx.DiGraph,
                    fraud_score: float, subgraph_nodes: list) -> str:
    """Generate explanation for a single suspicious account."""
    node = G.nodes.get(account_id, {})
    age     = node.get("account_age_days", 0)
    deg_out = node.get("degree_out", 0)
    deg_in  = node.get("degree_in",  0)
    fnr     = node.get("fraud_neighbor_ratio", 0)
    vel     = node.get("velocity_24h", 0)

    risk_signals = []
    if age < 30:      risk_signals.append(f"New account ({age} days old)")
    if deg_out > 8:   risk_signals.append(f"High fan-out ({deg_out} outbound txns)")
    if deg_in > 8:    risk_signals.append(f"High fan-in ({deg_in} inbound txns)")
    if fnr > 0.4:     risk_signals.append(f"{fnr*100:.0f}% suspicious neighbours")
    if vel > 5:       risk_signals.append(f"High velocity ({vel} txns/24h)")

    prompt = f"""
Suspicious account {account_id} flagged by graph anomaly detection.
- Fraud probability score: {fraud_score*100:.0f}%
- Account age: {age} days
- Outbound transactions: {deg_out}
- Inbound transactions: {deg_in}  
- Suspicious neighbor ratio: {fnr*100:.0f}%
- Transaction velocity (24h): {vel}
- Risk signals: {', '.join(risk_signals) if risk_signals else 'structural graph anomaly'}
- Connected to {len(subgraph_nodes)} accounts in its fraud subgraph

Write a 2-sentence fraud alert for a compliance analyst.
"""

    nim_response = _call_nim(prompt)
    if nim_response:
        return nim_response

    return _rule_based_explanation(account_id, G, fraud_score, risk_signals)


def detect_fraud_rings(G: nx.DiGraph, fraud_scores: dict, threshold: float = 0.5) -> list:
    """Find suspicious cycles in the transaction graph."""
    suspicious = {n for n, s in fraud_scores.items() if s >= threshold}

    fraud_subgraph = G.subgraph(suspicious).copy()
    rings = []

    try:
        cycles = list(nx.simple_cycles(fraud_subgraph))
        for cycle in cycles:
            if 3 <= len(cycle) <= 12:
                avg_score = np.mean([fraud_scores.get(n, 0) for n in cycle])
                rings.append({
                    "nodes": cycle,
                    "size": len(cycle),
                    "avg_fraud_score": round(float(avg_score), 3),
                    "explanation": explain_fraud_ring(cycle, G, fraud_scores)
                })
    except Exception:
        pass

    rings.sort(key=lambda r: r["avg_fraud_score"], reverse=True)
    return rings[:10]
