"""
gat_model.py
Pure-numpy Graph Attention Network (GAT) for node classification.
No PyTorch/TF dependency — runs anywhere, trains fast.

Architecture:
  Input node features (6)
  → GAT Layer 1: multi-head attention aggregation (hidden=32, heads=4)
  → GAT Layer 2: aggregation → fraud probability per node (out=1)
  → Sigmoid → fraud score
"""

import numpy as np
import json
import joblib
from pathlib import Path


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def relu(x):
    return np.maximum(0, x)


class GATLayer:
    """Single Graph Attention Layer."""

    def __init__(self, in_dim, out_dim, n_heads=4, seed=0):
        rng = np.random.RandomState(seed)
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        head_dim = out_dim // n_heads

        # Weight matrices per head
        scale = np.sqrt(2.0 / (in_dim + head_dim))
        self.W  = rng.randn(n_heads, in_dim, head_dim).astype(np.float32) * scale
        # Attention vectors
        self.a  = rng.randn(n_heads, 2 * head_dim).astype(np.float32) * 0.01
        self.bias = np.zeros((out_dim,), dtype=np.float32)

    def forward(self, X, edge_index):
        """
        X:          (N, in_dim)
        edge_index: (E, 2)  [src, dst]
        Returns:    (N, out_dim)
        """
        N = X.shape[0]
        head_dim = self.out_dim // self.n_heads
        out = np.zeros((N, self.out_dim), dtype=np.float32)

        for h in range(self.n_heads):
            Wh = X @ self.W[h]              # (N, head_dim)
            h_start = h * head_dim
            h_end   = h_start + head_dim

            if edge_index.shape[0] == 0:
                out[:, h_start:h_end] = Wh
                continue

            src = edge_index[:, 0]
            dst = edge_index[:, 1]

            # Attention scores
            concat = np.concatenate([Wh[src], Wh[dst]], axis=1)  # (E, 2*head_dim)
            e = relu(concat @ self.a[h])                           # (E,)

            # Softmax per destination node
            alpha = np.zeros(len(src), dtype=np.float32)
            for i in range(N):
                mask = (dst == i)
                if mask.sum() > 0:
                    alpha[mask] = softmax(e[mask])

            # Aggregate
            for i in range(N):
                mask = (dst == i)
                if mask.sum() > 0:
                    agg = (alpha[mask, None] * Wh[src[mask]]).sum(axis=0)
                    out[i, h_start:h_end] = agg
                else:
                    out[i, h_start:h_end] = Wh[i]

        return relu(out + self.bias)


class FraudGAT:
    """Two-layer GAT for binary node classification (fraud/not-fraud)."""

    def __init__(self, in_dim=6, hidden=32, n_heads=4, lr=0.01, seed=42):
        self.layer1 = GATLayer(in_dim,  hidden,  n_heads, seed)
        self.layer2 = GATLayer(hidden,  hidden // 2, 2,   seed + 1)
        rng = np.random.RandomState(seed + 2)
        self.W_out  = rng.randn(hidden // 2, 1).astype(np.float32) * 0.1
        self.b_out  = np.zeros(1, dtype=np.float32)
        self.lr     = lr

    def forward(self, X, edge_index):
        h1 = self.layer1.forward(X, edge_index)
        h2 = self.layer2.forward(h1, edge_index)
        logits = h2 @ self.W_out + self.b_out   # (N, 1)
        probs  = sigmoid(logits).flatten()       # (N,)
        return probs, h2

    def train(self, X, y, edge_index, epochs=80, patience=10):
        """Mini-training loop with binary cross-entropy."""
        best_loss = float("inf")
        patience_count = 0
        history = []

        # Class weights for imbalance
        pos = y.sum()
        neg = len(y) - pos
        w_pos = neg / (pos + 1e-9)

        for epoch in range(epochs):
            probs, h2 = self.forward(X, edge_index)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)

            # Weighted BCE loss
            loss = -(
                w_pos * y * np.log(probs) +
                (1 - y) * np.log(1 - probs)
            ).mean()

            # Gradient on output layer (simplified backprop)
            dL = (probs - y) / len(y)
            dL[y == 1] *= w_pos
            grad_W_out = h2.T @ dL[:, None]
            grad_b_out = dL.mean()

            self.W_out -= self.lr * grad_W_out
            self.b_out -= self.lr * grad_b_out

            history.append(float(loss))

            if loss < best_loss - 1e-4:
                best_loss = loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

            if epoch % 20 == 0:
                preds = (probs >= 0.5).astype(int)
                acc = (preds == y).mean()
                fraud_recall = (preds[y == 1] == 1).mean() if y.sum() > 0 else 0
                print(f"  Epoch {epoch:3d} | loss {loss:.4f} | acc {acc:.3f} | fraud recall {fraud_recall:.3f}")

        return history

    def predict_proba(self, X, edge_index):
        probs, _ = self.forward(X, edge_index)
        return probs

    def save(self, path: str):
        data = {
            "layer1": {
                "W": self.layer1.W.tolist(),
                "a": self.layer1.a.tolist(),
                "bias": self.layer1.bias.tolist(),
                "in_dim": self.layer1.in_dim,
                "out_dim": self.layer1.out_dim,
                "n_heads": self.layer1.n_heads,
            },
            "layer2": {
                "W": self.layer2.W.tolist(),
                "a": self.layer2.a.tolist(),
                "bias": self.layer2.bias.tolist(),
                "in_dim": self.layer2.in_dim,
                "out_dim": self.layer2.out_dim,
                "n_heads": self.layer2.n_heads,
            },
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        model = cls.__new__(cls)
        for layer_name in ("layer1", "layer2"):
            d = data[layer_name]
            layer = GATLayer.__new__(GATLayer)
            layer.W = np.array(d["W"], dtype=np.float32)
            layer.a = np.array(d["a"], dtype=np.float32)
            layer.bias = np.array(d["bias"], dtype=np.float32)
            layer.in_dim = d["in_dim"]
            layer.out_dim = d["out_dim"]
            layer.n_heads = d["n_heads"]
            setattr(model, layer_name, layer)
        model.W_out = np.array(data["W_out"], dtype=np.float32)
        model.b_out = np.array(data["b_out"], dtype=np.float32)
        model.lr = 0.01
        return model


def train_and_save():
    from graph_builder import build_transaction_graph, graph_to_arrays
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.ensemble import GradientBoostingClassifier

    print("Building transaction graph...")
    G, stats = build_transaction_graph(n_accounts=400, n_legit_txns=2000, n_fraud_rings=10)
    print("Graph stats:", stats)

    X, y, edge_index, edge_attr, node_idx, means, stds = graph_to_arrays(G)
    print(f"GAT embedding on {len(y)} nodes, {y.sum()} fraud ({y.mean()*100:.1f}%)...")

    # Step 1: GAT produces graph-aware embeddings
    gat = FraudGAT(in_dim=X.shape[1], hidden=32, n_heads=4, lr=0.008)
    gat.train(X, y.astype(np.float32), edge_index, epochs=60, patience=10)
    _, gat_embeddings = gat.forward(X, edge_index)  # (N, 16)

    # Step 2: Concat raw features + GAT embeddings -> GBM classifier
    # This is the production pattern: GNN for structural features, GB for classification
    X_augmented = np.hstack([X, gat_embeddings])
    print(f"Training GBM on augmented features ({X_augmented.shape[1]} dims)...")

    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    clf.fit(X_augmented, y)
    probs = clf.predict_proba(X_augmented)[:, 1]
    preds = (probs >= 0.4).astype(int)

    auc = roc_auc_score(y, probs)
    print(f"\nFinal ROC-AUC: {auc:.4f}")
    print(classification_report(y, preds, target_names=["Legit", "Fraud"], zero_division=0))

    import joblib
    joblib.dump(clf, "model/gbm_classifier.pkl")

    Path("model").mkdir(exist_ok=True)
    gat.save("model/gat_weights.json")
    joblib.dump(clf, "model/gbm_classifier.pkl")
    np.save("model/norm_means.npy", means)
    np.save("model/norm_stds.npy",  stds)

    import pickle
    with open("model/graph.pkl", "wb") as f:
        pickle.dump(G, f)

    metrics = {
        "roc_auc": round(auc, 4),
        "n_nodes": stats["n_nodes"],
        "n_edges": stats["n_edges"],
        "fraud_node_pct": stats["fraud_node_pct"],
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Model and graph saved to model/")
    return gat, clf, G, X, y, edge_index, probs


if __name__ == "__main__":
    train_and_save()
