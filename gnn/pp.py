
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# Training Loss Visualization
# ============================================================
def plot_loss_curve(loss_history):
    """
    Plot training loss curve.
    """
    plt.figure(figsize=(6,4))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ============================================================
# Node Embeddings
# ============================================================
def visualize_node_embeddings(model, G, data):
    """
    Visualize node embeddings from a trained GNN model.
    """
    model.eval()
    with torch.no_grad():
        try:
            node_embeds = model.conv1(data.x, data.edge_index)
        except AttributeError:
            try:
                node_embeds = model(data.x, data.edge_index, getattr(data, "batch", None))
            except Exception:
                raise AttributeError(
                    "Cannot extract node embeddings. "
                    "Check your model definition."
                )

        node_embeds = node_embeds.detach().cpu()
        if len(node_embeds.shape) > 1:
            node_embeds = node_embeds[:, 0].numpy()
        else:
            node_embeds = node_embeds.numpy()

    pos = {n: (G.nodes[n]['x'][1], G.nodes[n]['x'][2]) for n in G.nodes}

    if G.edges:
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_embeds, cmap=plt.cm.viridis, node_size=300
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Node-level Embeddings from GNN Model")

    if nodes:
        plt.colorbar(nodes, label="Node Embedding Value")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_embedding_clusters(node_embeds):
    """
    Reduce high-dimensional embeddings with PCA and visualize clusters.
    """
    node_embeds = node_embeds.detach().cpu().numpy()
    if node_embeds.shape[1] > 2:
        reduced = PCA(n_components=2).fit_transform(node_embeds)
    else:
        reduced = node_embeds

    plt.figure(figsize=(6,6))
    plt.scatter(reduced[:,0], reduced[:,1], c="blue", alpha=0.6)
    plt.title("Node Embedding Clusters (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()


# ============================================================
# Predictions vs Targets
# ============================================================
def plot_predictions_vs_targets(preds, targets):
    """
    Scatter plot of predicted vs. target values.
    """
    preds = [float(p) for p in preds]
    targets = [float(t) for t in targets]

    plt.figure(figsize=(5,5))
    plt.scatter(targets, preds, alpha=0.7, edgecolor="k")
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], "r--")
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Predicted vs Target Values")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_residuals(preds, targets):
    """
    Histogram of residuals (prediction - target).
    """
    residuals = [float(p) - float(t) for p, t in zip(preds, targets)]
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=20, color="purple", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual (Prediction - Target)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()


# ============================================================
# Graph-level Prediction Visualization
# ============================================================
def visualize_graph_prediction(G, prediction):
    """
    Color all nodes in the graph by a graph-level prediction value.
    """
    pos = {n: (G.nodes[n]['x'][1], G.nodes[n]['x'][2]) for n in G.nodes}
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=[prediction]*len(G.nodes), cmap=plt.cm.coolwarm, node_size=300
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)
    plt.colorbar(nodes, label="Graph Prediction")
    plt.title("Graph-level Prediction Visualization")
    plt.axis("off")
    plt.show()
