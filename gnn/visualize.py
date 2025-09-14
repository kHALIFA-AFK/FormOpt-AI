import torch
import networkx as nx
import matplotlib.pyplot as plt

def plot_loss_curve(loss_history):
    """
    Plot training loss curve.

    Args:
        loss_history (list or np.array): Training loss values per epoch
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


def visualize_node_embeddings(model, G, data):
    """
    Visualize node embeddings from a trained GNN model.
    
    Args:
        model: Trained GNN model
        G (networkx.Graph): Original graph
        data (torch_geometric.data.Data): Processed data for the graph
    """
    model.eval()
    with torch.no_grad():
        try:
            node_embeds = model.conv1(data.x, data.edge_index)
        except AttributeError:
            try:
                node_embeds = model(data.x, data.edge_index, data.batch)
            except Exception:
                raise AttributeError(
                    "Cannot find appropriate method to get node embeddings. "
                    "Check your model definition."
                )
        
        if len(node_embeds.shape) > 1:
            node_embeds = node_embeds[:, 0].cpu().numpy()
        else:
            node_embeds = node_embeds.cpu().numpy()

    # Get positions for plotting (using X, Y coordinates)
    pos = {n: (G.nodes[n]['x'][1], G.nodes[n]['x'][2]) for n in G.nodes}

    # Draw edges
    if G.edges:
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

    # Draw nodes
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
