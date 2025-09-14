import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_node_embeddings(model, data, layout="tsne", save_file=None):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        embeddings = model.conv1(x, edge_index)
        embeddings = embeddings.detach().cpu().numpy()
    if layout == "tsne":
        reducer = TSNE(n_components=2, random_state=42) 
    else:  # pca
        reducer = PCA(n_components=2) 
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    if hasattr(data, 'y'):
        classes = data.y.cpu().numpy()
        unique_classes = np.unique(classes)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        for i, cls in enumerate(unique_classes):
            idx = classes == cls
            plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                      c=[colors[i]], label=f"Class {cls}", alpha=0.7, s=100)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
    
    plt.title(f"Node Embeddings Visualization ({layout.upper()})", fontsize=14)
    plt.legend()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_model_predictions(G, model, data, pos=None, save_file=None):
    # Get predictions
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).cpu().numpy()
    
    plt.figure(figsize=(16, 14))
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    node_colors = []
    for i, node in enumerate(G.nodes()):
        node_name = G.nodes[node].get('label', '')
        # Force Exits to always be red regardless of prediction
        if 'Exit' in str(node_name):
            node_colors.append('red')
        elif pred[i] == 0:
            node_colors.append('lightblue')
        else:
            node_colors.append('red')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Use original node names as labels
    labels = {}
    for i, node in enumerate(G.nodes()):
        # Use the full original node name
        labels[node] = G.nodes[node].get('label', str(node))
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)  # Smaller font size for full names
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Regular Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Critical Node')
    ]
    plt.legend(handles=legend_elements)
    plt.axis('off')
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def compare_untrained_vs_trained(G, untrained_model, trained_model, data, pos=None, save_file=None):
    """Compare predictions from untrained and trained models with original node names."""
    # Get predictions from both models
    untrained_model.eval()
    trained_model.eval()
    
    with torch.no_grad():
        # Untrained model predictions
        out_untrained = untrained_model(data)
        pred_untrained = out_untrained.argmax(dim=1).cpu().numpy()
        
        # Trained model predictions
        out_trained = trained_model(data)
        pred_trained = out_trained.argmax(dim=1).cpu().numpy()
    
    # Get positions for nodes
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Use original node names as labels
    labels = {}
    for i, node in enumerate(G.nodes()):
        # Use the full original node name
        labels[node] = G.nodes[node].get('label', str(node))
    
    # Plot untrained model predictions
    ax1.set_title("Untrained Model Predictions", fontsize=16)
    for i, node in enumerate(G.nodes()):
        node_name = G.nodes[node].get('label', '')
        # Force Exits to always be red regardless of prediction
        if 'Exit' in str(node_name):
            color = 'red'
        else:
            color = 'red' if pred_untrained[i] == 1 else 'lightblue'
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, 
                             node_size=700, ax=ax1)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', 
                         connectionstyle='arc3,rad=0.1', ax=ax1)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax1)  # Smaller font for full names
    ax1.axis('off')
    
    # Plot trained model predictions
    ax2.set_title("Trained Model Predictions", fontsize=16)
    for i, node in enumerate(G.nodes()):
        node_name = G.nodes[node].get('label', '')
        # Force Exits to always be red regardless of prediction
        if 'Exit' in str(node_name):
            color = 'red'
        else:
            color = 'red' if pred_trained[i] == 1 else 'lightblue'
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, 
                             node_size=700, ax=ax2)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', 
                         connectionstyle='arc3,rad=0.1', ax=ax2)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax2)  # Smaller font for full names
    ax2.axis('off')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
              markersize=10, label='Regular Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
              markersize=10, label='Critical Node')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_node_types(G, save_file=None):
    """Visualize graph colored by original node types (Exit, Corridor, etc.)"""
    plt.figure(figsize=(16, 14))
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'Unknown')
        if node_type == 'Exit':
            node_colors.append('red')
        elif node_type == 'Corridor':
            node_colors.append('orange')
        elif node_type == 'Door':
            node_colors.append('green')
        else:
            node_colors.append('lightblue')
    
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=700)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Exit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Corridor'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Door'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Unit')
    ]
    plt.legend(handles=legend_elements)
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()