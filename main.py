import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from gnn.train import train_model
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from gnn.visualize import plot_loss_curve, visualize_node_embeddings

list_of_networkx_graphs = []
feature_columns = ['Node_Nos.', 'X_Co-ordinate', 'Y_Co-ordinate', 'Z_Co-ordinate']
env_columns = ['Sunhour_Per_Sqmt_Facad', 'Solar_Radiatiion_Per_Sqmt_Facad']

# Example: graph-level targets for each building
graph_targets = [0.75]

for i in range(1):
    try:
        node_df = pd.read_csv(f"building_{i}_nodes.csv")
        try:
            edge_df = pd.read_csv(f"building_{i}_edges.csv")
        except FileNotFoundError:
            edge_df = pd.DataFrame(columns=['source', 'target'])
    except Exception as e:
        print(f"Error reading files for iteration {i}: {e}")
        continue

    try:
        G = nx.Graph()
        for idx, row in node_df.iterrows():
            G.add_node(row['Node_Nos.'],
                       x=row[feature_columns].tolist(),
                       y=row[env_columns].tolist())
        for idx, row in edge_df.iterrows():
            G.add_edge(row['source'], row['target'])
        G.graph['graph_y'] = graph_targets[i]
        list_of_networkx_graphs.append(G)
    except Exception as e:
        print(f"Error processing graph for iteration {i}: {e}")
        continue

# Convert networkx graphs to PyTorch Geometric format
data_list = []
for G in list_of_networkx_graphs:
    data = from_networkx(G)
    data.x = torch.stack([torch.tensor(G.nodes[n]['x'], dtype=torch.float) for n in G.nodes])
    if G.edges:
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    data.edge_index = edge_index
    # Fix: Make sure the target has the right shape - should be scalar for MSE loss
    data.graph_y = torch.tensor(G.graph['graph_y'], dtype=torch.float)  # Shape: [] (scalar)
    data_list.append(data)

# Train the model
input_dim = len(feature_columns)  # Should be 4 based on your feature_columns
model, loss_history = train_model(data_list, input_dim)
model.eval()

# Plot loss curve
plot_loss_curve(loss_history)


if list_of_networkx_graphs and data_list:
    G = list_of_networkx_graphs[0]
    data = data_list[0]
    visualize_node_embeddings(model, G, data)

# Print some debugging information
print(f"Number of graphs: {len(list_of_networkx_graphs)}")
print(f"Number of data objects: {len(data_list)}")
if data_list:
    print(f"First graph - Nodes: {data_list[0].x.shape[0]}, Edges: {data_list[0].edge_index.shape[1]}")
print(f"Model type: {type(model)}")
