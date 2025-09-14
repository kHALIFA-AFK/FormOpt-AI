import torch
from gnn.model import EvacuationGCN
from torch_geometric.utils import to_networkx
import networkx as nx

def train_model(data, num_classes=2):
    try:
        model = EvacuationGCN(input_dim=data.num_node_features, hidden_dim=16, output_dim=num_classes)
        
        untrained_model = EvacuationGCN(input_dim=data.num_node_features, hidden_dim=16, output_dim=num_classes)
        untrained_model.load_state_dict(model.state_dict())
        
        # Try a smaller learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Increase learning rate
        
        # Weight the loss function to care more about critical nodes
        class_weights = torch.FloatTensor([1.0, 5.0])  # Regular=1, Critical=5
        loss_fn = torch.nn.NLLLoss(weight=class_weights)

        # Create more distinctive labels
        data.y = torch.zeros(data.num_nodes, dtype=torch.long)
        
        # Mark ALL exits and high-centrality nodes as important (Class 1)
        G_nx = to_networkx(data, to_undirected=True)
        
        # Calculate centrality (identifies junction nodes)
        centrality = nx.betweenness_centrality(G_nx)
        
        for idx, node in enumerate(G_nx.nodes()):
            # Mark exits as critical
            if hasattr(data, 'type') and idx < len(data.type) and data.type[idx] == 'Exit':
                data.y[idx] = 1  # Mark as critical
            # Mark nodes with 3+ connections as critical (junction nodes)
            elif G_nx.degree(node) >= 3:
                data.y[idx] = 1  # Mark as critical
            # Also consider high centrality nodes
            elif centrality[node] > 0.05:
                data.y[idx] = 1  # Also mark as critical
        
        # Count junction nodes (degree >= 3) for reporting
        junction_count = sum(1 for node in G_nx.nodes() if G_nx.degree(node) >= 3)
        print(f"Junction nodes (3+ connections): {junction_count}")
        
        # Print number of critical nodes identified
        print(f"Critical nodes identified: {torch.sum(data.y).item()} out of {data.num_nodes}")
        print(f"Starting training with {torch.sum(data.y).item()} critical nodes out of {data.num_nodes}")
        
        data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)

        # Track loss history
        loss_history = []

        # Training loop
        model.train()
        for epoch in range(300):  # Instead of 100
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Record loss
            loss_history.append(loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
                
        return model, untrained_model, loss_history
    except Exception as e:
        print(f"Error during training: {e}")
        # Return default values or raise a more informative error