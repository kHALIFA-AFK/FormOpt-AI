import torch
from gnn.model import FormGCN
import os
from torch_geometric.loader import DataLoader

def train_model(loader, input_dim, target_dim=1):
    try:
        model = FormGCN(input_dim=input_dim, hidden_dim=16, output_dim=target_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        loss_fn = torch.nn.MSELoss()  # For regression

        loss_history = []

        model.train()
        for epoch in range(300):
            total_loss = 0
            for data in loader:
                optimizer.zero_grad()
                out = model(data)
                if out.dim() > 1:
                    out = out.squeeze()
                loss = loss_fn(out, data.graph_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
        return model, loss_history

    except Exception as e:
        print(f"Error during training: {e}")
        return None, []

