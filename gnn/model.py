import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class FormGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FormGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x = self.lin(x)  # [batch_size, output_dim]
        return x

if __name__ == "__main__":
    print("FormGCN model for building form optimization loaded.")