import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adjacency_matrix, features):
        support = features @ self.weight
        output = adjacency_matrix @ support
        return output

class GNNModel(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConvLayer(input_features, hidden_features)
        self.conv2 = GraphConvLayer(hidden_features, output_features)

    def forward(self, adjacency_matrix, features):
        x = F.relu(self.conv1(adjacency_matrix, features))
        x = self.conv2(adjacency_matrix, x)
        return x
