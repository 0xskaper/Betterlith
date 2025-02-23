import torch
import torch.nn as nn
from typing import List


class DeepFMLayer(nn.Module):
    """DeepFM model layer combining FM and deep components."""

    def __init__(self, num_features: int, embedding_dim: int, hidden_layers: List[int]):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # FM first-order term
        self.first_order = nn.Embedding(num_features, 1)

        # FM second-order term
        self.second_order = nn.Embedding(num_features, embedding_dim)

        # Deep component
        deep_layers = []
        input_dim = num_features * embedding_dim
        for hidden_dim in hidden_layers:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.Sequential(*deep_layers)

    def forward(self, feature_ids: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        # First order
        first_order_weights = self.first_order(feature_ids).squeeze()
        first_order = torch.sum(first_order_weights * feature_values, dim=1)

        # Second order
        # batch_size x num_features x embedding_dim
        embeddings = self.second_order(feature_ids)
        square_sum = torch.sum(embeddings, dim=1).pow(2)
        sum_square = torch.sum(embeddings.pow(2), dim=1)
        second_order = 0.5 * torch.sum(square_sum - sum_square, dim=1)

        # Deep component
        deep_input = embeddings.reshape(-1,
                                        self.num_features * self.embedding_dim)
        deep = self.deep_layers(deep_input).squeeze()

        return torch.sigmoid(first_order + second_order + deep)
