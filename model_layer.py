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
        self.deep_layers = []
        input_dim = embedding_dim

        for hidden_dim in hidden_layers:
            self.deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                # Changed BatchNorm
                nn.BatchNorm1d(hidden_dim, track_running_stats=False),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        self.deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.ModuleList(self.deep_layers)

    def forward(self, feature_ids: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            feature_ids: (batch_size, num_input_features)
            feature_values: (batch_size, num_input_features)
        """
        # First order
        first_order_weights = self.first_order(feature_ids).squeeze(-1)
        first_order = torch.sum(first_order_weights * feature_values, dim=1)

        # Second order
        embeddings = self.second_order(feature_ids)
        embeddings = embeddings * feature_values.unsqueeze(-1)

        # Sum pooling
        pooled_embedding = torch.sum(embeddings, dim=1)

        # Second order term
        square_sum = torch.sum(embeddings, dim=1).pow(2)
        sum_square = torch.sum(embeddings.pow(2), dim=1)
        second_order = 0.5 * torch.sum(square_sum - sum_square, dim=1)

        # Deep component
        deep = pooled_embedding
        for layer in self.deep_layers:
            if isinstance(layer, nn.BatchNorm1d) and deep.size(0) == 1:
                # Skip BatchNorm for single sample prediction
                continue
            deep = layer(deep)
        deep = deep.squeeze(-1)

        return torch.sigmoid(first_order + second_order + deep)

    def predict_single(self, feature_ids: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        """Special method for single-item prediction."""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            return self.forward(feature_ids, feature_values)
