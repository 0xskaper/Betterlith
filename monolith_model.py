import torch
import torch.nn as nn
from typing import List, Optional
from embedding_table import CuckooHashTable
from model_layer import DeepFMLayer


class MonolithModel:
    """Core Monolith model combining collisionless embedding table with DeepFM."""

    def __init__(self,
                 embedding_dim: int,
                 hidden_layers: List[int],
                 max_tries: int = 20,
                 feature_expiration: float = 86400,  # 24 hours in seconds
                 min_frequency: int = 10):
        self.embedding_tables = {}  # Multiple embedding tables for different feature types
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.max_tries = max_tries
        self.feature_expiration = feature_expiration
        self.min_frequency = min_frequency

        # Initialize DeepFM model
        self.model = None

    def create_embedding_table(self, table_name: str):
        """Create a new embedding table for a feature type."""
        self.embedding_tables[table_name] = CuckooHashTable(
            embedding_dim=self.embedding_dim,
            max_tries=self.max_tries
        )

    def update_embedding(self, table_name: str, feature_id: int, embedding: torch.Tensor):
        """Update an embedding in the specified table."""
        if table_name not in self.embedding_tables:
            self.create_embedding_table(table_name)
        self.embedding_tables[table_name].insert(feature_id, embedding)

    def get_embedding(self, table_name: str, feature_id: int) -> Optional[torch.Tensor]:
        """Get an embedding from the specified table."""
        if table_name not in self.embedding_tables:
            return None
        return self.embedding_tables[table_name].lookup(feature_id)

    def clean_expired_features(self):
        """Clean expired features from all embedding tables."""
        for table in self.embedding_tables.values():
            table.clean_expired_features(self.feature_expiration)

    def filter_infrequent_features(self):
        """Filter infrequent features from all embedding tables."""
        for table in self.embedding_tables.values():
            table.filter_by_frequency(self.min_frequency)

    def initialize_model(self, num_features: int):
        """Initialize the DeepFM model."""
        self.model = DeepFMLayer(
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers
        )

    def train_step(self,
                   feature_ids: torch.Tensor,
                   feature_values: torch.Tensor,
                   labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer):
        """Perform a single training step."""
        if self.model is None:
            self.initialize_model(feature_ids.shape[1])

        predictions = self.model(feature_ids, feature_values)
        loss = nn.BCELoss()(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self, feature_ids: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        """Make predictions for the given features."""
        if self.model is None:
            raise ValueError("Model not initialized. Call train_step first.")

        with torch.no_grad():
            return self.model(feature_ids, feature_values)
