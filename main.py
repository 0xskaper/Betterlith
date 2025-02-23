import torch
import numpy as np
from monolith_model import MonolithModel
from typing import Dict, List, Tuple
import random


class RecommendationDataGenerator:
    def __init__(self, num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items

        # Define feature spaces
        self.num_gender_categories = 2  # M, F
        self.num_age_buckets = 10  # age divided into 10 buckets
        self.num_price_buckets = 10  # price divided into 10 buckets
        self.num_categories = 10  # 10 item categories

        # Calculate total feature space
        self.total_features = (
            num_users +  # user IDs
            num_items +  # item IDs
            self.num_gender_categories +  # gender
            self.num_age_buckets +  # age buckets
            self.num_price_buckets +  # price buckets
            self.num_categories  # item categories
        )

        # Generate synthetic user data
        self.user_features = {
            'age': np.random.randint(18, 80, num_users),
            'gender': np.random.choice(['M', 'F'], num_users),
        }

        # Generate synthetic item data
        self.item_features = {
            'price': np.random.uniform(10, 1000, num_items),
            'category': np.random.randint(0, self.num_categories, num_items)
        }

        # Calculate starting indices for each feature type
        self.feature_starts = {
            'user_id': 0,
            'item_id': num_users,
            'gender': num_users + num_items,
            'age': num_users + num_items + self.num_gender_categories,
            'price': num_users + num_items + self.num_gender_categories + self.num_age_buckets,
            'category': num_users + num_items + self.num_gender_categories + self.num_age_buckets + self.num_price_buckets
        }

    def _bucketize(self, value: float, min_val: float, max_val: float, num_buckets: int) -> int:
        """Convert a continuous value into a bucket index."""
        buckets = np.linspace(min_val, max_val, num_buckets + 1)
        return np.digitize(value, buckets[1:-1])

    def encode_features(self, user_id: int, item_id: int) -> Tuple[List[int], List[float]]:
        """Encode a user-item pair into feature IDs and values."""
        feature_ids = []
        feature_values = []

        # Add user ID
        feature_ids.append(self.feature_starts['user_id'] + user_id)
        feature_values.append(1.0)

        # Add item ID
        feature_ids.append(self.feature_starts['item_id'] + item_id)
        feature_values.append(1.0)

        # Add gender
        gender_idx = 0 if self.user_features['gender'][user_id] == 'M' else 1
        feature_ids.append(self.feature_starts['gender'] + gender_idx)
        feature_values.append(1.0)

        # Add age bucket
        age_bucket = self._bucketize(
            self.user_features['age'][user_id], 18, 80, self.num_age_buckets)
        feature_ids.append(self.feature_starts['age'] + age_bucket)
        feature_values.append(1.0)

        # Add price bucket
        price_bucket = self._bucketize(
            self.item_features['price'][item_id], 10, 1000, self.num_price_buckets)
        feature_ids.append(self.feature_starts['price'] + price_bucket)
        feature_values.append(1.0)

        # Add category
        feature_ids.append(
            self.feature_starts['category'] + self.item_features['category'][item_id])
        feature_values.append(1.0)

        # Verify all feature IDs are within bounds
        assert all(0 <= fid < self.total_features for fid in feature_ids), \
            f"Feature ID out of bounds. Max allowed: {
                self.total_features-1}, Got: {max(feature_ids)}"

        return feature_ids, feature_values

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of training data."""
        all_feature_ids = []
        all_feature_values = []
        all_labels = []

        for _ in range(batch_size):
            user_id = random.randint(0, self.num_users - 1)
            item_id = random.randint(0, self.num_items - 1)

            # Generate synthetic label (like/dislike)
            # Make it somewhat dependent on user-item features for realistic data
            user_age = self.user_features['age'][user_id]
            item_price = self.item_features['price'][item_id]

            # Simple logic: younger users prefer cheaper items
            prob_like = 1 - abs((user_age/80.0) - (item_price/1000.0))
            label = float(random.random() < prob_like)

            feature_ids, feature_values = self.encode_features(
                user_id, item_id)

            all_feature_ids.append(feature_ids)
            all_feature_values.append(feature_values)
            all_labels.append(label)

        return (torch.tensor(all_feature_ids),
                torch.tensor(all_feature_values),
                torch.tensor(all_labels))


def train_model(model: MonolithModel, data_generator: RecommendationDataGenerator,
                num_epochs: int, batch_size: int):
    """Train the model."""
    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 100  # Train on 100 batches per epoch

        for batch in range(num_batches):
            feature_ids, feature_values, labels = data_generator.generate_batch(
                batch_size)
            loss = model.train_step(
                feature_ids, feature_values, labels, optimizer)
            total_loss += loss

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def main():
    # Model parameters
    num_users = 1000
    num_items = 500
    embedding_dim = 16
    hidden_layers = [64, 32]

    print("Initializing data generator...")
    data_generator = RecommendationDataGenerator(num_users, num_items)

    print("Creating model...")
    model = MonolithModel(
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers
    )

    # Initialize model with correct number of features
    print(f"Total feature space: {data_generator.total_features}")
    model.initialize_model(data_generator.total_features)

    # Train model
    print("\nStarting training...")
    train_model(model, data_generator, num_epochs=5, batch_size=32)

    # Generate recommendations for a sample user
    print("\nGenerating sample recommendations...")
    sample_user_id = random.randint(0, num_users - 1)
    print(f"\nRecommendations for user {sample_user_id}:")
    print(f"Age: {data_generator.user_features['age'][sample_user_id]}")
    print(f"Gender: {data_generator.user_features['gender'][sample_user_id]}")

    # Get predictions for some items
    test_items = random.sample(range(num_items), 10)
    predictions = []

    # Create a batch of predictions instead of one at a time
    batch_feature_ids = []
    batch_feature_values = []

    for item_id in test_items:
        feature_ids, feature_values = data_generator.encode_features(
            sample_user_id, item_id)
        batch_feature_ids.append(feature_ids)
        batch_feature_values.append(feature_values)

    # Convert to tensors
    batch_feature_ids = torch.tensor(batch_feature_ids)
    batch_feature_values = torch.tensor(batch_feature_values)

    # Get predictions
    with torch.no_grad():
        batch_predictions = model.predict(
            batch_feature_ids, batch_feature_values)

    # Store predictions
    for item_id, pred in zip(test_items, batch_predictions):
        predictions.append((item_id, pred.item()))

    # Sort by prediction score
    predictions.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 recommended items:")
    print("Item ID | Category | Price  | Score")
    print("-" * 40)
    for item_id, score in predictions[:5]:
        print(f"{item_id:7d} | {data_generator.item_features['category'][item_id]:8d} | "
              f"${data_generator.item_features['price'][item_id]:6.2f} | {score:.4f}")


if __name__ == "__main__":
    main()
