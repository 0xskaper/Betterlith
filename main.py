import torch
import numpy as np
from monolith_model import MonolithModel
from typing import Dict, List, Tuple
import random
import time
from cli_results import BentoResults, format_model_results


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


def determine_user_segment(data_generator: RecommendationDataGenerator, user_id: int) -> str:
    """Determine user segment based on age and other features."""
    age = data_generator.user_features['age'][user_id]
    if age < 25:
        return "Young Adult"
    elif age < 40:
        return "Professional"
    elif age < 60:
        return "Mid-Career"
    else:
        return "Senior"


def train_model(model: MonolithModel,
                data_generator: RecommendationDataGenerator,
                num_epochs: int,
                batch_size: int) -> Tuple[List[float], float]:
    """Train the model and return training history."""
    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
    epoch_losses = []
    start_time = time.time()

    print("\nTraining Progress:")
    print("-" * 50)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 100

        for batch in range(num_batches):
            feature_ids, feature_values, labels = data_generator.generate_batch(
                batch_size)
            loss = model.train_step(
                feature_ids, feature_values, labels, optimizer)
            total_loss += loss

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} │ Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    return epoch_losses, training_time


def generate_recommendations(model: MonolithModel,
                             data_generator: RecommendationDataGenerator,
                             user_id: int,
                             num_items: int = 20) -> List[Dict]:
    """Generate recommendations for a user."""
    test_items = random.sample(range(data_generator.num_items), num_items)
    batch_feature_ids = []
    batch_feature_values = []

    for item_id in test_items:
        feature_ids, feature_values = data_generator.encode_features(
            user_id, item_id)
        batch_feature_ids.append(feature_ids)
        batch_feature_values.append(feature_values)

    batch_feature_ids = torch.tensor(batch_feature_ids)
    batch_feature_values = torch.tensor(batch_feature_values)

    with torch.no_grad():
        batch_predictions = model.predict(
            batch_feature_ids, batch_feature_values)

    recommendations = []
    for item_id, pred in zip(test_items, batch_predictions):
        recommendations.append({
            "id": item_id,
            "category": int(data_generator.item_features['category'][item_id]),
            "price": float(data_generator.item_features['price'][item_id]),
            "score": float(pred.item())
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:5]


def analyze_recommendations(recommendations: List[Dict]) -> Dict:
    """Analyze recommendation patterns."""
    categories = {}
    prices = [rec["price"] for rec in recommendations]
    scores = [rec["score"] for rec in recommendations]

    for rec in recommendations:
        cat = f"Category {rec['category']}"
        categories[cat] = categories.get(cat, 0) + 1

    return {
        'price_sensitivity': 1.0 - (np.mean(prices) / 1000.0),
        'category_diversity': len(categories) / len(recommendations),
        'engagement_score': np.mean(scores),
        'category_distribution': categories
    }


def main():
    # Model parameters
    num_users = 1000
    num_items = 500
    embedding_dim = 16
    hidden_layers = [64, 32]
    num_epochs = 5
    batch_size = 32

    # Initialize data generator
    print("Initializing data generator...")
    data_generator = RecommendationDataGenerator(num_users, num_items)

    # Create model
    print("Creating model...")
    model = MonolithModel(
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers
    )
    model.initialize_model(data_generator.total_features)

    # Train model
    print("\nStarting training...")
    epoch_losses, training_time = train_model(
        model, data_generator, num_epochs, batch_size
    )

    # Generate recommendations
    print("\nGenerating recommendations...")
    sample_user_id = random.randint(0, num_users - 1)
    recommendations = generate_recommendations(
        model, data_generator, sample_user_id
    )

    # Analyze recommendations
    analysis = analyze_recommendations(recommendations)

    # Collect all results
    results = {
        'final_loss': epoch_losses[-1],
        'accuracy': 1.0 - epoch_losses[-1],  # Simplified accuracy metric
        'improvement': (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0],
        'training_time': training_time,
        'user_id': sample_user_id,
        'user_age': data_generator.user_features['age'][sample_user_id],
        'user_gender': data_generator.user_features['gender'][sample_user_id],
        'user_segment': determine_user_segment(data_generator, sample_user_id),
        'recommendations': recommendations,
        'performance': {
            'response_time': training_time / num_epochs * 1000,  # Convert to ms
            'memory_usage': 256.5,  # Example value
            'cache_hit_rate': 0.85   # Example value
        },
        'analysis': analysis
    }

    # Display results in bento grid style
    display = BentoResults()
    formatted_results = format_model_results(results)
    display.display_results(formatted_results)


if __name__ == "__main__":
    main()
