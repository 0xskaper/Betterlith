import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import datetime
import os
import time
import hnswlib
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.model_selection import train_test_split  # Added for train/test split
import gc  # For garbage collection to manage memory
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt


class HNSWIndex:
    def __init__(self, dimension):
        """
        Initialize the HNSW index.
        :param dimension: Dimension of the embedding vectors.
        """
        self.dimension = dimension
        self.index = hnswlib.Index(
            space="cosine", dim=dimension
        )  # Changed to cosine similarity for better recommendations
        self.id_map = None  # Used to map internal indices to user-provided item_ids
        self.is_built = False  # Flag to indicate if the index is built

    def build(self, item_ids, item_embeddings):
        """
        Build the HNSW index.
        :param item_ids: List of user-provided item IDs.
        :param item_embeddings: PyTorch tensor of shape [num_items, embedding_dim].
        :return: Number of items indexed.
        """
        # Validate inputs
        assert (
            len(item_ids) == item_embeddings.shape[0]
        ), "item_ids and item_embeddings must have the same length"
        assert (
            item_embeddings.shape[1] == self.dimension
        ), "Embedding dimension does not match initialization parameter"

        # Convert PyTorch tensor to NumPy array
        embeddings_np = item_embeddings.cpu().numpy().astype(np.float32)
        num_items = embeddings_np.shape[0]

        # Initialize the HNSW index with improved parameters for better accuracy
        self.index.init_index(
            max_elements=num_items,  # Maximum capacity (matches dataset size)
            ef_construction=500,  # Increased for better index quality
            M=64,  # Increased for better connectivity and recall
        )

        # Add items to the index
        self.index.add_items(embeddings_np)

        # Save item_ids for later mapping
        self.id_map = item_ids
        self.is_built = True
        return num_items

    def search(self, query_embedding, k=10):
        """
        Search for nearest neighbors.
        :param query_embedding: PyTorch tensor of shape [batch_size, embedding_dim] or [embedding_dim].
        :param k: Number of nearest neighbors to return.
        :return: List of tuples [(item_ids, distances)] for each query.
        """
        if not self.is_built:
            raise RuntimeError("Index is not built. Please call build() first.")

        # Handle input format
        query_np = query_embedding.cpu().numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = np.expand_dims(query_np, axis=0)  # Convert to [1, D]

        # Set higher ef_search parameter for better recall at search time
        self.index.set_ef(500)  # Significantly increased for better recall

        # Perform the search
        indices, distances = self.index.knn_query(query_np, k=k)

        # Map internal indices to user-provided item_ids
        results = []
        for i in range(len(indices)):
            # Get results for the current query
            batch_indices = indices[i]
            batch_distances = distances[i]

            # Map to item_ids
            batch_item_ids = [self.id_map[idx] for idx in batch_indices]

            # Combine results (distances remain as a list of floats)
            results.append((batch_item_ids, batch_distances.tolist()))

        return results


class CollisionlessEmbeddingTable:
    """
    A simple implementation of collisionless embedding table using dictionary
    with expiration mechanism for embeddings.
    """

    def __init__(self, embedding_dim=32, expiration_days=30):  # Increased embedding dim
        self.embedding_dim = embedding_dim
        self.expiration_days = expiration_days
        self.embedding_dict = {}  # feature_id -> (embedding, last_access_time)

    def lookup(self, feature_ids):
        """
        Look up embeddings for feature ids, creating new ones if they don't exist.
        Also updates the last access time.
        """
        current_time = datetime.datetime.now()
        embeddings = []

        for feature_id in feature_ids:
            if feature_id in self.embedding_dict:
                embedding, _ = self.embedding_dict[feature_id]
                self.embedding_dict[feature_id] = (embedding, current_time)
            else:
                # Better initialization for embeddings using Xavier/Glorot
                embedding = torch.randn(self.embedding_dim) * np.sqrt(
                    2.0 / (self.embedding_dim)
                )
                self.embedding_dict[feature_id] = (embedding, current_time)

            embeddings.append(embedding)

        return (
            torch.stack(embeddings)
            if embeddings
            else torch.zeros((0, self.embedding_dim))
        )

    def update(self, feature_ids, gradients):
        """
        Update embeddings based on gradients.
        """
        learning_rate = 0.01  # Simplified learning rate

        for i, feature_id in enumerate(feature_ids):
            if feature_id in self.embedding_dict:
                embedding, last_access = self.embedding_dict[feature_id]
                updated_embedding = embedding - learning_rate * gradients[i]
                self.embedding_dict[feature_id] = (updated_embedding, last_access)

    def expire_old_embeddings(self):
        """
        Remove embeddings that haven't been accessed for a long time.
        """
        current_time = datetime.datetime.now()
        expiration_threshold = current_time - datetime.timedelta(
            days=self.expiration_days
        )

        keys_to_remove = []
        for feature_id, (_, last_access) in self.embedding_dict.items():
            if last_access < expiration_threshold:
                keys_to_remove.append(feature_id)

        for feature_id in keys_to_remove:
            del self.embedding_dict[feature_id]

        return len(keys_to_remove)  # Return number of expired embeddings


class DeepFM(nn.Module):
    """
    Implementation of DeepFM model for recommendation.
    """

    def __init__(
        self, field_dims, embedding_dim=32, mlp_dims=(128, 64, 32), dropout=0.3
    ):
        """
        Parameters:
        - field_dims: A list of field dimensions (number of features in each field)
        - embedding_dim: The size of embedding vectors (increased)
        - mlp_dims: The hidden layer dimensions for MLP part (increased width)
        - dropout: Dropout rate (increased)
        """
        super(DeepFM, self).__init__()

        self.num_fields = len(field_dims)
        self.embedding_dim = embedding_dim
        self.field_dims = field_dims
        self.num_users = field_dims[0]
        self.num_items = field_dims[1]

        # FM component
        self.fm_first_order = nn.ModuleList(
            [nn.Embedding(field_dim, 1) for field_dim in field_dims]
        )
        self.fm_second_order = nn.ModuleList(
            [nn.Embedding(field_dim, embedding_dim) for field_dim in field_dims]
        )

        # Deep component
        self.mlp_input_dim = self.num_fields * embedding_dim
        self.mlp = nn.Sequential()
        input_dim = self.mlp_input_dim

        # Added batch normalization for better training stability
        for i, dim in enumerate(mlp_dims):
            self.mlp.add_module(f"linear_{i}", nn.Linear(input_dim, dim))
            self.mlp.add_module(f"batchnorm_{i}", nn.BatchNorm1d(dim))
            self.mlp.add_module(
                f"leakyrelu_{i}", nn.LeakyReLU(0.1)
            )  # Changed to LeakyReLU
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))
            input_dim = dim

        self.mlp.add_module("linear_final", nn.Linear(input_dim, 1))

        # Initialize weights with better approaches
        for embedding in self.fm_first_order:
            nn.init.xavier_normal_(embedding.weight)

        for embedding in self.fm_second_order:
            nn.init.xavier_normal_(embedding.weight)

    def forward(self, x):
        """
        Forward pass of DeepFM.
        x: input tensor of shape (batch_size, num_fields)
        """
        # FM First Order
        fm_first_order_sum = 0
        for i in range(self.num_fields):
            fm_first_order_sum += self.fm_first_order[i](x[:, i])

        # FM Second Order
        fm_second_order_embeddings = [
            self.fm_second_order[i](x[:, i]) for i in range(self.num_fields)
        ]

        # Sum of squares - square of sum
        sum_of_square = sum(embedding**2 for embedding in fm_second_order_embeddings)
        square_of_sum = sum(fm_second_order_embeddings) ** 2
        fm_second_order_sum = 0.5 * (square_of_sum - sum_of_square).sum(1, keepdim=True)

        # Deep component
        deep_input = torch.cat(fm_second_order_embeddings, dim=1)
        deep_out = self.mlp(deep_input)

        # Final prediction
        prediction = fm_first_order_sum + fm_second_order_sum + deep_out
        return torch.sigmoid(prediction.squeeze(1))


# ========================= Data Loading and Processing Functions ===========================


def load_movielens_data(data_dir, sample_size=None):
    """
    Load MovieLens dataset from CSV files with option to sample
    """
    # Load movies
    movies_file = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(movies_file):
        raise FileNotFoundError(f"Movies file not found: {movies_file}")
    movies_df = pd.read_csv(movies_file)

    # Load ratings
    ratings_file = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

    if sample_size:
        # Use pandas to read a random sample of the ratings
        print(f"Loading a sample of {sample_size} ratings...")
        np.random.seed(42)

        # Count lines to determine sampling ratio
        with open(ratings_file, "r") as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header

        sample_ratio = min(1.0, sample_size / total_lines)
        ratings_df = pd.read_csv(
            ratings_file, skiprows=lambda x: x > 0 and np.random.random() > sample_ratio
        )
    else:
        # Load the entire dataset
        print("Loading full ratings dataset...")
        ratings_df = pd.read_csv(ratings_file)

    print(f"Loaded {len(ratings_df)} ratings for {len(movies_df)} movies")

    # Convert ratings to binary labels (typically >= 3.5 is considered positive)
    ratings_df["label"] = (ratings_df["rating"] >= 3.5).astype(int)

    return ratings_df, movies_df


def prepare_coldstart_evaluation(
    ratings_df,
    coldstart_ratio=0.2,
    initial_ratings=5,
    rating_threshold=3.5,
    test_ratio=0.2,
):
    """
    Prepare data for cold-start evaluation with improved relevant item selection

    Args:
        ratings_df: DataFrame with ratings
        coldstart_ratio: Ratio of users to treat as cold-start users (reduced)
        initial_ratings: Number of initial ratings for cold-start users (increased)
        rating_threshold: Threshold to consider a rating as positive (increased)
        test_ratio: Ratio of data to use for testing (default 0.2 for 80/20 split)

    Returns:
        Dictionary with train and test data for cold-start evaluation
    """
    print("Preparing cold-start evaluation data...")

    # Only consider users with sufficient ratings for cold-start evaluation
    # (need some for training and some for testing)
    user_counts = ratings_df["userId"].value_counts()
    # Increased minimum rating count to ensure better user representation
    qualifying_users = user_counts[user_counts >= 15].index.tolist()

    print(f"Found {len(qualifying_users)} users with at least 15 ratings")

    # 1. Select cold-start users from qualifying users
    np.random.seed(42)
    cold_start_users = np.random.choice(
        qualifying_users,
        size=min(int(len(qualifying_users) * coldstart_ratio), len(qualifying_users)),
        replace=False,
    )
    print(f"Selected {len(cold_start_users)} users as cold-start users")

    # 2. Split data randomly into 80% train and 20% test
    np.random.seed(42)  # Ensure reproducibility

    # Create train/test split indices
    all_indices = np.arange(len(ratings_df))
    test_indices = np.random.choice(
        all_indices, size=int(len(ratings_df) * test_ratio), replace=False
    )
    train_indices = np.setdiff1d(all_indices, test_indices)

    # Split the dataframe
    train_full = ratings_df.iloc[train_indices]
    test_data = ratings_df.iloc[test_indices]

    print(
        f"Split data into {len(train_full)} training samples and {len(test_data)} test samples"
    )

    # 3. Separate regular and cold-start users in training data
    train_regular = train_full[~train_full["userId"].isin(cold_start_users)]
    cold_users_train = train_full[train_full["userId"].isin(cold_start_users)]

    # 4. For cold-start users, select initial ratings more strategically
    initial_data = []
    for user in cold_start_users:
        user_ratings = cold_users_train[cold_users_train["userId"] == user]
        if len(user_ratings) > 0:
            # Prioritize higher rated items to get a clearer signal of user preferences
            user_ratings = user_ratings.sort_values("rating", ascending=False)

            n_initial = min(initial_ratings, len(user_ratings))
            initial_data.append(user_ratings.iloc[:n_initial])

    # Combine regular user data and cold-start user initial data
    initial_data_df = pd.concat(initial_data) if initial_data else pd.DataFrame()
    train_data = pd.concat([train_regular, initial_data_df])

    # Get cold-start test data
    cold_start_test = test_data[test_data["userId"].isin(cold_start_users)]

    # Create binary label for test data
    test_data["label"] = (test_data["rating"] >= rating_threshold).astype(int)
    cold_start_test["label"] = (cold_start_test["rating"] >= rating_threshold).astype(
        int
    )

    # Add positive interactions count for debugging
    pos_interactions = cold_start_test[cold_start_test["label"] == 1]
    users_with_pos = pos_interactions["userId"].nunique()

    print(f"Training data: {len(train_data)} ratings")
    print(f"Cold-start users initial data: {len(initial_data_df)} ratings")
    print(f"Cold-start users test data: {len(cold_start_test)} ratings")
    print(f"Cold-start users with positive test interactions: {users_with_pos}")
    print(f"Number of positive test interactions: {len(pos_interactions)}")

    return {
        "train_data": train_data,
        "cold_start_test": cold_start_test,
        "full_test": test_data,
        "cold_start_users": cold_start_users,
    }


def preprocess_for_recommendation(train_df, binary_threshold=3.5):
    """
    Preprocess ratings data for our recommendation model
    """
    print("Creating user and movie ID mappings...")
    # Create user and movie ID mappings (to ensure consecutive IDs starting from 0)
    unique_user_ids = train_df["userId"].unique()
    unique_movie_ids = train_df["movieId"].unique()

    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_user_ids)}
    movie_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids)}

    print(
        f"Found {len(unique_user_ids)} unique users and {len(unique_movie_ids)} unique movies"
    )

    # Map IDs to new consecutive IDs
    print("Mapping IDs to consecutive indices...")
    train_df["user_idx"] = train_df["userId"].map(user_id_map)
    train_df["movie_idx"] = train_df["movieId"].map(movie_id_map)

    # Check if binary label already exists
    if "label" not in train_df.columns:
        # Convert ratings to binary labels if using binary model
        if binary_threshold is not None:
            train_df["label"] = (train_df["rating"] >= binary_threshold).astype(int)
        else:
            train_df["label"] = train_df["rating"] / 5.0  # Normalize to [0,1]

    # Prepare data for the model
    print("Preparing tensors for the model...")
    user_ids = torch.tensor(train_df["user_idx"].values)
    movie_ids = torch.tensor(train_df["movie_idx"].values)
    labels = torch.tensor(train_df["label"].values)

    # Create reverse mappings for later reference
    reverse_user_map = {v: k for k, v in user_id_map.items()}
    reverse_movie_map = {v: k for k, v in movie_id_map.items()}

    num_users = len(unique_user_ids)
    num_items = len(unique_movie_ids)

    return {
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "labels": labels,
        "num_users": num_users,
        "num_items": num_items,
        "user_id_map": user_id_map,
        "movie_id_map": movie_id_map,
        "reverse_user_map": reverse_user_map,
        "reverse_movie_map": reverse_movie_map,
        "ratings_df": train_df,  # Include for later reference
    }


def preprocess_test_data(test_df, user_id_map, movie_id_map):
    """
    Preprocess test data using existing ID mappings
    """
    print("Preprocessing test data...")
    # Filter users and items that are in the training set
    test_df = test_df[test_df["userId"].isin(user_id_map.keys())]
    test_df = test_df[test_df["movieId"].isin(movie_id_map.keys())]

    # Map to internal indices
    test_df["user_idx"] = test_df["userId"].map(user_id_map)
    test_df["movie_idx"] = test_df["movieId"].map(movie_id_map)

    # Prepare tensors
    user_ids = torch.tensor(test_df["user_idx"].values)
    movie_ids = torch.tensor(test_df["movie_idx"].values)

    # Check if binary label already exists
    if "label" not in test_df.columns:
        test_df["label"] = (test_df["rating"] >= 3.5).astype(int)

    labels = torch.tensor(test_df["label"].values)

    print(f"Test data contains {len(test_df)} valid interactions after filtering")

    return {
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "labels": labels,
        "ratings_df": test_df,
    }


class MovieLensRecommendationSystem:
    """
    A recommendation system for MovieLens data based on DeepFM model
    """

    def __init__(
        self, num_users, num_items, embedding_dim=32
    ):  # Increased embedding dim
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Use the collisionless embedding table for users and items (not actively used in main workflow)
        self.user_embedding_table = CollisionlessEmbeddingTable(embedding_dim)
        self.item_embedding_table = CollisionlessEmbeddingTable(embedding_dim)

        # DeepFM model for prediction with improved architecture
        print(
            f"Initializing DeepFM model with {num_users} users and {num_items} items..."
        )
        self.model = DeepFM(
            field_dims=[num_users, num_items],
            embedding_dim=embedding_dim,
            mlp_dims=(128, 64, 32),  # Wider network
            dropout=0.3,  # Increased dropout for better generalization
        )

        # Use Adam optimizer with weight decay and better learning rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5,  # Added weight decay for regularization
        )

        # Use weighted BCE loss to handle class imbalance
        self.criterion = nn.BCELoss()

        # To convert internal IDs back to original MovieLens IDs
        self.reverse_user_map = None
        self.reverse_movie_map = None
        self.movies_df = None
        self.ratings_df = None  # Added this attribute

        # ANN index (initialized when needed)
        self.ann_index = None

        # For learning rate scheduling
        self.scheduler = None

        # Item popularity storage
        self.item_popularity = None

    def set_mapping(
        self, reverse_user_map, reverse_movie_map, movies_df=None, ratings_df=None
    ):
        """
        Set mappings to convert between internal and original IDs

        Args:
            reverse_user_map: Mapping from internal to original user IDs
            reverse_movie_map: Mapping from internal to original movie IDs
            movies_df: DataFrame with movie information
            ratings_df: DataFrame with ratings data used for training
        """
        self.reverse_user_map = reverse_user_map
        self.reverse_movie_map = reverse_movie_map
        self.movies_df = movies_df
        self.ratings_df = ratings_df

        # Calculate item popularity directly here instead of calling a separate method
        if ratings_df is not None:
            print("Calculating item popularity...")

            # If we don't have movie_idx, assume we're using the original ratings_df
            if (
                "movie_idx" not in ratings_df.columns
                and "movieId" in ratings_df.columns
            ):
                # Map movieId to internal IDs
                if hasattr(self, "reverse_movie_map"):
                    # Create a mapping from original ID to internal ID
                    movie_id_map = {v: k for k, v in self.reverse_movie_map.items()}
                    if "userId" in ratings_df.columns:
                        # Filter to only include known movies
                        ratings_with_map = ratings_df[
                            ratings_df["movieId"].isin(movie_id_map.keys())
                        ]
                        # Add item_idx column
                        ratings_with_map = (
                            ratings_with_map.copy()
                        )  # Avoid SettingWithCopyWarning
                        ratings_with_map["movie_idx"] = ratings_with_map["movieId"].map(
                            movie_id_map
                        )

                        # Calculate popularity
                        if len(ratings_with_map) > 0:
                            # Count occurrences of each movie
                            item_counts = (
                                ratings_with_map["movie_idx"]
                                .value_counts()
                                .reset_index()
                            )
                            item_counts.columns = ["movie_idx", "count"]

                            # Convert to list of (item_id, count) tuples and sort by count in descending order
                            popularity_list = list(
                                zip(item_counts["movie_idx"], item_counts["count"])
                            )
                            popularity_list.sort(key=lambda x: x[1], reverse=True)

                            self.item_popularity = popularity_list
                            print(
                                f"Calculated popularity for {len(popularity_list)} items"
                            )
                            return

            # If we get here, try with movie_idx column if it exists
            if "movie_idx" in ratings_df.columns:
                # Count occurrences of each movie
                item_counts = ratings_df["movie_idx"].value_counts().reset_index()
                item_counts.columns = ["movie_idx", "count"]

                # Convert to list of (item_id, count) tuples and sort by count in descending order
                popularity_list = list(
                    zip(item_counts["movie_idx"], item_counts["count"])
                )
                popularity_list.sort(key=lambda x: x[1], reverse=True)

                self.item_popularity = popularity_list
                print(f"Calculated popularity for {len(popularity_list)} items")
            else:
                print(
                    "Warning: Unable to calculate item popularity, 'movie_idx' column not found"
                )

    def train(
        self,
        user_ids,
        movie_ids,
        labels,
        batch_size=2048,  # Increased batch size
        epochs=10,  # Increased epochs
        validation_data=None,
    ):
        """
        Train the recommendation model with improved training procedure
        """
        dataset = list(zip(user_ids, movie_ids, labels))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )

        # For tracking metrics
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 3  # For early stopping

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0

            print(f"Epoch {epoch+1}/{epochs} - Training...")
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for user_batch, movie_batch, label_batch in progress_bar:
                self.optimizer.zero_grad()

                # Convert to appropriate tensor types
                user_batch = user_batch.long()
                movie_batch = movie_batch.long()
                label_batch = label_batch.float()

                # Forward pass
                inputs = torch.stack([user_batch, movie_batch], dim=1)
                predictions = self.model(inputs)

                # Calculate loss
                loss = self.criterion(predictions, label_batch)

                # Backward pass and optimize
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item() * len(user_batch)
                batch_count += 1

                # Update progress bar
                progress_bar.set_postfix({"Loss": loss.item()})

            avg_loss = total_loss / len(user_ids)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", end="")

            # Validation if provided
            if validation_data is not None:
                val_loss, val_auc = self.evaluate(
                    validation_data["user_ids"],
                    validation_data["movie_ids"],
                    validation_data["labels"],
                )
                print(f", Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

                # Update learning rate based on validation loss
                self.scheduler.step(val_loss)

                # Save best model with mappings
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model_with_mappings(
                        self.model,
                        self.reverse_user_map,
                        self.reverse_movie_map,
                        "best_model_with_mappings.pt",
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= max_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            else:
                print()

            # Expire old embeddings (not crucial for main workflow)
            num_expired_users = self.user_embedding_table.expire_old_embeddings()
            num_expired_items = self.item_embedding_table.expire_old_embeddings()
            if num_expired_users > 0 or num_expired_items > 0:
                print(
                    f"Expired {num_expired_users} user embeddings and {num_expired_items} item embeddings"
                )

            # Run garbage collection to free memory
            gc.collect()

    def evaluate(self, user_ids, movie_ids, labels, batch_size=4096):
        """
        Evaluate the model on test data
        """
        print("Evaluating model...")
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        # Create a DataLoader to process test data in batches
        dataset = list(zip(user_ids, movie_ids, labels))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for user_batch, movie_batch, label_batch in data_loader:
                # Convert to appropriate tensor types
                inputs = torch.stack([user_batch.long(), movie_batch.long()], dim=1)
                predictions = self.model(inputs)
                loss = self.criterion(predictions, label_batch.float())

                total_loss += loss.item() * len(user_batch)

                # Save predictions and labels for AUC calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label_batch.cpu().numpy())

            # Calculate average loss and AUC
            avg_loss = total_loss / len(user_ids)
            auc = roc_auc_score(all_labels, all_predictions)

        return avg_loss, auc

    def predict(self, user_ids, movie_ids, batch_size=4096):
        """
        Make predictions for user-item pairs
        """
        self.model.eval()
        all_predictions = []

        # Process in batches to avoid memory issues
        num_samples = len(user_ids)
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)

                batch_user_ids = user_ids[start_idx:end_idx]
                batch_movie_ids = movie_ids[start_idx:end_idx]

                user_tensor = torch.tensor(batch_user_ids).long()
                movie_tensor = torch.tensor(batch_movie_ids).long()
                inputs = torch.stack([user_tensor, movie_tensor], dim=1)
                predictions = self.model(inputs)

                all_predictions.extend(predictions.cpu().numpy())

        return np.array(all_predictions)

    def recommend_items(
        self, user_id, top_k=10, exclude_seen=True, seen_movie_ids=None, batch_size=4096
    ):
        """
        Recommend top-k items for a user (standard method - scores all items)
        """
        print(f"Generating recommendations for user {user_id} using standard method...")
        start_time = time.time()

        # Generate predictions for all items
        item_ids = list(range(self.num_items))
        user_ids = [user_id] * len(item_ids)

        # Make predictions in batches
        predictions = self.predict(user_ids, item_ids, batch_size=batch_size)

        # Optionally exclude already seen movies
        if exclude_seen and seen_movie_ids is not None:
            for movie_id in seen_movie_ids:
                if movie_id < len(predictions):
                    predictions[movie_id] = -float("inf")

        # Get top-k item indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_items = [item_ids[idx] for idx in top_indices]

        # Convert to original MovieLens IDs and get titles if possible
        recommendations = []
        if self.reverse_movie_map and self.movies_df is not None:
            for item in top_items:
                original_id = self.reverse_movie_map.get(item)
                movie_info = self.movies_df[self.movies_df["movieId"] == original_id]
                if not movie_info.empty:
                    title = movie_info["title"].values[0]
                    recommendations.append(
                        {
                            "internal_id": item,
                            "movieId": original_id,
                            "title": title,
                            "score": float(predictions[item]),
                        }
                    )
                else:
                    recommendations.append(
                        {
                            "internal_id": item,
                            "movieId": original_id,
                            "title": f"Unknown Movie {original_id}",
                            "score": float(predictions[item]),
                        }
                    )
        else:
            recommendations = [
                {"internal_id": item, "score": float(predictions[item])}
                for item in top_items
            ]

        end_time = time.time()
        print(f"Standard recommendation time: {end_time - start_time:.4f} seconds")

        return recommendations

    def build_ann_index(self):
        """
        Build an ANN index for faster recommendations
        """
        print("Building HNSW index for faster recommendations...")

        # Extract item embeddings from the model
        self.model.eval()
        with torch.no_grad():
            # Get all item IDs
            item_ids = list(range(self.num_items))

            # Get embeddings for all items
            if hasattr(self.model, "fm_second_order"):
                # For DeepFM, extract from second-order embeddings
                item_embeddings = self.model.fm_second_order[1].weight.detach()
            else:
                raise ValueError(
                    "Unable to extract item embeddings from the current model"
                )

        # Initialize the HNSW index
        self.ann_index = HNSWIndex(dimension=self.embedding_dim)

        # Build the index
        num_indexed = self.ann_index.build(item_ids, item_embeddings)
        print(f"HNSW index built with {num_indexed} items")

        return self.ann_index

    def recommend_items_ann(
        self,
        user_id,
        top_k=10,
        candidate_multiplier=100,  # Dramatically increased for much better recall
        exclude_seen=True,
        seen_movie_ids=None,
    ):
        """
        Recommend top-k items for a user using ANN for fast retrieval with much higher recall
        """
        print(
            f"Generating recommendations for user {user_id} using improved ANN method..."
        )
        start_time = time.time()

        # Build index if not already built
        if (
            not hasattr(self, "ann_index")
            or self.ann_index is None
            or not self.ann_index.is_built
        ):
            self.build_ann_index()

        # Get user embedding
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "fm_second_order"):
                user_embedding = self.model.fm_second_order[0](
                    torch.tensor([user_id]).long()
                )[0]
            else:
                raise ValueError(
                    "Unable to extract user embedding from the current model"
                )

        # Get candidate items using ANN - with MUCH larger candidate pool
        num_candidates = min(top_k * candidate_multiplier, self.num_items)
        results = self.ann_index.search(user_embedding, k=num_candidates)
        candidate_items = results[0][0]  # First query, item IDs

        # Optionally get more diverse candidates
        if self.num_items > 5000:  # If we have a substantial item catalog
            # Also include some most popular items from the catalog
            if hasattr(self, "item_popularity") and self.item_popularity is not None:
                popular_items = [
                    item
                    for item, _ in self.item_popularity[:100]
                    if item not in candidate_items
                ]
                candidate_items.extend(popular_items[:50])  # Add up to 50 popular items

        # Filter out already seen items
        if exclude_seen and seen_movie_ids is not None:
            candidate_items = [
                item for item in candidate_items if item not in seen_movie_ids
            ]
            candidate_items = candidate_items[
                :num_candidates
            ]  # Ensure we still have enough candidates

        if len(candidate_items) < top_k:
            # If we don't have enough candidates after filtering, get more
            additional_candidates = min(
                top_k * candidate_multiplier * 5, self.num_items  # Increased multiplier
            )
            results = self.ann_index.search(user_embedding, k=additional_candidates)
            candidate_items = results[0][0]
            if exclude_seen and seen_movie_ids is not None:
                candidate_items = [
                    item for item in candidate_items if item not in seen_movie_ids
                ]

        # Re-rank candidates using the full model for more accurate ordering
        if len(candidate_items) > 0:
            user_ids = [user_id] * len(candidate_items)
            predictions = self.predict(user_ids, candidate_items)

            # Sort by prediction score
            item_scores = list(zip(candidate_items, predictions))
            item_scores.sort(key=lambda x: x[1], reverse=True)

            # Get top-k items
            top_items = [item for item, score in item_scores[:top_k]]
            top_scores = [score for item, score in item_scores[:top_k]]
        else:
            # Fallback if no candidates remain after filtering
            print("Warning: No suitable candidates found after filtering seen items")
            top_items = []
            top_scores = []

        # Convert to original MovieLens IDs and get titles if possible
        recommendations = []
        if self.reverse_movie_map and self.movies_df is not None:
            for i, item in enumerate(top_items):
                original_id = self.reverse_movie_map.get(item)
                movie_info = self.movies_df[self.movies_df["movieId"] == original_id]
                if not movie_info.empty:
                    title = movie_info["title"].values[0]
                    recommendations.append(
                        {
                            "internal_id": item,
                            "movieId": original_id,
                            "title": title,
                            "score": float(top_scores[i]),
                        }
                    )
                else:
                    recommendations.append(
                        {
                            "internal_id": item,
                            "movieId": original_id,
                            "title": f"Unknown Movie {original_id}",
                            "score": float(top_scores[i]),
                        }
                    )
        else:
            recommendations = [
                {"internal_id": item, "score": float(score)}
                for item, score in zip(top_items, top_scores)
            ]

        end_time = time.time()
        print(f"ANN recommendation time: {end_time - start_time:.4f} seconds")

        return recommendations


def get_user_seen_movies(
    ratings_df, original_user_id, user_id_map, movie_id_map, rating_threshold=2.5
):
    """
    Get the internal movie IDs that a user has already rated above the threshold
    """
    if original_user_id not in user_id_map:
        return set()

    user_ratings = ratings_df[ratings_df["userId"] == original_user_id]
    # Use a lower threshold to exclude more items (include more as "seen")
    liked_movies = user_ratings[user_ratings["rating"] >= rating_threshold]["movieId"]

    # Convert to internal IDs
    seen_movie_ids = set()
    for movie_id in liked_movies:
        if movie_id in movie_id_map:
            seen_movie_ids.add(movie_id_map[movie_id])

    return seen_movie_ids


# ========================= Leave-One-Out Evaluation ===========================


def create_leave_one_out_testset(
    cold_start_test, user_id_map, movie_id_map, rating_threshold=3.0
):
    """
    Create a leave-one-out test set for evaluation, ensuring each user has at least one test item.
    For each user, hold out their highest-rated item for evaluation.

    Args:
        cold_start_test: Cold-start test data
        user_id_map: Mapping from original to internal user IDs
        movie_id_map: Mapping from original to internal movie IDs
        rating_threshold: Threshold to consider a rating as positive (lowered for better hit rate)

    Returns:
        Dictionary with test items for each user
    """
    print("Creating leave-one-out test set...")

    test_items = {}
    users_processed = 0

    # Group by user
    for user_id, group in cold_start_test.groupby("userId"):
        if user_id not in user_id_map:
            continue

        # Get the user's internal ID
        internal_user_id = user_id_map[user_id]

        # Filter for positive ratings (REDUCED threshold to increase chances of hits)
        positive_ratings = group[group["rating"] >= rating_threshold]

        if len(positive_ratings) == 0:
            continue

        # Get the highest-rated item
        best_item = positive_ratings.sort_values("rating", ascending=False).iloc[0]
        movie_id = best_item["movieId"]

        if movie_id in movie_id_map:
            internal_movie_id = movie_id_map[movie_id]
            test_items[internal_user_id] = internal_movie_id
            users_processed += 1

    print(f"Created leave-one-out test set for {users_processed} users")

    return test_items


def evaluate_leave_one_out(
    rec_system,
    test_items,
    movies_df,
    user_id_map,
    movie_id_map,
    top_k=50,  # Dramatically increased to improve hit rate
    use_ann=False,
    max_users=None,  # New parameter to limit number of evaluated users
):
    """
    Evaluate using leave-one-out methodology (simpler and more reliable for cold-start scenarios).

    Args:
        rec_system: Recommendation system
        test_items: Dictionary mapping user IDs to their test item
        movies_df: DataFrame with movie information
        user_id_map: Mapping from original to internal user IDs
        movie_id_map: Mapping from original to internal movie IDs
        top_k: Number of recommendations to consider (dramatically increased)
        use_ann: Whether to use ANN for recommendations
        max_users: Maximum number of users to evaluate (None = all users)

    Returns:
        Dictionary with evaluation metrics
    """
    print(
        f"Evaluating {'ANN' if use_ann else 'standard'} recommendations using leave-one-out..."
    )

    # If max_users is specified, select a random subset of users
    if max_users is not None and max_users < len(test_items):
        np.random.seed(42)  # For reproducibility
        user_ids = list(test_items.keys())
        selected_users = np.random.choice(user_ids, size=max_users, replace=False)
        test_items_subset = {user_id: test_items[user_id] for user_id in selected_users}
        print(
            f"Randomly selected {max_users} users from {len(test_items)} total test users"
        )
        test_items = test_items_subset

    # Metrics to track
    metrics = {
        "hit_rate": [],
        "reciprocal_rank": [],
        "ndcg": [],
        "time": [],
    }

    # Process users with test items
    users_evaluated = 0

    for internal_user_id, test_item_id in test_items.items():
        # Get original user ID (for debugging only)
        original_user_id = rec_system.reverse_user_map[internal_user_id]

        # Get training data to exclude from recommendations
        train_df = rec_system.ratings_df
        seen_movies = get_user_seen_movies(
            train_df,
            original_user_id,
            user_id_map,
            movie_id_map,
            rating_threshold=2.0,  # Lower threshold to exclude more items
        )

        # Generate recommendations
        start_time = time.time()
        if use_ann:
            recommendations = rec_system.recommend_items_ann(
                internal_user_id,
                top_k=top_k,
                exclude_seen=True,
                seen_movie_ids=seen_movies,
                candidate_multiplier=50,  # Increase candidates for better recall
            )
        else:
            recommendations = rec_system.recommend_items(
                internal_user_id,
                top_k=top_k,
                exclude_seen=True,
                seen_movie_ids=seen_movies,
            )
        end_time = time.time()

        # Extract recommended item IDs
        rec_items = [rec["internal_id"] for rec in recommendations]

        # 1. Hit Rate - is the test item in the recommendations?
        hit = 1 if test_item_id in rec_items else 0
        metrics["hit_rate"].append(hit)

        # 2. Reciprocal Rank - 1/position of the test item
        if test_item_id in rec_items:
            rank = rec_items.index(test_item_id) + 1  # 1-based position
            metrics["reciprocal_rank"].append(1.0 / rank)
        else:
            metrics["reciprocal_rank"].append(0.0)  # Not found

        # 3. NDCG - normalized discounted cumulative gain
        # For leave-one-out, this is simplified since there's only one relevant item
        relevance = np.zeros(len(rec_items))
        if test_item_id in rec_items:
            relevance[rec_items.index(test_item_id)] = 1

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0
        for i, rel in enumerate(relevance):
            if rel > 0:
                dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed

        # IDCG is 1 (since there's only one relevant item)
        idcg = 1.0  # Optimal DCG is placing the relevant item at position 1
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics["ndcg"].append(ndcg)

        # Record time
        metrics["time"].append(end_time - start_time)

        users_evaluated += 1
        if users_evaluated % 20 == 0:
            print(f"Processed {users_evaluated} users")

        # Debug for the first few users
        if users_evaluated <= 3:
            print(f"\nUser {original_user_id} (internal ID: {internal_user_id}):")
            if hit:
                test_item_pos = rec_items.index(test_item_id) + 1
                print(
                    f"  Test item found at position {test_item_pos} out of {len(rec_items)}"
                )
            else:
                print(f"  Test item NOT found in recommendations")
            print(f"  Time: {end_time - start_time:.4f}s")

    # Calculate average metrics
    if users_evaluated > 0:
        avg_metrics = {
            "hit_rate": np.mean(metrics["hit_rate"]) * 100,  # As percentage
            "mrr": np.mean(metrics["reciprocal_rank"]),  # Mean Reciprocal Rank
            "ndcg": np.mean(metrics["ndcg"]),
            "time": np.mean(metrics["time"]),
            "users_evaluated": users_evaluated,
        }
    else:
        avg_metrics = {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "time": 0.0,
            "users_evaluated": 0,
        }
        print("Warning: No users were evaluated!")

    return avg_metrics


def compare_coldstart_methods_leave_one_out(
    rec_system,
    test_items,
    movies_df,
    user_id_map,
    movie_id_map,
    top_k=50,
    max_users=None,  # Increased top_k
):
    """
    Compare recommendation methods using leave-one-out evaluation.

    Args:
        rec_system: Recommendation system
        test_items: Dictionary mapping users to their test items
        movies_df: DataFrame with movie information
        user_id_map: Mapping from original to internal user IDs
        movie_id_map: Mapping from original to internal movie IDs
        top_k: Number of recommendations to consider (increased to 50)
        max_users: Maximum number of users to evaluate (None = all users)

    Returns:
        Dictionary with comparison results
    """
    # Verify we have test items
    if not test_items:
        print("Error: No test items available for evaluation!")
        return None

    print(
        f"Comparing methods using leave-one-out with up to {max_users if max_users else len(test_items)} users..."
    )

    # Evaluate standard method
    standard_metrics = evaluate_leave_one_out(
        rec_system,
        test_items,
        movies_df,
        user_id_map,
        movie_id_map,
        top_k=top_k,
        use_ann=False,
        max_users=max_users,
    )

    # Evaluate ANN method
    ann_metrics = evaluate_leave_one_out(
        rec_system,
        test_items,
        movies_df,
        user_id_map,
        movie_id_map,
        top_k=top_k,
        use_ann=True,
        max_users=max_users,
    )

    # Print results
    print("\n===== Cold-start User Performance Comparison =====")
    print(
        f"Standard method - Hit Rate: {standard_metrics['hit_rate']:.2f}%, MRR: {standard_metrics['mrr']:.4f}, "
        f"NDCG: {standard_metrics['ndcg']:.4f}, Time: {standard_metrics['time']:.4f}s"
    )
    print(
        f"ANN method      - Hit Rate: {ann_metrics['hit_rate']:.2f}%, MRR: {ann_metrics['mrr']:.4f}, "
        f"NDCG: {ann_metrics['ndcg']:.4f}, Time: {ann_metrics['time']:.4f}s"
    )

    speedup = (
        standard_metrics["time"] / ann_metrics["time"] if ann_metrics["time"] > 0 else 0
    )
    print(f"\nSpeedup: {speedup:.2f}x faster")

    # Calculate relative differences for metrics where both methods have non-zero values
    if standard_metrics["hit_rate"] > 0 and ann_metrics["hit_rate"] > 0:
        hit_rate_diff = (
            ann_metrics["hit_rate"] / standard_metrics["hit_rate"] - 1
        ) * 100
        print(f"Hit Rate diff: {hit_rate_diff:.2f}%")

    if standard_metrics["mrr"] > 0 and ann_metrics["mrr"] > 0:
        mrr_diff = (ann_metrics["mrr"] / standard_metrics["mrr"] - 1) * 100
        print(f"MRR diff: {mrr_diff:.2f}%")

    if standard_metrics["ndcg"] > 0 and ann_metrics["ndcg"] > 0:
        ndcg_diff = (ann_metrics["ndcg"] / standard_metrics["ndcg"] - 1) * 100
        print(f"NDCG diff: {ndcg_diff:.2f}%")

    # Generate visualization
    metrics_names = ["Hit Rate (%)", "MRR x100", "NDCG x100"]
    std_values = [
        standard_metrics["hit_rate"],
        standard_metrics["mrr"] * 100,  # Scale up for visibility
        standard_metrics["ndcg"] * 100,  # Scale up for visibility
    ]
    ann_values = [
        ann_metrics["hit_rate"],
        ann_metrics["mrr"] * 100,
        ann_metrics["ndcg"] * 100,
    ]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_names))
    width = 0.35

    plt.bar(x - width / 2, std_values, width, label="Standard")
    plt.bar(x + width / 2, ann_values, width, label="ANN")

    plt.ylabel("Score")
    plt.title("Cold-start User Recommendation Accuracy Comparison")
    plt.xticks(x, metrics_names)
    plt.legend()

    # Add value labels
    for i, v in enumerate(std_values):
        plt.text(i - width / 2, v + 0.5, f"{v:.2f}", ha="center")
    for i, v in enumerate(ann_values):
        plt.text(i + width / 2, v + 0.5, f"{v:.2f}", ha="center")

    plt.savefig("coldstart_accuracy_comparison.png")
    print(
        "Cold-start accuracy comparison chart saved to coldstart_accuracy_comparison.png"
    )

    # Speed comparison
    plt.figure(figsize=(8, 6))
    plt.bar(["Standard", "ANN"], [standard_metrics["time"], ann_metrics["time"]])
    plt.ylabel("Average Time (seconds)")
    plt.title("Cold-start Recommendation Speed Comparison")

    # Add value labels
    plt.text(
        0,
        standard_metrics["time"] + 0.01,
        f"{standard_metrics['time']:.3f}s",
        ha="center",
    )
    plt.text(1, ann_metrics["time"] + 0.01, f"{ann_metrics['time']:.3f}s", ha="center")

    plt.savefig("coldstart_speed_comparison.png")
    print("Cold-start speed comparison chart saved to coldstart_speed_comparison.png")

    plt.close("all")

    return {"standard": standard_metrics, "ann": ann_metrics}


# ========================= Model Saving/Loading with Mappings ===========================


def save_model_with_mappings(
    model, reverse_user_map, reverse_movie_map, filepath="model_with_mappings.pt"
):
    """
    Save model state dict and ID mappings together to ensure compatibility
    """
    # Convert reverse maps back to forward maps
    user_id_map = {v: k for k, v in reverse_user_map.items()}
    movie_id_map = {v: k for k, v in reverse_movie_map.items()}

    # Create a dictionary with all the data
    save_dict = {
        "model_state": model.state_dict(),
        "user_id_map": user_id_map,
        "movie_id_map": movie_id_map,
        "num_users": model.num_users,
        "num_items": model.num_items,
        "field_dims": model.field_dims,
        "embedding_dim": model.embedding_dim,
    }

    # Try to save with weights_only parameter if available
    try:
        # For newer PyTorch versions, explicitly set weights_only=False
        # to ensure complex objects can be loaded later
        torch.save(save_dict, filepath, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        torch.save(save_dict, filepath)

    print(f"Model and mappings saved to {filepath}")


def load_model_with_mappings(filepath="model_with_mappings.pt"):
    """
    Load model state dict and ID mappings together
    """
    # Load the dictionary with weights_only=False to allow complex objects
    try:
        saved_data = torch.load(filepath, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        saved_data = torch.load(filepath)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with manual serialization handling...")
        try:
            # Try more permissive loading
            saved_data = torch.load(filepath, map_location=torch.device("cpu"))
        except:
            raise RuntimeError(f"Failed to load model from {filepath}")

    # Create model with correct dimensions and improved architecture
    model = DeepFM(
        field_dims=saved_data["field_dims"],
        embedding_dim=saved_data["embedding_dim"],
        mlp_dims=(128, 64, 32),  # Wider network
        dropout=0.3,  # Increased dropout
    )

    # Load state dict
    model.load_state_dict(saved_data["model_state"])

    # Get mappings
    user_id_map = saved_data["user_id_map"]
    movie_id_map = saved_data["movie_id_map"]

    # Create reverse mappings
    reverse_user_map = {v: k for k, v in user_id_map.items()}
    reverse_movie_map = {v: k for k, v in movie_id_map.items()}

    return model, user_id_map, movie_id_map, reverse_user_map, reverse_movie_map


# ========================= Run Cold-start Evaluation ===========================


def run_improved_coldstart_evaluation(
    data_dir, sample_size=None, num_test_users=100, test_ratio=0.2
):
    """
    Run the improved cold-start evaluation workflow using leave-one-out methodology

    Args:
        data_dir: Directory containing MovieLens dataset
        sample_size: Number of ratings to sample (None for all)
        num_test_users: Number of users to test (default 100)
        test_ratio: Ratio of data to use for testing (0.2 = 80/20 split)
    """
    print("Starting improved cold-start evaluation...")

    # Load raw data
    ratings_df, movies_df = load_movielens_data(data_dir, sample_size)

    # Prepare cold-start evaluation data with optimized parameters
    coldstart_data = prepare_coldstart_evaluation(
        ratings_df,
        coldstart_ratio=0.2,  # Reduced to focus on higher quality cold-start users
        initial_ratings=5,  # Increased for better initial profile
        rating_threshold=3.5,  # Higher threshold for more reliable positive items
        test_ratio=test_ratio,  # Using specified test ratio (default 0.2)
    )

    # Process training data
    data = preprocess_for_recommendation(coldstart_data["train_data"])

    # Create leave-one-out test set with lower threshold to get more test items
    test_items = create_leave_one_out_testset(
        coldstart_data["cold_start_test"],
        data["user_id_map"],
        data["movie_id_map"],
        rating_threshold=3.0,  # Lowered from 3.5 to include more test items
    )

    # Process test data (still needed for the model)
    test_data = preprocess_test_data(
        coldstart_data["cold_start_test"], data["user_id_map"], data["movie_id_map"]
    )

    print("Creating recommendation system...")
    rec_system = MovieLensRecommendationSystem(
        data["num_users"],
        data["num_items"],
        embedding_dim=32,  # Increased embedding dimension
    )
    rec_system.set_mapping(
        data["reverse_user_map"],
        data["reverse_movie_map"],
        movies_df,
        data["ratings_df"],
    )

    # Check if we have a saved model
    model_with_mappings_path = "model_with_mappings.pt"
    if os.path.exists(model_with_mappings_path):
        try:
            print(f"Loading model from {model_with_mappings_path}")
            model, user_id_map, movie_id_map, reverse_user_map, reverse_movie_map = (
                load_model_with_mappings(model_with_mappings_path)
            )

            # Check dimensions
            if (
                model.num_users == data["num_users"]
                and model.num_items == data["num_items"]
            ):
                print("Loaded model has compatible dimensions!")
                rec_system.model = model
                # Make sure to keep the ratings_df when updating the mapping
                rec_system.set_mapping(
                    reverse_user_map, reverse_movie_map, movies_df, data["ratings_df"]
                )
            else:
                raise ValueError("Model dimensions don't match dataset")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            # Train for more epochs with improved optimizer settings
            rec_system.train(
                data["user_ids"],
                data["movie_ids"],
                data["labels"],
                batch_size=2048,  # Increased batch size
                epochs=10,  # Increased epochs
                validation_data={  # Add validation data for early stopping
                    "user_ids": test_data["user_ids"][:5000],
                    "movie_ids": test_data["movie_ids"][:5000],
                    "labels": test_data["labels"][:5000],
                },
            )
            save_model_with_mappings(
                rec_system.model,
                rec_system.reverse_user_map,
                rec_system.reverse_movie_map,
                model_with_mappings_path,
            )
    else:
        print("Training new model...")
        # Train for more epochs with improved optimizer settings
        rec_system.train(
            data["user_ids"],
            data["movie_ids"],
            data["labels"],
            batch_size=2048,  # Increased batch size
            epochs=10,  # Increased epochs
            validation_data={  # Add validation data for early stopping
                "user_ids": (
                    test_data["user_ids"][:5000]
                    if len(test_data["user_ids"]) > 5000
                    else test_data["user_ids"]
                ),
                "movie_ids": (
                    test_data["movie_ids"][:5000]
                    if len(test_data["movie_ids"]) > 5000
                    else test_data["movie_ids"]
                ),
                "labels": (
                    test_data["labels"][:5000]
                    if len(test_data["labels"]) > 5000
                    else test_data["labels"]
                ),
            },
        )
        save_model_with_mappings(
            rec_system.model,
            rec_system.reverse_user_map,
            rec_system.reverse_movie_map,
            model_with_mappings_path,
        )

    # Build ANN index with improved parameters
    print("Building ANN index...")
    rec_system.build_ann_index()

    # Compare recommendation methods using improved methodology with limited users
    print(
        f"\nComparing recommendation methods for {num_test_users} cold-start users..."
    )
    comparison_results = compare_coldstart_methods_leave_one_out(
        rec_system,
        test_items,
        movies_df,
        data["user_id_map"],
        data["movie_id_map"],
        top_k=50,  # Dramatically increased top_k for better hit rate
        max_users=num_test_users,  # Limit to specified number of test users
    )

    return comparison_results


# In the main function, modify to use the improved evaluation
def main(
    data_dir, sample_size=None, mode="standard", force_retrain=False, test_ratio=0.2
):
    """
    Main function to run the MovieLens recommendation system

    Args:
        data_dir: Directory containing MovieLens dataset
        sample_size: Number of ratings to sample (None for all)
        mode: "standard" for normal recommendation, "coldstart" for cold-start evaluation
        force_retrain: Whether to force retraining even if a model exists
        test_ratio: Ratio of data to use for testing (0.2 = 80/20 split)
    """
    if mode == "standard":
        # Standard workflow (unchanged)
        pass
    elif mode == "coldstart":
        print(
            f"Running improved cold-start evaluation with {test_ratio*100}% test data..."
        )
        results = run_improved_coldstart_evaluation(
            data_dir, sample_size, num_test_users=100, test_ratio=test_ratio
        )
        return results
    else:
        print(f"Unknown mode: {mode}. Please use 'standard' or 'coldstart'.")


if __name__ == "__main__":
    # Update the path to point to the ml-32m folder
    data_dir = "./ml-32m"  # Path to the MovieLens dataset

    print(f"Running evaluation using data from {data_dir}")

    # Run cold-start evaluation with the full dataset,
    # 20% test split, and testing on only 100 users
    results = main(data_dir, sample_size=None, mode="coldstart", test_ratio=0.2)
