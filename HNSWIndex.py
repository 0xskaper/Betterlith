import hnswlib
import numpy as np
import torch

class HNSWIndex:
    def __init__(self, dimension):
        """
        Initialize the HNSW index.
        :param dimension: Dimension of the embedding vectors.
        """
        self.dimension = dimension
        self.index = hnswlib.Index(space="l2", dim=dimension)  # Default to Euclidean distance (L2)
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
        assert len(item_ids) == item_embeddings.shape[0], "item_ids and item_embeddings must have the same length"
        assert item_embeddings.shape[1] == self.dimension, "Embedding dimension does not match initialization parameter"

        # Convert PyTorch tensor to NumPy array
        embeddings_np = item_embeddings.cpu().numpy().astype(np.float32)
        num_items = embeddings_np.shape[0]

        # Initialize the HNSW index
        self.index.init_index(
            max_elements=num_items,  # Maximum capacity (matches dataset size)
            ef_construction=200,     # Controls the quality/speed trade-off during construction
            M=16                     # Number of inter-layer connections (affects memory and performance)
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


# Test code
if __name__ == "__main__":
    # Initialize the index
    dim = 128
    index = HNSWIndex(dimension=dim)

    # Generate mock data
    num_items = 1000
    item_ids = list(range(num_items))  # Assume item_ids are consecutive integers
    item_embeddings = torch.randn(num_items, dim)  # Random embeddings

    # Build the index
    num_indexed = index.build(item_ids, item_embeddings)
    print(f"Indexed {num_indexed} items")

    # Generate a query
    query = torch.randn(1, dim)  # Single query
    results = index.search(query, k=3)

    # Output results
    print("Top 3 nearest neighbors:")
    for item_id, distance in zip(results[0][0], results[0][1]):
        print(f"Item ID: {item_id}, Distance: {distance:.4f}")