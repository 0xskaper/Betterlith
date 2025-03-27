# DeepFM-ANN Recommendation System

This repository contains an optimized recommendation system for the MovieLens dataset, with a specific focus on cold-start user scenarios using a hybrid approach combining DeepFM and Approximate Nearest Neighbors (ANN).

## Overview

The recommendation system combines several state-of-the-art techniques:

- **DeepFM Model**: Combines factorization machines for recommendation with deep neural networks
- **HNSW Approximate Nearest Neighbors**: For fast candidate retrieval during recommendation
- **Cold-start Optimization**: Specialized handling for users with limited rating history

## Key Features

- **Efficient Recommendations**: ANN-based retrieval enables fast recommendations with minimal accuracy loss
- **Cold-start Handling**: Optimized for new users with limited rating history
- **Comprehensive Evaluation**: Leave-one-out methodology with relevant metrics (Hit Rate, MRR, NDCG)
- **Memory-efficient Design**: Batch processing for handling large datasets
- **Model Persistence**: Save/load capabilities for model weights and mappings

## Requirements

```
numpy
pandas
torch
hnswlib
scikit-learn
matplotlib
tqdm
```

## Usage

### Basic Usage

```python
# Load and preprocess data
ratings_df, movies_df = load_movielens_data("path/to/movielens")
data = preprocess_for_recommendation(ratings_df)

# Create recommendation system
rec_system = MovieLensRecommendationSystem(data["num_users"], data["num_items"])
rec_system.set_mapping(data["reverse_user_map"], data["reverse_movie_map"], movies_df, data["ratings_df"])

# Train model
rec_system.train(data["user_ids"], data["movie_ids"], data["labels"], epochs=10)

# Build ANN index for fast recommendations
rec_system.build_ann_index()

# Get recommendations for a user
recommendations = rec_system.recommend_items_ann(user_id, top_k=10)
```

### Cold-start Evaluation

```python
# Run cold-start evaluation
results = run_improved_coldstart_evaluation(
    "path/to/movielens", 
    sample_size=None,
    num_test_users=100,
    test_ratio=0.2
)
```

## Structure

The code is organized into several key components:

1. **Data Loading and Processing**
   - `load_movielens_data()`: Loads and samples MovieLens dataset
   - `prepare_coldstart_evaluation()`: Creates training/testing splits for cold-start evaluation
   - `preprocess_for_recommendation()`: Prepares data for the recommendation model

2. **Core Models**
   - `DeepFM`: Neural recommendation model combining factorization machines and deep learning
   - `HNSWIndex`: Wrapper for HNSW approximate nearest neighbors index
   - `CollisionlessEmbeddingTable`: Embedding table with expiration mechanism

3. **Recommendation System**
   - `MovieLensRecommendationSystem`: Main class that integrates models for recommendations
   - Methods for both standard and ANN-based recommendations

4. **Evaluation**
   - `create_leave_one_out_testset()`: Creates test data for leave-one-out evaluation
   - `evaluate_leave_one_out()`: Evaluates recommendation methods using leave-one-out methodology
   - `compare_coldstart_methods_leave_one_out()`: Compares different recommendation approaches

## Cold-start Optimization

The system handles cold-start users with several strategies:

1. Strategic initial rating selection based on user preferences
2. Larger candidate pool during ANN retrieval for better recall
3. Hybrid re-ranking approach combining popularity and similarity
4. Optimized thresholds for seen/unseen item determination

## Evaluation Metrics

The system is evaluated using:

- **Hit Rate (HR@k)**: Percentage of users for whom the held-out item is in the top-k recommendations
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of the held-out items
- **Normalized Discounted Cumulative Gain (NDCG)**: Measures the ranking quality of recommendations

## Results

Comparative evaluation between standard and ANN-based recommendation methods:

- ANN method is significantly faster (typically 5-10x speedup)
- Loss in recommendation quality (comparable Hit Rate, MRR, and NDCG)

## License

[MIT License](LICENSE)
