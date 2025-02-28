from manim import *
import numpy as np


class Embedding2DANNAnimation(Scene):
    def construct(self):
        # Set custom background color
        self.camera.background_color = "#f2e5bf"

        # Set up colors and constants
        EMBEDDING_COLOR = "#1f2022"
        QUERY_COLOR = "#cc241d"
        HIGHLIGHTED_COLOR = GREEN
        SEARCH_COLOR = "#1f2022"
        NEIGHBOR_COLOR = "#076678"
        TOP_K_NEIGHBOR_COLOR = "#cc241d"  # Color for top-k neighbors

        # Number of embeddings and dimensions
        num_embeddings = 30
        embedding_dim = 2  # Using 2D for visualization
        k_neighbors = 10  # Number of nearest neighbors to highlight

        # Scale for visualization with constraints to keep within frame
        scale_factor = 2.5  # Reduced to keep dots in frame

        # Create 2D embedding points
        np.random.seed(42)  # For reproducibility
        embeddings = []
        for _ in range(num_embeddings):
            # Generate and clip 2D coordinates
            x = np.clip(np.random.normal(0, 1), -1.5, 1.5)
            y = np.clip(np.random.normal(0, 1), -1.5, 1.5)
            embeddings.append(np.array([x, y]))

        # Create dots for embeddings
        embedding_dots = VGroup(*[
            Dot(point=np.array([emb[0] * scale_factor, emb[1] * scale_factor, 0]),
                color=EMBEDDING_COLOR,
                z_index=1)  # Ensure dots are in front
            for emb in embeddings
        ])

        # Create embedding vectors visualization
        self.play(FadeIn(embedding_dots))
        self.wait(1)

        # Create query vector (with appropriate scaling)
        x = np.clip(np.random.normal(0, 1), -1.5, 1.5)
        y = np.clip(np.random.normal(0, 1), -1.5, 1.5)
        query_vector = np.array([x, y])
        query_dot = Dot(
            point=np.array([query_vector[0] * scale_factor,
                           query_vector[1] * scale_factor, 0]),
            color=QUERY_COLOR,
            radius=0.15,
            z_index=2  # Query dot should be on top of everything
        )

        self.play(FadeIn(query_dot))
        self.wait(1)

        # PART 1: LINEAR SEARCH VISUALIZATION

        # Simulate linear search
        search_lines = []
        for i, emb in enumerate(embeddings):
            line = Line(
                start=query_dot.get_center(),
                end=np.array(
                    [emb[0] * scale_factor, emb[1] * scale_factor, 0]),
                color=SEARCH_COLOR,
                z_index=-1  # Ensure lines are behind dots
            )
            search_lines.append(line)
            self.play(Create(line), run_time=0.15)

        # Calculate distances (using original vectors, not scaled)
        distances = [np.linalg.norm(query_vector - emb) for emb in embeddings]

        # Get indices of top k nearest neighbors
        top_k_indices = np.argsort(distances)[:k_neighbors]

        # The nearest neighbor (first in top_k_indices)
        nearest_idx = top_k_indices[0]

        # Store references to the top-k lines from the original search lines
        top_k_lines = [search_lines[i]
                       for i in top_k_indices[1:]]  # Skip nearest
        nearest_line = search_lines[nearest_idx]

        # Fade out all lines EXCEPT those to the top-k neighbors
        lines_to_fade = [line for i, line in enumerate(
            search_lines) if i not in top_k_indices]
        self.play(*[FadeOut(line) for line in lines_to_fade])
        self.wait(0.5)

        # Change the color and width of the remaining lines to highlight them
        self.play(
            *[line.animate.set_color(TOP_K_NEIGHBOR_COLOR).set_stroke(width=3)
              for line in top_k_lines],
            nearest_line.animate.set_color(
                HIGHLIGHTED_COLOR).set_stroke(width=4)
        )

        # Highlight the 9 next-nearest neighbors with their lines
        top_k_animations = []
        for idx in top_k_indices[1:]:
            top_k_animations.append(
                embedding_dots[idx].animate.set_color(TOP_K_NEIGHBOR_COLOR))
            top_k_animations.append(embedding_dots[idx].animate.scale(1.2))

        self.play(*top_k_animations, *[Create(line) for line in top_k_lines])
        self.wait(0.5)

        # Highlight the nearest neighbor with its line
        self.play(
            embedding_dots[nearest_idx].animate.set_color(HIGHLIGHTED_COLOR),
            embedding_dots[nearest_idx].animate.scale(1.5),
            Create(nearest_line)
        )
        self.wait(1)

        # CLEAR ALL TOP-K LINES COMPLETELY
        self.play(*[FadeOut(line)
                  for line in top_k_lines], FadeOut(nearest_line))
        self.wait(1)

        # Reset the highlighted dots and remove text
        reset_animations = []
        for idx in top_k_indices[1:]:
            reset_animations.append(
                embedding_dots[idx].animate.set_color(EMBEDDING_COLOR))
            reset_animations.append(embedding_dots[idx].animate.scale(1/1.2))

        reset_animations.append(
            embedding_dots[nearest_idx].animate.set_color(EMBEDDING_COLOR))
        reset_animations.append(
            embedding_dots[nearest_idx].animate.scale(1/1.5))

        self.play(*reset_animations)
        self.wait(1)

        # PART 2: ANN VISUALIZATION
        self.wait(1)

        # Create clusters using k-means algorithm
        num_clusters = 4

        # Simple k-means implementation (could be replaced with sklearn's KMeans)
        # Initialize cluster centers randomly from embeddings
        cluster_centers_idx = np.random.choice(
            range(num_embeddings), num_clusters, replace=False)
        cluster_centers = [embeddings[i].copy() for i in cluster_centers_idx]

        # Run k-means for a few iterations to get better cluster centers
        for _ in range(5):  # 5 iterations of k-means
            # Assign points to clusters
            clusters = [[] for _ in range(num_clusters)]
            for i, emb in enumerate(embeddings):
                distances = [np.linalg.norm(emb - center)
                             for center in cluster_centers]
                nearest_cluster = np.argmin(distances)
                clusters[nearest_cluster].append(i)

            # Update cluster centers
            for i, cluster in enumerate(clusters):
                if cluster:  # Only update if cluster has points
                    cluster_points = [embeddings[idx] for idx in cluster]
                    cluster_centers[i] = np.mean(cluster_points, axis=0)

        # Visualize cluster centers
        cluster_dots = VGroup(*[
            Dot(point=np.array([center[0] * scale_factor, center[1] * scale_factor, 0]),
                color=NEIGHBOR_COLOR, radius=0.15, z_index=1.5)  # Cluster dots above normal dots
            for center in cluster_centers
        ])

        # Create connections between embeddings and nearest cluster center
        clusters = [[] for _ in range(num_clusters)]
        for i, emb in enumerate(embeddings):
            # Find closest cluster
            cluster_distances = [np.linalg.norm(
                emb - center) for center in cluster_centers]
            nearest_cluster = np.argmin(cluster_distances)
            clusters[nearest_cluster].append(i)

        # Visualize clusters with connections
        cluster_connections = VGroup()
        for cluster_idx, point_indices in enumerate(clusters):
            for point_idx in point_indices:
                connection = Line(
                    start=np.array([embeddings[point_idx][0] * scale_factor,
                                   embeddings[point_idx][1] * scale_factor, 0]),
                    end=np.array([cluster_centers[cluster_idx][0] * scale_factor,
                                  cluster_centers[cluster_idx][1] * scale_factor, 0]),
                    color=NEIGHBOR_COLOR,
                    stroke_width=2,
                    stroke_opacity=0.5,
                    z_index=-1  # Ensure connections are behind dots
                )
                cluster_connections.add(connection)

        # Create connections first (so they'll be behind dots)
        self.play(Create(cluster_connections), run_time=2)
        self.wait(1)

        # Show cluster centers (will appear on top of connections)
        self.play(FadeIn(cluster_dots))
        self.wait(1)

        # Simulate ANN search - find closest cluster center
        cluster_search_lines = []

        # Create all lines first (will be behind dots)
        for cluster_idx in range(num_clusters):
            line = Line(
                start=query_dot.get_center(),
                end=cluster_dots[cluster_idx].get_center(),
                color=SEARCH_COLOR,
                z_index=-1  # Ensure lines are behind dots
            )
            cluster_search_lines.append(line)
            self.play(Create(line), run_time=0.3)

        # Find nearest cluster (using original vectors, not scaled)
        cluster_dists = [np.linalg.norm(query_vector - center)
                         for center in cluster_centers]
        nearest_cluster_idx = np.argmin(cluster_dists)

        # Highlight nearest cluster
        self.play(
            cluster_dots[nearest_cluster_idx].animate.set_color(
                HIGHLIGHTED_COLOR),
            cluster_dots[nearest_cluster_idx].animate.scale(1.5)
        )
        self.wait(1)

        # Remove other cluster search lines
        self.play(*[FadeOut(line)
                  for i, line in enumerate(cluster_search_lines) if i != nearest_cluster_idx])
        self.wait(0.5)

        # Search only within the nearest cluster
        nearest_cluster_points = clusters[nearest_cluster_idx]
        point_search_lines = []

        for point_idx in nearest_cluster_points:
            line = Line(
                start=query_dot.get_center(),
                end=embedding_dots[point_idx].get_center(),
                color=SEARCH_COLOR,
                z_index=-1  # Ensure lines are behind dots
            )
            point_search_lines.append(line)
            self.play(Create(line), run_time=0.2)

        # Find nearest neighbor within cluster
        cluster_point_dists = [(np.linalg.norm(query_vector - embeddings[idx]), idx)
                               for idx in nearest_cluster_points]

        if cluster_point_dists:  # Make sure the list is not empty
            _, approximate_nearest_idx = min(cluster_point_dists)
        else:
            # Fallback if no points in the cluster (shouldn't happen but good practice)
            approximate_nearest_idx = 0

        # Highlight approximate nearest neighbor
        self.play(
            embedding_dots[approximate_nearest_idx].animate.set_color(
                HIGHLIGHTED_COLOR),
            embedding_dots[approximate_nearest_idx].animate.scale(1.5)
        )
        self.wait(1)

        # Compare with true nearest (from linear search)
        if nearest_idx != approximate_nearest_idx:
            # Show the true nearest neighbor from linear search
            self.play(
                embedding_dots[nearest_idx].animate.set_color(RED),
                embedding_dots[nearest_idx].animate.scale(1.5)
            )

            # Create text to explain
            explanation = Text(
                "ANN found an approximate nearest neighbor\nwhich may differ from the true nearest",
                font_size=24
            ).to_edge(DOWN)

            self.play(FadeIn(explanation))
            self.wait(2)

            # Clean up
            self.play(
                embedding_dots[nearest_idx].animate.set_color(EMBEDDING_COLOR),
                embedding_dots[nearest_idx].animate.scale(1/1.5),
                FadeOut(explanation)
            )

        self.wait(1)

        # Fade everything out
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )
        self.wait(1)

# To render this animation, run:
# manim -pql embedding_ann_2d.py Embedding2DANNAnimation
# For high quality: manim -pqh embedding_ann_2d.py Embedding2DANNAnimation
