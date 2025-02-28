from manim import *
import numpy as np
import random


class HNSWAnimation(Scene):
    def construct(self):
        # Configuration
        num_points = 20  # Reduced from 30
        num_layers = 3
        seed = 42
        random.seed(seed)
        np.random.seed(seed)

        # Create layers group
        layers = VGroup()
        graph_objects = {}
        points_by_layer = {}

        # Generate base points (2D for visualization)
        base_points = [(np.random.uniform(-3, 3), np.random.uniform(-1, 1))
                       for _ in range(num_points)]
        point_objects = {}

        # Calculate probabilities for each layer
        layer_probs = [1]
        for i in range(1, num_layers):
            layer_probs.append(layer_probs[-1] / 4)

        # Assign points to layers
        for layer_idx in range(num_layers):
            points_by_layer[layer_idx] = []

            # For layer 0 (base layer), include all points
            if layer_idx == 0:
                points_by_layer[layer_idx] = base_points.copy()
            else:
                # For higher layers, include points based on probability
                for point in base_points:
                    if random.random() < layer_probs[layer_idx]:
                        points_by_layer[layer_idx].append(point)

        # Create the visual representation of each layer
        layer_separation = 2.0  # Reduced from 2.5
        for layer_idx in range(num_layers):
            # Create a layer rectangle
            layer_height = 1.2  # Reduced from 1.5
            layer_width = 6.5   # Reduced from 8
            layer_rect = Rectangle(
                height=layer_height, width=layer_width, color=BLUE_A, fill_opacity=0.2)
            layer_rect.move_to(DOWN * (layer_idx * layer_separation - 1))

            # Create point objects
            layer_points = VGroup()
            for i, point in enumerate(points_by_layer[layer_idx]):
                x, y = point

                # Scale x to fit within the layer
                x = x * (layer_width / 8)

                # Position the point in the layer
                pos = layer_rect.get_center() + np.array([x, 0, 0])
                dot = Dot(pos, color=BLUE, radius=0.08)  # Smaller dots

                # Store reference to the dot
                point_key = (layer_idx, i)
                point_objects[point_key] = dot
                layer_points.add(dot)

            # Group the layer elements
            layer_group = VGroup(layer_rect, layer_points)
            layers.add(layer_group)

            # Create connections within the layer (small world graph)
            edges = VGroup()
            for i, point1 in enumerate(points_by_layer[layer_idx]):
                # Connect to k nearest neighbors
                k = min(3, len(points_by_layer[layer_idx]) - 1)

                # Calculate distances to all other points
                distances = []
                for j, point2 in enumerate(points_by_layer[layer_idx]):
                    if i != j:
                        dist = np.sqrt(
                            (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                        distances.append((j, dist))

                # Sort by distance and take k closest
                distances.sort(key=lambda x: x[1])
                for j, _ in distances[:k]:
                    dot1 = point_objects[(layer_idx, i)]
                    dot2 = point_objects[(layer_idx, j)]
                    edge = Line(dot1.get_center(), dot2.get_center(),
                                color=BLUE_D, stroke_opacity=0.5, stroke_width=1)
                    edges.add(edge)

            graph_objects[layer_idx] = edges

        # Show all layers
        self.play(Create(layers))
        self.wait(1)

        # Show connections within each layer
        for layer_idx in range(num_layers):
            self.play(Create(graph_objects[layer_idx]), run_time=1)
        self.wait(1)

        # Animate a search query
        # Create a query point
        query_pos = np.array([1.5, 0, 0])
        query_point = Dot(layers[0][0].get_center() +
                          query_pos, color=RED, radius=0.12)

        self.play(Create(query_point))
        self.wait(1)

        # Simulate the HNSW search process
        # 1. Start at random entry point in the top layer
        top_layer = num_layers - 1
        if len(points_by_layer[top_layer]) > 0:
            entry_idx = random.randint(0, len(points_by_layer[top_layer]) - 1)
            entry_dot = point_objects[(top_layer, entry_idx)]

            # Highlight entry point
            entry_highlight = Dot(entry_dot.get_center(),
                                  color=YELLOW, radius=0.12)
            self.play(Create(entry_highlight))
            self.wait(0.5)

            # Search in each layer
            for layer_idx in range(top_layer, -1, -1):
                # Simulate greedy search in this layer
                current_idx = entry_idx if layer_idx == top_layer else best_idx
                current_dot = point_objects.get((layer_idx, current_idx), None)

                if current_dot is None:
                    # Find a new entry point if the best point from upper layer doesn't exist here
                    if len(points_by_layer[layer_idx]) > 0:
                        current_idx = random.randint(
                            0, len(points_by_layer[layer_idx]) - 1)
                        current_dot = point_objects[(layer_idx, current_idx)]

                if current_dot:
                    # Highlight current point
                    current_highlight = Dot(
                        current_dot.get_center(), color=YELLOW, radius=0.12)
                    self.play(Create(current_highlight))

                    # Find the best neighbor (simulate the search)
                    best_dist = float('inf')
                    best_idx = current_idx

                    # Examine neighbors
                    neighbors_visited = []

                    # Find neighbors by examining the lines connected to current_dot
                    for obj in graph_objects[layer_idx]:
                        if isinstance(obj, Line):
                            if np.array_equal(obj.get_start(), current_dot.get_center()):
                                for key, dot in point_objects.items():
                                    if key[0] == layer_idx and np.array_equal(dot.get_center(), obj.get_end()):
                                        neighbors_visited.append((key[1], dot))
                            elif np.array_equal(obj.get_end(), current_dot.get_center()):
                                for key, dot in point_objects.items():
                                    if key[0] == layer_idx and np.array_equal(dot.get_center(), obj.get_start()):
                                        neighbors_visited.append((key[1], dot))

                    # Visualize checking neighbors
                    for neighbor_idx, neighbor_dot in neighbors_visited:
                        # Highlight neighbor being checked
                        check_line = Line(current_dot.get_center(
                        ), neighbor_dot.get_center(), color=YELLOW)
                        neighbor_highlight = Dot(
                            neighbor_dot.get_center(), color=YELLOW, radius=0.12)
                        self.play(Create(check_line), Create(
                            neighbor_highlight), run_time=0.4)

                        # Calculate distance to query
                        query_projected = query_point.get_center().copy()
                        query_projected[1] = neighbor_dot.get_center()[1]
                        dist_line = DashedLine(
                            neighbor_dot.get_center(), query_projected, color=RED)
                        self.play(Create(dist_line), run_time=0.4)

                        # Determine if this is the best neighbor so far
                        dist = np.linalg.norm(
                            query_projected - neighbor_dot.get_center())
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = neighbor_idx

                            # Update best
                            best_highlight = Dot(
                                neighbor_dot.get_center(), color=GREEN, radius=0.12)
                            self.play(Create(best_highlight), run_time=0.4)

                        # Clean up
                        self.play(
                            FadeOut(check_line),
                            FadeOut(neighbor_highlight),
                            FadeOut(dist_line),
                            run_time=0.2
                        )

                    # Show best point found in this layer
                    if len(neighbors_visited) > 0:
                        best_point = point_objects[(layer_idx, best_idx)]
                        best_final = Dot(best_point.get_center(),
                                         color=GREEN, radius=0.12)
                        self.play(
                            FadeOut(current_highlight),
                            Create(best_final)
                        )

                    # Move to next layer
                    self.wait(0.5)

        # Finish animation
        self.wait(2)
