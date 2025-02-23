from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import box
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict


class DetailedAnalyzer:
    def __init__(self):
        self.console = Console()

    def analyze_training_metrics(self, epoch_losses: List[float]) -> Dict:
        """Analyze training loss progression."""
        return {
            "min_loss": min(epoch_losses),
            "max_loss": max(epoch_losses),
            "avg_loss": np.mean(epoch_losses),
            "std_loss": np.std(epoch_losses),
            "improvement": epoch_losses[0] - epoch_losses[-1],
            "improvement_percent": ((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0]) * 100
        }

    def analyze_recommendations(self, recommendations: List[Dict], user_features: Dict) -> Dict:
        """Analyze recommendation patterns."""
        categories = defaultdict(int)
        price_ranges = defaultdict(int)
        scores = []

        for rec in recommendations:
            categories[rec['category']] += 1
            price = rec['price']
            if price < 50:
                price_ranges['Low (<$50)'] += 1
            elif price < 200:
                price_ranges['Medium ($50-$200)'] += 1
            else:
                price_ranges['High (>$200)'] += 1
            scores.append(rec['score'])

        return {
            "category_distribution": dict(categories),
            "price_distribution": dict(price_ranges),
            "avg_score": np.mean(scores),
            "score_std": np.std(scores),
            "max_score": max(scores),
            "min_score": min(scores)
        }

    def print_detailed_results(self, training_metrics: Dict, recommendation_analysis: Dict):
        """Print comprehensive analysis results."""
        # Training Analysis
        self.console.print(
            "\n[bold cyan]Detailed Training Analysis[/bold cyan]")
        training_table = Table(
            show_header=True, header_style="bold magenta", box=box.DOUBLE_EDGE)
        training_table.add_column("Metric", style="cyan")
        training_table.add_column("Value", justify="right")

        training_table.add_row(
            "Final Loss", f"{training_metrics['min_loss']:.4f}")
        training_table.add_row(
            "Average Loss", f"{training_metrics['avg_loss']:.4f}")
        training_table.add_row(
            "Loss Std Dev", f"{training_metrics['std_loss']:.4f}")
        training_table.add_row("Total Improvement", f"{
                               training_metrics['improvement']:.4f}")
        training_table.add_row("Improvement %", f"{
                               training_metrics['improvement_percent']:.2f}%")

        self.console.print(
            Panel(training_table, title="Training Metrics", border_style="blue"))

        # Recommendation Analysis
        self.console.print("\n[bold cyan]Recommendation Analysis[/bold cyan]")
        rec_table = Table(show_header=True,
                          header_style="bold magenta", box=box.DOUBLE_EDGE)
        rec_table.add_column("Aspect", style="cyan")
        rec_table.add_column("Details", justify="right")

        # Category Distribution
        category_str = " | ".join(
            [f"Cat{k}: {v}" for k, v in recommendation_analysis['category_distribution'].items()])
        rec_table.add_row("Category Distribution", category_str)

        # Price Distribution
        price_str = " | ".join(
            [f"{k}: {v}" for k, v in recommendation_analysis['price_distribution'].items()])
        rec_table.add_row("Price Distribution", price_str)

        # Score Statistics
        rec_table.add_row("Average Score", f"{
                          recommendation_analysis['avg_score']:.4f}")
        rec_table.add_row("Score Std Dev", f"{
                          recommendation_analysis['score_std']:.4f}")
        rec_table.add_row("Score Range", f"{
                          recommendation_analysis['min_score']:.4f} - {recommendation_analysis['max_score']:.4f}")

        self.console.print(
            Panel(rec_table, title="Recommendation Patterns", border_style="green"))

    def print_feature_importance(self, model, feature_names: List[str]):
        """Analyze and print feature importance based on embeddings."""
        # Get embedding weights
        with torch.no_grad():
            first_order_weights = model.model.first_order.weight.abs().mean(dim=1)
            second_order_weights = model.model.second_order.weight.abs().mean(dim=1)

        # Combine weights
        total_importance = first_order_weights + second_order_weights.mean()

        # Create importance table
        importance_table = Table(
            show_header=True, header_style="bold yellow", box=box.DOUBLE_EDGE)
        importance_table.add_column("Feature", style="cyan")
        importance_table.add_column("Importance Score", justify="right")

        # Normalize and sort importances
        normalized_importance = total_importance / total_importance.sum()
        feature_importance = list(zip(feature_names, normalized_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for feature, importance in feature_importance[:10]:  # Top 10 features
            importance_table.add_row(
                feature,
                f"{importance.item()*100:.2f}%"
            )

        self.console.print(
            Panel(importance_table, title="Top Feature Importance", border_style="yellow"))

    def print_user_segment_analysis(self, user_features: Dict, recommendations: List[Dict]):
        """Analyze recommendations by user segment."""
        segment_table = Table(
            show_header=True, header_style="bold blue", box=box.DOUBLE_EDGE)
        segment_table.add_column("User Segment")
        segment_table.add_column("Preferred Categories")
        segment_table.add_column("Avg Price Range")
        segment_table.add_column("Avg Match Score")

        # Age segments
        age = user_features['age']
        if age < 30:
            age_segment = "Young Adult"
        elif age < 50:
            age_segment = "Middle Age"
        else:
            age_segment = "Senior"

        # Analyze preferences
        categories = defaultdict(int)
        avg_price = np.mean([rec['price'] for rec in recommendations])
        avg_score = np.mean([rec['score'] for rec in recommendations])

        for rec in recommendations:
            categories[rec['category']] += 1

        top_categories = sorted(
            categories.items(), key=lambda x: x[1], reverse=True)[:2]
        category_str = ", ".join(
            [f"Category {cat}" for cat, _ in top_categories])

        segment_table.add_row(
            f"{age_segment} ({age} years)",
            category_str,
            f"${avg_price:.2f}",
            f"{avg_score:.2%}"
        )

        self.console.print(
            Panel(segment_table, title="User Segment Analysis", border_style="blue"))
