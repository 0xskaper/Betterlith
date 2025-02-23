from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from typing import Dict, List


class BentoResults:
    def __init__(self):
        self.console = Console()

    def create_header(self, text: str) -> Panel:
        """Create a header with custom style."""
        return Panel(
            Text(text, style="bold magenta", justify="center"),
            box=box.SQUARE,
            style="magenta"
        )

    def create_metrics_panel(self, title: str, metrics: Dict) -> Panel:
        """Create a panel for metrics display."""
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column(justify="right")

        for key, value in metrics.items():
            color = "green" if isinstance(
                value, float) and value > 0.8 else "yellow"
            grid.add_row(
                f"[blue]{key}[/blue]",
                f"[{color}]{value}[/{color}]"
            )

        return Panel(
            grid,
            title=f"[b]{title}[/b]",
            box=box.ROUNDED,
            border_style="blue"
        )

    def create_recommendation_table(self, recommendations: List[Dict]) -> Table:
        """Create a table for recommendations."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.MINIMAL_DOUBLE_HEAD,
            expand=True
        )

        table.add_column("Item", justify="center")
        table.add_column("Category", justify="left")
        table.add_column("Price", justify="right")
        table.add_column("Score", justify="right")

        for rec in recommendations:
            score_color = "green" if rec["score"] > 0.8 else "yellow"
            table.add_row(
                f"[white]{rec['id']}[/white]",
                f"[blue]{rec['category']}[/blue]",
                f"[yellow]${rec['price']:.2f}[/yellow]",
                f"[{score_color}]{rec['score']*100:.1f}%[/{score_color}]"
            )

        return table

    def display_results(self, results: Dict):
        """Display all results in bento grid style."""
        # Create layout
        layout = Layout()

        # Split into main sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body")
        )

        # Add header
        layout["header"].update(
            self.create_header("🎯 Monolith Re  commendation System Results")
        )

        # Split body into rows
        layout["body"].split_row(
            Layout(name="left_column", ratio=1),
            Layout(name="right_column", ratio=1)
        )

        # Split left column
        layout["left_column"].split(
            Layout(name="training_metrics"),
            Layout(name="user_profile"),
            Layout(name="recommendations")
        )

        # Split right column
        layout["right_column"].split(
            Layout(name="performance_metrics"),
            Layout(name="segment_analysis"),
            Layout(name="category_distribution")
        )

        # Update all panels with content
        layout["training_metrics"].update(
            self.create_metrics_panel("📈 Training Metrics", {
                "Final Loss": f"{results['training']['loss']:.4f}",
                "Accuracy": f"{results['training']['accuracy']:.2%}",
                "Improvement": f"{results['training']['improvement']:.1%}",
                "Training Time": f"{results['training']['time']:.2f}s"
            })
        )

        layout["user_profile"].update(
            self.create_metrics_panel("👤 User Profile", {
                "User ID": results['user']['id'],
                "Age": results['user']['age'],
                "Gender": results['user']['gender'],
                "Segment": results['user']['segment']
            })
        )

        layout["recommendations"].update(
            Panel(
                self.create_recommendation_table(results['recommendations']),
                title="[b]🎁 Top Recommendati  ons[/b]",
                border_style="cyan"
            )
        )

        layout["performance_metrics"].update(
            self.create_metrics_panel("⚡ Performance Metrics", {
                "Response Time": f"{results['performance']['response_time']:.2f}ms",
                "Memory Usage": f"{results['performance']['memory_usage']:.1f}MB",
                "Cache Hit Rate": f"{results['performance']['cache_hit_rate']:.1%}"
            })
        )

        layout["segment_analysis"].update(
            self.create_metrics_panel("🎯 Segment Ana  lysis", {
                "Price Sensitivity": f"{results['analysis']['price_sensitivity']:.2f}",
                "Category Diversity": f"{results['analysis']['category_diversity']:.2f}",
                "Engagement Score": f"{results['analysis']['engagement_score']:.2f}"
            })
        )

        category_table = Table.grid()
        category_table.add_column()
        category_table.add_column(justify="right")

        for cat, count in results['analysis']['category_distribution'].items():
            category_table.add_row(
                f"[blue]{cat}[/blue]",
                f"[yellow]{count}[/yellow]"
            )

        layout["category_distribution"].update(
            Panel(
                category_table,
                title="[b]📊 Category Distribution[/b]",
                border_style="yellow"
            )
        )

        # Print the layout
        self.console.print(layout)


def format_model_results(model_results: Dict) -> Dict:
    """Format raw model results into display format."""
    return {
        'training': {
            'loss': model_results['final_loss'],
            'accuracy': model_results['accuracy'],
            'improvement': model_results['improvement'],
            'time': model_results['training_time']
        },
        'user': {
            'id': model_results['user_id'],
            'age': model_results['user_age'],
            'gender': model_results['user_gender'],
            'segment': model_results['user_segment']
        },
        'recommendations': [
            {
                'id': rec['id'],
                'category': rec['category'],
                'price': rec['price'],
                'score': rec['score']
            }
            for rec in model_results['recommendations']
        ],
        'performance': {
            'response_time': model_results['performance']['response_time'],
            'memory_usage': model_results['performance']['memory_usage'],
            'cache_hit_rate': model_results['performance']['cache_hit_rate']
        },
        'analysis': {
            'price_sensitivity': model_results['analysis']['price_sensitivity'],
            'category_diversity': model_results['analysis']['category_diversity'],
            'engagement_score': model_results['analysis']['engagement_score'],
            'category_distribution': model_results['analysis']['category_distribution']
        }
    }        # Split         La        # Split
