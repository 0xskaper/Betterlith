from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
import time
from typing import List, Dict, Any


class CLIPresenter:
    def __init__(self):
        self.console = Console()

    def print_header(self):
        """Print beautiful header for the application."""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold yellow]Monolith Recommendation System[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        ))
        self.console.print("\n")

    def create_training_progress(self) -> Progress:
        """Create a progress bar for training."""
        return Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        )

    def print_epoch_results(self, epoch: int, total_epochs: int, loss: float):
        """Print epoch results in a nice format."""
        self.console.print(
            f"Epoch {epoch}/{total_epochs} │ ",
            end=""
        )
        # Color the loss based on value
        if loss < 0.3:
            color = "green"
        elif loss < 0.5:
            color = "yellow"
        else:
            color = "red"
        self.console.print(f"Loss: [{color}]{loss:.4f}[/{color}]")

    def print_user_profile(self, user_id: int, age: int, gender: str):
        """Print user profile information."""
        profile = Table.grid(padding=1)
        profile.add_column(style="bold cyan", justify="right")
        profile.add_column(style="yellow")

        profile.add_row("User ID:", str(user_id))
        profile.add_row("Age:", str(age))
        profile.add_row("Gender:", gender)

        self.console.print(Panel(
            profile,
            title="[bold]User Profile[/bold]",
            border_style="cyan",
            padding=(1, 2)
        ))

    def print_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Print recommendations in a beautiful table."""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            box=box.DOUBLE_EDGE
        )

        table.add_column("Item ID", justify="center", style="cyan")
        table.add_column("Category", justify="center")
        table.add_column("Price ($)", justify="right")
        table.add_column("Score (%)", justify="right")

        for rec in recommendations:
            score_color = "green" if rec["score"] > 0.8 else "yellow" if rec["score"] > 0.5 else "red"
            table.add_row(
                str(rec["id"]),
                f"Category {rec['category']}",
                f"{rec['price']:.2f}",
                f"[{score_color}]{rec['score']*100:.1f}[/{score_color}]"
            )

        self.console.print(Panel(
            table,
            title="[bold]Top Recommendations[/bold]",
            border_style="magenta",
            padding=(1, 2)
        ))

    def print_model_stats(self, stats: Dict[str, Any]):
        """Print model statistics."""
        stats_table = Table.grid(padding=1)
        stats_table.add_column(style="bold green", justify="right")
        stats_table.add_column(style="yellow")

        stats_table.add_row("Total Parameters:", f"{stats['total_params']:,}")
        stats_table.add_row("Training Time:", f"{stats['train_time']:.2f}s")
        stats_table.add_row("Final Loss:", f"{stats['final_loss']:.4f}")

        self.console.print(Panel(
            stats_table,
            title="[bold]Model Statistics[/bold]",
            border_style="green",
            padding=(1, 2)
        ))

    def print_feature_info(self, feature_counts: Dict[str, int]):
        """Print feature information."""
        table = Table(
            show_header=True,
            header_style="bold blue",
            border_style="blue",
            box=box.ROUNDED
        )

        table.add_column("Feature Type", justify="left")
        table.add_column("Count", justify="right")

        for feature_type, count in feature_counts.items():
            table.add_row(feature_type, str(count))

        self.console.print(Panel(
            table,
            title="[bold]Feature Information[/bold]",
            border_style="blue",
            padding=(1, 2)
        ))


def print_training_summary(num_epochs: int, avg_loss: float, elapsed_time: float):
    """Print training summary with rich formatting."""
    console = Console()
    console.print("\n")
    console.print(Panel(
        f"Training completed in [yellow]{elapsed_time:.2f}s[/yellow]\n"
        f"Final average loss: [green]{avg_loss:.4f}[/green]\n"
        f"Total epochs: [blue]{num_epochs}[/blue]",
        title="[bold]Training Summary[/bold]",
        border_style="green"
    ))
    console.print("\n")
