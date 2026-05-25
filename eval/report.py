"""
Rich terminal report for anomaly detection results.
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def print_detection_table(results: pd.DataFrame) -> None:
    """Print a rich table of detection results."""
    table = Table(
        title="Anomaly Detection Results — All Detectors x All Events",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Detector",   style="cyan",  no_wrap=True)
    table.add_column("Event",      style="white")
    table.add_column("Peak Date",  style="dim")
    table.add_column("Z-Score",    style="yellow", justify="right")
    table.add_column("Detected",   justify="center")

    for _, row in results.iterrows():
        detected_str = "[green]YES[/green]" if row["detected"] else "[red]NO[/red]"
        z = row["z_score"]
        z_str = f"{z:.2f}" if pd.notna(z) else "N/A"
        table.add_row(
            str(row["detector"]),
            str(row["event"]),
            str(row["peak_date"]),
            z_str,
            detected_str,
        )

    console.print(table)


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print detection count summary."""
    table = Table(
        title="Detection Rate by Detector",
        box=box.ROUNDED,
    )
    table.add_column("Detector",        style="cyan")
    table.add_column("Events Detected", justify="right", style="green")
    table.add_column("Total Events",    justify="right")
    table.add_column("Detection Rate",  justify="right", style="yellow")

    for detector, row in summary.iterrows():
        table.add_row(
            str(detector),
            str(int(row["detected"])),
            str(int(row["total"])),
            f"{row['detection_rate']:.0%}",
        )
    console.print(table)


def print_vix_report(report_str: str) -> None:
    console.print(f"\n[bold blue]{report_str}[/bold blue]\n")
