from __future__ import annotations

import logging
import sys

import uvicorn
import typer
from rich.console import Console
from rich.logging import RichHandler

from .api.app import app
from .config import get_settings

console = Console()
cli_app = typer.Typer(name="qimc", help="QA Intelli Media Comparator CLI")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)


@cli_app.command()
def serve(
    host: str = typer.Option(None, help="Override QIMC_HOST"),
    port: int = typer.Option(None, help="Override QIMC_PORT"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev only)"),
    log_level: str = typer.Option(None, help="Log level"),
) -> None:
    """Start the FastAPI microservice."""
    settings = get_settings()
    uvicorn.run(
        "qa_intelli_media_comparator.api.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_level=log_level or settings.log_level,
    )


@cli_app.command()
def analyze(
    media: str = typer.Argument(..., help="Path to media file"),
    crop: bool = typer.Option(True, help="Auto-crop preview UI"),
) -> None:
    """Run no-reference analysis on a single media file (no server needed)."""
    from pathlib import Path
    from rich.table import Table
    from .services.pipeline import ComparisonPipeline

    path = Path(media)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Analyzing:[/cyan] {path}")
    pipeline = ComparisonPipeline()
    pipeline.preload_models()
    report = pipeline.run(dut_path=path, crop_preview=crop)

    _print_report(report)


@cli_app.command()
def compare(
    dut: str = typer.Argument(..., help="DUT media file"),
    reference: str = typer.Argument(..., help="Golden reference media file"),
    crop: bool = typer.Option(True, help="Auto-crop preview UI"),
    sync: str = typer.Option("auto", help="Video sync mode: auto|frame_by_frame"),
) -> None:
    """Compare DUT against golden reference (no server needed)."""
    from pathlib import Path
    from .services.pipeline import ComparisonPipeline
    from .models.enums import SyncMode

    dut_path = Path(dut)
    ref_path = Path(reference)
    for p in (dut_path, ref_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            raise typer.Exit(1)

    console.print(f"[cyan]Comparing:[/cyan] {dut_path.name} vs {ref_path.name}")
    pipeline = ComparisonPipeline()
    pipeline.preload_models()
    report = pipeline.run(
        dut_path=dut_path,
        reference_path=ref_path,
        sync_mode=SyncMode(sync),
        crop_preview=crop,
    )
    _print_report(report)


def _print_report(report) -> None:
    from rich.table import Table
    from rich.panel import Panel

    grade_color = {"pass": "green", "warning": "yellow", "fail": "red"}.get(
        report.overall_grade.value, "white"
    )
    console.print(
        Panel(
            f"[bold {grade_color}]{report.overall_grade.value.upper()}[/bold {grade_color}]",
            title=f"Report {report.report_id}",
        )
    )

    if report.failure_reasons:
        console.print("[bold red]Failure reasons:[/bold red]")
        for reason in report.failure_reasons:
            console.print(f"  [red]•[/red] {reason}")

    t = Table(title="Quality Metrics")
    t.add_column("Metric")
    t.add_column("Value")
    qm = report.quality_metrics
    rows = [
        ("Sharpness (Laplacian)", f"{qm.blur_score:.1f}" if qm.blur_score else "N/A"),
        ("Noise Sigma", f"{qm.noise_sigma:.2f}" if qm.noise_sigma else "N/A"),
        ("Exposure Mean", f"{qm.exposure_mean:.1f} L*" if qm.exposure_mean else "N/A"),
        ("Highlight Clip%", f"{qm.highlight_clipping_pct:.2f}%" if qm.highlight_clipping_pct else "N/A"),
        ("Shadow Clip%", f"{qm.shadow_clipping_pct:.2f}%" if qm.shadow_clipping_pct else "N/A"),
        ("Saturation Mean", f"{qm.saturation_mean:.3f}" if qm.saturation_mean else "N/A"),
        ("Dynamic Range", f"{qm.dynamic_range_stops:.1f} EV" if qm.dynamic_range_stops else "N/A"),
        ("Chrom. Aberration", f"{qm.chromatic_aberration_score:.2f} px" if qm.chromatic_aberration_score else "N/A"),
    ]
    for name, val in rows:
        t.add_row(name, val)
    console.print(t)

    if report.annotated_image_path:
        console.print(f"[green]Annotated image:[/green] {report.annotated_image_path}")
    console.print(f"[dim]Processed in {report.processing_time_ms} ms[/dim]")


if __name__ == "__main__":
    cli_app()
