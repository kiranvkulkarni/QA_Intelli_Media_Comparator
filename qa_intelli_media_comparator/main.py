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
    mode: str = typer.Option(
        "quality",
        help=(
            "Analysis depth: 'functional' (fast, ~50ms — is the camera working? "
            "no neural IQA) or 'quality' (full IQA pipeline, default)."
        ),
    ),
) -> None:
    """Run no-reference analysis on a single media file (no server needed)."""
    from pathlib import Path
    from rich.table import Table
    from .services.pipeline import ComparisonPipeline

    path = Path(media)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Analyzing ({mode} mode):[/cyan] {path}")
    pipeline = ComparisonPipeline()
    if mode == "quality":
        pipeline.preload_models()
    report = pipeline.run(dut_path=path, crop_preview=crop, analysis_mode=mode)

    _print_report(report)


@cli_app.command()
def compare(
    dut: str = typer.Argument(..., help="DUT media file"),
    reference: str = typer.Argument(..., help="Golden reference media file"),
    crop: bool = typer.Option(True, help="Auto-crop preview UI"),
    sync: str = typer.Option("auto", help="Video sync mode: auto|frame_by_frame"),
    mode: str = typer.Option(
        "quality",
        help=(
            "Analysis depth: 'functional' (fast, ~50ms — is the camera working? "
            "no neural IQA) or 'quality' (full IQA pipeline, default)."
        ),
    ),
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

    console.print(f"[cyan]Comparing ({mode} mode):[/cyan] {dut_path.name} vs {ref_path.name}")
    pipeline = ComparisonPipeline()
    if mode == "quality":
        pipeline.preload_models()
    report = pipeline.run(
        dut_path=dut_path,
        reference_path=ref_path,
        sync_mode=SyncMode(sync),
        crop_preview=crop,
        analysis_mode=mode,
    )
    _print_report(report)


def _print_report(report) -> None:
    from rich.table import Table
    from rich.panel import Panel

    _gc = {"pass": "green", "warning": "yellow", "fail": "red"}
    grade_color    = _gc.get(report.overall_grade.value, "white")
    func_color     = _gc.get(report.functional_grade.value, "white")
    mode_label     = f"  [{report.analysis_mode} mode]"

    console.print(
        Panel(
            f"[bold {func_color}]FUNCTIONAL: {report.functional_grade.value.upper()}[/bold {func_color}]"
            + (f"\n[bold {grade_color}]QUALITY:    {report.overall_grade.value.upper()}[/bold {grade_color}]"
               if report.analysis_mode == "quality" else ""),
            title=f"Report {report.report_id}{mode_label}",
        )
    )

    if report.functional_reasons:
        console.print("[bold red]Functional issues:[/bold red]")
        for reason in report.functional_reasons:
            console.print(f"  [red]•[/red] {reason}")

    if report.failure_reasons:
        console.print("[bold red]Quality failure reasons:[/bold red]")
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

    # DUT vs Reference quality comparison (compare mode only)
    qc = getattr(report, "quality_comparison", None)
    if qc is not None:
        from rich.table import Table as RichTable
        ct = RichTable(title="DUT vs Reference — Quality Comparison")
        ct.add_column("Metric", style="dim")
        ct.add_column("DUT", justify="right")
        ct.add_column("REF", justify="right")
        ct.add_column("Delta", justify="right")
        ct.add_column("Δ%", justify="right")
        ct.add_column("Status", justify="center")

        _rows = [
            ("Sharpness",          qc.sharpness,           True),
            ("Noise σ",            qc.noise,               False),
            ("Exposure (L*)",      qc.exposure,            None),
            ("Highlight clip%",    qc.highlight_clipping,  False),
            ("Shadow clip%",       qc.shadow_clipping,     False),
            ("WB Deviation",       qc.white_balance,       False),
            ("Chrom. Aberr. (px)", qc.chromatic_aberration, False),
        ]
        for label, m, higher_better in _rows:
            if m is None:
                continue
            sign = "+" if m.delta >= 0 else ""
            pct_str = f"{sign}{m.delta_pct:.1f}%" if m.delta_pct is not None else "—"
            if m.regression:
                status = "[bold red]REGRESSION[/bold red]"
            elif higher_better is None:
                status = "[dim]—[/dim]"
            elif (higher_better and m.delta >= 0) or (not higher_better and m.delta <= 0):
                status = "[green]OK[/green]"
            else:
                status = "[yellow]WORSE[/yellow]"
            ct.add_row(label, f"{m.dut:.3f}", f"{m.ref:.3f}",
                       f"{sign}{m.delta:.3f}", pct_str, status)
        console.print(ct)

    if report.annotated_image_path:
        console.print(f"[green]Annotated image:[/green] {report.annotated_image_path}")
    console.print(f"[dim]Processed in {report.processing_time_ms} ms[/dim]")


if __name__ == "__main__":
    cli_app()
