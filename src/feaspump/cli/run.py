from pathlib import Path
from typing import Annotated

import typer

from feaspump import FeasPump, Pump
from feaspump.cli.export import Exporter
from feaspump.cli.options import ExportMode, Options
from feaspump.mip import MIP

app = typer.Typer(
    help="Feasibility Pump CLI: classic and differentiable variants."
)


def run(mip: MIP, pump: Pump, opts: Options, exporter: Exporter) -> None:
    for seed in opts.seeds:
        opts.set_seed(seed)
        pump.run(mip)
        exporter.export(seed, pump, opts)


@app.callback()
def callback(
    ctx: typer.Context,
    file: Annotated[
        Path | None, typer.Option(help="Path to local file (if not remote)")
    ] = None,
    instance: Annotated[
        str | None, typer.Option(help="MIPLIB instance name (if remote)")
    ] = None,
    n_trials: Annotated[
        int | None, typer.Option(help="Number of trials (auto-generates seeds)")
    ] = None,
    seeds: Annotated[
        list[int] | None, typer.Option(help="List of seeds")
    ] = None,
    export_mode: Annotated[
        ExportMode, typer.Option(help="Export mode")
    ] = ExportMode.LIGHT,
    export_path: Annotated[
        Path, typer.Option(help="Path to save results")
    ] = Path("results"),
    *,
    remote: Annotated[
        bool, typer.Option("--remote", "-r", help="Use remote mode")
    ] = False,
    gpu: Annotated[bool, typer.Option(help="Use GPU if available")] = False,
) -> None:
    ctx.obj = Options(
        remote=remote,
        file=file,
        instance=instance,
        gpu=gpu,
        seeds=seeds or [],
        n_trials=n_trials,
        export_mode=export_mode,
        export_path=export_path,
    )


@app.command()
def classic(
    ctx: typer.Context,
    max_iterations: Annotated[
        int, typer.Option(help="Maximum number of iterations")
    ] = 1000,
    *,
    use_restarts: Annotated[
        bool, typer.Option(help="Use restarts in the algorithm")
    ] = True,
) -> None:
    opts = ctx.ensure_object(Options)
    opts.validate()
    mip = opts.mip
    pump = FeasPump(
        max_iterations=max_iterations,
        use_restarts=use_restarts,
    )
    exporter = Exporter(opts.export_path, opts.export_mode)
    run(mip, pump, opts, exporter)


@app.command()
def diff(
    ctx: typer.Context,
    beta: Annotated[float, typer.Option(help="Beta for regularization")] = 0.1,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1.0,
) -> None:
    opts = ctx.ensure_object(Options)
    opts.validate()
    typer.echo("🟣 Running differentiable feasibility pump")
    typer.echo(f"Beta: {beta} — LR: {lr}")
    typer.echo(f"Remote: {opts.remote}")
    typer.echo(f"File: {opts.file}")
    typer.echo(f"Instance: {opts.instance}")
    typer.echo(f"GPU: {opts.gpu}")
    typer.echo(f"Seeds: {opts.seeds}")
