from pathlib import Path
from typing import Annotated

import typer

from feaspump import DiffPump, FeasPump, Pump
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
    flip_temp: Annotated[
        float, typer.Option(help="Temperature for flipping")
    ] = 50.0,
    flip_temp_low: Annotated[
        float, typer.Option(help="Low temperature for flipping")
    ] = 0.5,
    flip_temp_up: Annotated[
        float, typer.Option(help="High temperature for flipping")
    ] = 1.5,
    perturb_freq: Annotated[
        int, typer.Option(help="Frequency of perturbation")
    ] = 100,
    perturb_rho: Annotated[
        float, typer.Option(help="Rho for perturbation")
    ] = -0.3,
    history_length: Annotated[int, typer.Option(help="Length of history")] = 2,
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
        flip_temp=flip_temp,
        flip_temp_low=flip_temp_low,
        flip_temp_up=flip_temp_up,
        perturb_freq=perturb_freq,
        perturb_rho=perturb_rho,
        history_length=history_length,
    )
    exporter = Exporter(opts.export_path, opts.export_mode)
    run(mip, pump, opts, exporter)
    mip.close()
    opts.close()


@app.command()
def diff(
    ctx: typer.Context,
    max_iterations: Annotated[
        int, typer.Option(help="Maximum number of iterations")
    ] = 1000,
    flip_temp: Annotated[
        float, typer.Option(help="Temperature for flipping")
    ] = 50.0,
    flip_temp_low: Annotated[
        float, typer.Option(help="Low temperature for flipping")
    ] = 0.5,
    flip_temp_up: Annotated[
        float, typer.Option(help="High temperature for flipping")
    ] = 1.5,
    perturb_freq: Annotated[
        int, typer.Option(help="Frequency of perturbation")
    ] = 100,
    perturb_rho: Annotated[
        float, typer.Option(help="Rho for perturbation")
    ] = -0.3,
    history_length: Annotated[int, typer.Option(help="Length of history")] = 2,
    *,
    use_restarts: Annotated[
        bool, typer.Option(help="Use restarts in the algorithm")
    ] = True,
) -> None:
    opts = ctx.ensure_object(Options)
    opts.validate()
    pump = DiffPump(
        max_iterations=max_iterations,
        use_restarts=use_restarts,
        flip_temp=flip_temp,
        flip_temp_low=flip_temp_low,
        flip_temp_up=flip_temp_up,
        perturb_freq=perturb_freq,
        perturb_rho=perturb_rho,
        history_length=history_length,
    )
    mip = opts.mip
    exporter = Exporter(opts.export_path, opts.export_mode)
    run(mip, pump, opts, exporter)
    mip.close()
    opts.close()
