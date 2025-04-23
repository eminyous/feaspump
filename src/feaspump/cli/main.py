import typer

from feaspump.cli import run

app = typer.Typer(name="feaspump", help="Feasibility Pump CLI")

app.add_typer(run.app, name="run", help="Run Feasibility Pump.")
