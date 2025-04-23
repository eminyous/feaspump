from datetime import UTC, datetime
from pathlib import Path

import yaml

from feaspump import Pump
from feaspump.cli.options import ExportMode, Options


class Exporter:
    root: Path
    mode: ExportMode
    yaml_dir: Path
    yaml_dirname: str = "yaml"

    def __init__(self, root: Path, mode: ExportMode) -> None:
        self.root = root
        self.mode = mode
        self.root.mkdir(parents=True, exist_ok=True)
        self.yaml_dir = self.root / self.yaml_dirname
        self.yaml_dir.mkdir(parents=True, exist_ok=True)

    def _get_yaml_file(self) -> Path:
        now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        yaml_filename = f"{now}.yaml"
        yaml_file = self.yaml_dir / yaml_filename
        yaml_files = sorted(self.yaml_dir.glob("*.yaml"))
        if yaml_files and yaml_files[-1].stat().st_size <= 1024 * 1024 * 10:
            return yaml_files[-1]
        yaml_file.touch()
        return yaml_file

    def export(self, seed: int, pump: Pump, opts: Options) -> None:
        yaml_file = self._get_yaml_file()
        info = self.to_yaml(seed, pump, opts)
        with yaml_file.open("a") as stream:
            yaml.safe_dump([info], stream)

    def to_yaml(self, seed: int, pump: Pump, opts: Options) -> dict:
        now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        return {
            "timestamp": now,
            "mode": str(self.mode),
            "seed": seed,
            "status": str(pump.status),
            "iteration": pump.iteration,
            "max_iterations": pump.max_iterations,
            "options": str(opts.path),
        }
