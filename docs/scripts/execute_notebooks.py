"""Execute the example notebooks in parallel and write outputs in place.

The mkdocs build is configured with ``execute: false`` so that mkdocs-jupyter
only renders pre-executed notebooks. This script is the producer step: it
discovers ``examples/**/*.ipynb``, runs each in its own kernel via ``nbclient``,
and writes the executed notebook back in place. ``docs/hooks.py`` then stages
``examples/`` into ``docs/examples/`` at build time, so the outputs produced here
are what gets rendered.

Running execution as a dedicated, parallel pre-build step (see
``.readthedocs.yaml``) is faster than mkdocs-jupyter's sequential execution, and
a notebook failure surfaces directly instead of being buried in MkDocs logging.

Each notebook gets its own kernel subprocess, so a ThreadPoolExecutor is
sufficient — the GIL is not the bottleneck. Notebooks run in their own
directory (``cwd = notebook.parent``) so relative paths inside cells behave
the same as in Jupyter Lab.

Some notebooks are excluded by default (``DEFAULT_EXCLUDED``): they need CDS or
Earth Data Hub credentials, a large ERA5 download, or a purchased
thewindpower.net dataset, and one is a recipe whose paths are placeholders — so
they cannot run in CI. They are rendered from whatever is committed, i.e. mostly
as source without outputs, which is how the previous Jupyter Book build treated
them (``_config.yml``'s ``execute.exclude_patterns``).

Usage:
    python docs/scripts/execute_notebooks.py                  # default: CPU-1 workers
    python docs/scripts/execute_notebooks.py --workers 4
    python docs/scripts/execute_notebooks.py --exclude 3_1_wind_workflows_overview.ipynb
    python docs/scripts/execute_notebooks.py --timeout 1200
    python docs/scripts/execute_notebooks.py --list

Exit code is non-zero if any notebook fails. Cell errors abort that notebook
but do not stop the rest of the batch — the summary at the end lists failures.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Notebooks that cannot be executed in CI, by filename:
#   1_1_1  downloads ERA5 from the CDS (needs a ~/.cdsapirc API key)
#   1_1_2  operates on the full ERA5 wind-vector download from 1_1_1
#   1_1_3  calls rk.download_and_process() -> CDS download (needs ~/.cdsapirc)
#   1_1_4  same as 1_1_3, for the ERA5-based solar workflows
#   1_3_1  processes power curves from a purchased thewindpower.net dataset
#   1_4_1  a recipe notebook: its paths are placeholders ("path_to_ERA5_data",
#          "some_path"), so it is not runnable anywhere, CI or otherwise
#   3_8    reads a remote Earth Data Hub Zarr store (needs ~/.netrc). Its
#          network cells are marked "# NBVAL_SKIP", which the nbval test suite
#          honours but nbclient does not — hence the exclusion here.
#
# This is the same set the GitHub Actions workflows leave out of
# `examples_to_execute`, plus 3_8, which they run only because nbval skips its
# network cells. The excluded notebooks are rendered from whatever is committed.
DEFAULT_EXCLUDED = [
    "1_1_1_how_to_download_era5_data.ipynb",
    "1_1_2_wind_speed_from_vectors_in_era5.ipynb",
    "1_1_3_prepare_era5_for_wind_workflow.ipynb",
    "1_1_4_prepare_era5_for_solar_workflow.ipynb",
    "1_3_1_process_power_curves_from_thewindpower_net.ipynb",
    "1_4_1_how_to_create_LRA_datasets.ipynb",
    "3_8_use_workflows_with_zarr.ipynb",
]


@dataclass
class Result:
    path: Path
    elapsed: float
    error: str | None


def execute_one(path: Path, timeout: int) -> Result:
    start = time.perf_counter()
    try:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        nbformat.write(nb, path)
        return Result(path, time.perf_counter() - start, None)
    except CellExecutionError as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else "CellExecutionError"
        return Result(path, time.perf_counter() - start, first_line)
    except Exception as exc:
        return Result(path, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path("examples"),
        help="Directory searched recursively for notebooks (default: examples)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        help="Per-cell timeout in seconds (default: 1000)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="FILENAME",
        help=(
            "Notebook filename to skip, in addition to the defaults "
            f"({', '.join(DEFAULT_EXCLUDED)}). Repeat for multiple."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the notebooks that would be executed and exit",
    )
    args = parser.parse_args(argv)
    # argparse's "append" action extends the default list in place, so build a
    # fresh list rather than mutating DEFAULT_EXCLUDED.
    args.exclude = DEFAULT_EXCLUDED + list(args.exclude or [])
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.notebooks_dir.is_dir():
        print(f"error: {args.notebooks_dir} is not a directory", file=sys.stderr)
        return 2

    notebooks = sorted(
        p
        for p in args.notebooks_dir.rglob("*.ipynb")
        if p.name not in args.exclude and ".ipynb_checkpoints" not in p.parts
    )
    if not notebooks:
        print(f"No notebooks to execute in {args.notebooks_dir}", file=sys.stderr)
        return 1

    if args.list:
        for nb in notebooks:
            print(nb.as_posix())
        return 0

    print(
        f"Executing {len(notebooks)} notebooks with {args.workers} worker(s), per-cell timeout {args.timeout}s",
        flush=True,
    )

    overall_start = time.perf_counter()
    results: list[Result] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_one, nb, args.timeout): nb for nb in notebooks}
        for fut in as_completed(futures):
            r = fut.result()
            tag = "FAIL" if r.error else "OK  "
            print(f"  [{tag}] {r.path.name:<50} {r.elapsed:6.1f}s", flush=True)
            results.append(r)

    total = time.perf_counter() - overall_start
    failures = [r for r in results if r.error]
    print(f"\nTotal: {total:.1f}s ({len(results) - len(failures)}/{len(results)} succeeded)")
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r.path}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
