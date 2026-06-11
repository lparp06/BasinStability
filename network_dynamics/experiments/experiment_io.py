"""
Output helpers for experiment scripts.

Experiment modules are intentionally print-heavy because their output is a
human-readable run log. This helper redirects that log to project-root
``output.txt`` without changing each individual print call.
"""

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parents[2] / "output.txt"


def run_with_output_file(function, output_path=OUTPUT_PATH):
    """
    Run ``function`` while writing stdout and stderr to ``output.txt``.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        with redirect_stdout(output_file), redirect_stderr(output_file):
            return function()
