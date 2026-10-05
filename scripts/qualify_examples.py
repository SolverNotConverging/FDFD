"""Import or run checkout examples without installing solver packages."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = """import runpy, sys
from unittest.mock import patch
from cem_common.contracts import SolverMixin, ResultMixin
with patch.object(SolverMixin, 'show'), patch.object(ResultMixin, 'show'), patch('matplotlib.pyplot.show'):
    runpy.run_path(sys.argv[1], run_name='__main__')
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    examples = sorted(ROOT.glob("*/examples/**/*.py"))
    examples = [p for p in examples if p.name != "__init__.py"]
    runner = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='example_import_check')"
    if not args.import_only:
        runner = RUN
    for path in examples:
        subprocess.run([sys.executable, "-c", runner, str(path)], cwd=ROOT,
                       env={**os.environ, "MPLBACKEND": "Agg", "CEM_EXAMPLE_QUALIFICATION": "1"}, check=True)
    print(f"Validated {len(examples)} checkout examples.")

if __name__ == "__main__":
    main()
