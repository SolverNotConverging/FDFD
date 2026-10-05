"""Validate or run direct examples without installing solver packages."""
from pathlib import Path
import ast
import importlib
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = """import runpy, sys
from unittest.mock import patch
from fdfd_common.contracts import SolverMixin, ResultMixin
from fdfd_mode_tracking import TrackedSweep
path = sys.argv[1]
sys.argv = [path]
with patch.object(SolverMixin, 'show'), patch.object(ResultMixin, 'show'), patch.object(TrackedSweep, 'show'), patch('matplotlib.pyplot.show'):
    runpy.run_path(path, run_name='__main__')
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syntax-only", action="store_true",
                        help="Validate syntax and imports without running numerical workflows.")
    args = parser.parse_args()
    examples = sorted(ROOT.glob("*/examples/**/*.py"),
                      key=lambda path: ("postprocessing" in path.parts, str(path)))
    examples = [p for p in examples if p.name != "__init__.py"]
    sys.path.insert(0, str(ROOT))
    for path in examples:
        if args.syntax_only:
            tree = ast.parse(path.read_text(), filename=str(path))
            compile(tree, str(path), 'exec')
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = importlib.import_module(node.module)
                    for name in node.names:
                        getattr(module, name.name)
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        importlib.import_module(name.name)
        else:
            subprocess.run([sys.executable, "-c", RUN, str(path)], cwd=ROOT,
                           env={**os.environ, "MPLBACKEND": "Agg"}, check=True)
    print(f"Validated {len(examples)} direct examples.")

if __name__ == "__main__":
    main()
