"""Run every installed solver example with interactive viewers suppressed."""
import argparse
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
IMPORT = r'''
import pathlib, runpy, sys
import matplotlib
matplotlib.use('Agg')
paths = sorted(pathlib.Path(sys.argv[1]).rglob('*.py'))
if not paths:
    raise SystemExit('No example scripts found.')
for path in paths:
    runpy.run_path(str(path), run_name='example_import_check')
print(f'Imported all {len(paths)} example scripts with installed packages.', flush=True)
'''
RUN = r'''
import runpy, sys
import matplotlib
matplotlib.use('Agg')
from unittest.mock import patch
from cem_common.contracts import SolverMixin, ResultMixin
with patch.object(SolverMixin, 'show'), patch.object(ResultMixin, 'show'), patch('matplotlib.pyplot.show'):
    runpy.run_path(sys.argv[1], run_name='__main__')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/example-qualification')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-I', '-c', IMPORT, str(ROOT/'examples')],
                   cwd=args.output, check=True)
    failures=[]
    examples = sorted(ROOT.glob('examples/*/*/*.py'))
    if not examples:
        raise SystemExit('No solver examples found under examples/.')
    environment = os.environ.copy()
    environment['CEM_EXAMPLE_QUALIFICATION'] = '1'
    for example in examples:
        if example.name.startswith('_'): continue
        print(f'Running {example.relative_to(ROOT)}', flush=True)
        relative = example.relative_to(ROOT/'examples')
        log=args.output/f'{relative.parts[0]}-{relative.parts[1]}-{example.stem}.log'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-I','-c',RUN,str(example)],cwd=args.output,
                                  stdout=stream,stderr=subprocess.STDOUT,env=environment)
        if result.returncode:failures.append(str(example.relative_to(ROOT)))
    if failures:raise SystemExit('Failed examples: '+', '.join(failures))
    print(f'All {len(examples)} solver examples passed with their original numerical settings.')


if __name__=='__main__':main()
