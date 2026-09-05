"""Run installed FEM example workflows with only interactive viewers suppressed."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = r'''
import runpy, sys
import matplotlib
matplotlib.use('Agg')
from unittest.mock import patch
from fem_common.contracts import FEMSolverMixin, ResultMixin
with patch.object(FEMSolverMixin, 'show'), patch.object(ResultMixin, 'show'), patch('matplotlib.pyplot.show'):
    runpy.run_path(sys.argv[1], run_name='__main__')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/example-qualification')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures=[]
    for example in sorted(ROOT.glob('solvers/fem/*/examples/*.py')):
        if example.name.startswith('_'): continue
        print(f'Running {example.relative_to(ROOT)}', flush=True)
        log=args.output/f'{example.parent.parent.name}-{example.stem}.log'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-c',RUN,str(example)],cwd=args.output,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:failures.append(str(example.relative_to(ROOT)))
    if failures:raise SystemExit('Failed examples: '+', '.join(failures))
    print('All FEM examples passed with their original numerical settings.')


if __name__=='__main__':main()
