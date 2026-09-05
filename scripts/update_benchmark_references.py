"""Run analytical checks, then refresh the deliberately tracked benchmark reports.

Ordinary benchmark runs write to ignored outputs/. Run this script explicitly
when updating the reviewed reference PNG/CSV files under benchmarks/.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'rectangular_waveguide_modes': ('comparison.csv', 'convergence.png'),
    'parallel_plate_electrostatics': ('comparison.csv', 'potential.csv', 'comparison.png'),
    'uniform_periodic_medium': ('comparison.csv', 'comparison.png'),
    'coaxial_waveguide_adaptivity': ('comparison.csv', 'adaptive_history.csv', 'convergence.png', 'meshes.png'),
}


def main():
    commands = []
    for case in CASES:
        command = ['python', f'benchmarks/analytical/{case}.py', '--check']
        commands.append(' '.join(command))
        subprocess.run([sys.executable, *command[1:]], cwd=ROOT,
                       env={**os.environ, 'MPLBACKEND': 'Agg'}, check=True)
    # Refresh only after every benchmark has passed its analytical checks.
    destination = ROOT / 'benchmarks/reference_results'
    artifacts = {}
    for case, filenames in CASES.items():
        target = destination / case
        target.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source = ROOT / 'outputs/benchmarks/analytical' / case / filename
            shutil.copyfile(source, target / filename)
            artifacts[f'{case}/{filename}'] = hashlib.sha256(source.read_bytes()).hexdigest()
    source_hash = hashlib.sha256()
    sources = list((ROOT / 'benchmarks/analytical').glob('*.py'))
    for parent in ('solvers', 'libraries'):
        sources.extend(p for p in (ROOT / parent).rglob('*')
                       if 'src' in p.parts and p.suffix in ('.py', '.cpp', '.h', '.hpp'))
    for source in sorted(sources):
        source_hash.update(source.relative_to(ROOT).as_posix().encode() + b'\0' + source.read_bytes() + b'\0')
    git = ['git', '-c', f'safe.directory={ROOT.as_posix()}']
    revision = subprocess.check_output([*git, 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output([*git, 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    manifest = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        base_git_revision=revision, working_tree_modified=dirty,
        source_sha256=source_hash.hexdigest(),
        source_hash_definition='Sorted benchmark .py and solver/library src .py/.cpp/.h/.hpp; relative POSIX path, NUL, bytes, NUL.',
        python=platform.python_version(), platform=platform.platform(),
        dependencies={name: version(name) for name in (
            'numpy', 'scipy', 'matplotlib', 'scikit-fem', 'gmsh', 'h5py')},
        commands=commands, checks='All four commands exited successfully with --check.',
        artifact_sha256=artifacts,
    )
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Refreshed checked references: {destination}')


if __name__ == '__main__':
    main()
