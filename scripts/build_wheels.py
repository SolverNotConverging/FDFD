"""Build 1.0 Python distributions with the active interpreter."""
from pathlib import Path
import argparse
import subprocess
import sys
import zipfile
import shutil
from tempfile import TemporaryDirectory

from install_python import PACKAGES, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "dist")
    parser.add_argument("--no-build-isolation", action="store_true", help="Use preinstalled build dependencies.")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cem-release-build-") as temporary:
        staging=Path(temporary).resolve()
        sources=[]
        for package in PACKAGES:
            original=ROOT/package
            target=staging/package
            target.mkdir(parents=True)
            shutil.copytree(original/'src',target/'src',ignore=shutil.ignore_patterns(
                '__pycache__','*.egg-info','*.pyc','*.pyd','*.so','_cython_kernels.c'))
            for name in ('pyproject.toml','README.rst','LICENSE'):
                if (original/name).is_file():shutil.copy2(original/name,target/name)
            sources.append(target)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", *( ["--no-build-isolation"] if args.no_build_isolation else []), "--wheel-dir", str(args.output),
                        *map(str,sources)], check=True)
    for package in PACKAGES:
        names={str(path.relative_to(ROOT/package/'src')).replace('\\','/')
               for path in (ROOT/package/'src').rglob('*.py')}
        distribution=next(iter(names)).split('/')[0]
        for wheel in args.output.glob(distribution+'-1.0.0-*.whl'):
            with zipfile.ZipFile(wheel) as archive:
                packaged={name for name in archive.namelist() if name.endswith('.py') and name.startswith(distribution+'/')}
            if packaged!=names:
                raise SystemExit(f'{wheel.name}: packaged Python sources differ from the maintained source tree.')
    native = list(args.output.glob("periodic_eigensolver-1.0.0-*.whl"))
    if len(native) != 1:
        raise SystemExit("Expected one periodic eigensolver 1.0.0 wheel; use a fresh output directory.")
    with zipfile.ZipFile(native[0]) as archive:
        if not any(name.endswith((".pyd", ".so")) for name in archive.namelist()):
            raise SystemExit("Release qualification failed: periodic eigensolver extension is missing.")


if __name__ == "__main__":
    main()
