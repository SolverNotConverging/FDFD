"""Stage and qualify native runtimes for the complete FDFD Windows wheel.

Run ``--phase stage`` after building/testing the root CMake project with VTK ON.
Then run ``--phase finish`` (network required) to preserve exact dependency source
packages and record their public source URLs. Neither phase publishes artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
BUNDLE_NAME = f"FDFD-{VERSION}-windows-x64"
APPS = {
    "fem_waveguide_scattering_viewer": (
        "fem-waveguide-scattering-viewer", "fem-waveguide-scattering-viewer-inspect"),
    "fem_periodic_mode_viewer": (
        "fem-periodic-mode-viewer", "fem-periodic-mode-inspect"),
    "transmission_line_calculator": (
        "transmission-line-calculator", "transmission-line-calculator-cli"),
}


def run(*command, **options):
    return subprocess.run([str(arg) for arg in command], check=True, text=True,
                          capture_output=True, **options).stdout


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_database(msys):
    packages, owners = {}, {}
    for directory in (msys / "var/lib/pacman/local").iterdir():
        if not (directory / "desc").is_file():
            continue
        fields = {}
        for match in re.finditer(r"%([^%]+)%\n(.*?)(?:\n\n|\Z)",
                                 (directory / "desc").read_text(), re.S):
            fields[match[1]] = match[2].strip()
        name = fields["NAME"]
        file_record = (directory / "files").read_text(encoding="utf-8")
        files = (file_record.split("%FILES%\n", 1)[1].split("\n\n", 1)[0].splitlines()
                 if "%FILES%\n" in file_record else [])
        packages[name] = {"name": name, "version": fields["VERSION"],
                          "base": fields["BASE"], "license": fields.get("LICENSE", ""),
                          "homepage": fields.get("URL", ""), "files": files}
        for relative in files:
            if not relative.endswith("/"):
                owners[relative.lower()] = name
    return packages, owners


def clean_environment(bin_dir):
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("QT_", "QML", "PYTHON", "CONDA", "VIRTUAL_ENV"))}
    windows = Path(os.environ["SystemRoot"])
    environment["PATH"] = os.pathsep.join(map(str, (bin_dir, windows / "System32", windows)))
    environment["QT_QPA_PLATFORM"] = "offscreen"
    return environment


def qualify(bundle):
    bin_dir = bundle / "bin"
    environment = clean_environment(bin_dir)
    samples = bundle / "samples"
    cases = [
        ("transmission-line-calculator", "--smoke-test"),
        ("transmission-line-calculator", "--calculate-smoke-test"),
        ("transmission-line-calculator-cli", "--smoke-test"),
        ("transmission-line-calculator-cli", "--version"),
        ("fem-periodic-mode-inspect", str(samples / "periodic-2d.h5"), "0", "0", "--coefficients"),
        ("fem-periodic-mode-inspect", str(samples / "periodic-3d.h5"), "0", "0", "--coefficients"),
        ("fem-periodic-mode-inspect", str(samples / "periodic-sweep.h5"), "1", "0", "--coefficients"),
        ("fem-periodic-mode-viewer", "--smoke-test", str(samples / "periodic-2d.h5")),
        ("fem-periodic-mode-viewer", "--smoke-test", str(samples / "periodic-3d.h5")),
        ("fem-periodic-mode-viewer", "--smoke-test-slice", str(samples / "periodic-3d.h5")),
        ("fem-periodic-mode-viewer", "--smoke-test", str(samples / "periodic-sweep.h5")),
        ("fem-waveguide-scattering-viewer-inspect", str(samples / "scattering.h5")),
        ("fem-waveguide-scattering-viewer-inspect", str(samples / "scattering-sweep.h5"), "1"),
        ("fem-waveguide-scattering-viewer", "--smoke-test", str(samples / "scattering.h5")),
        ("fem-waveguide-scattering-viewer", "--smoke-test", str(samples / "scattering-sweep.h5")),
    ]
    log = []
    for executable, *arguments in cases:
        output = run(bin_dir / (executable + ".exe"), *arguments, cwd=bundle,
                     env=environment, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
        log.append({"executable": executable, "arguments": arguments, "stdout": output})
        print(f"PASS {executable} {' '.join(arguments[:1])}", flush=True)
    return log


def stage(args):
    bundle = args.output / BUNDLE_NAME
    if bundle.exists():
        raise RuntimeError(f"Use a fresh output directory; staging already exists: {bundle}")
    bin_dir = bundle / "bin"
    bin_dir.mkdir(parents=True)
    msys = args.msys_prefix.parent
    mingw_bin = args.msys_prefix / "bin"
    packages, owners = package_database(msys)
    used = set()
    origins = {}

    def record(source, target):
        relative = source.relative_to(msys).as_posix().lower()
        owner = owners.get(relative)
        if not owner:
            raise RuntimeError(f"No package provenance for {source}")
        used.add(owner)
        origins[target.relative_to(bundle).as_posix()] = owner

    for directory, executables in APPS.items():
        for executable in executables:
            source = args.build / "apps" / directory / (executable + ".exe")
            shutil.copy2(source, bin_dir / source.name)

    environment = dict(os.environ, PATH=str(mingw_bin) + os.pathsep + os.environ["PATH"])
    for executables in APPS.values():
        run(mingw_bin / "windeployqt.exe", "--release", "--no-translations",
            "--no-system-d3d-compiler", "--no-patchqt", bin_dir / (executables[0] + ".exe"),
            env=environment, timeout=120)
    plugins = args.msys_prefix / "share/qt6/plugins"
    shutil.copy2(plugins / "platforms/qoffscreen.dll", bin_dir / "platforms/qoffscreen.dll")
    (bin_dir / "qt.conf").write_text("[Paths]\nPrefix=.\nPlugins=.\n", encoding="utf-8")

    # Record Qt's deployment, including plugins; then resolve PE imports recursively.
    for target in bin_dir.rglob("*.dll"):
        relative = target.relative_to(bin_dir)
        source = mingw_bin / target.name if len(relative.parts) == 1 else plugins / relative
        if not source.is_file():
            raise RuntimeError(f"Unrecognized deployed runtime: {target}")
        record(source, target)
    pending = list(bin_dir.rglob("*.exe")) + list(bin_dir.rglob("*.dll"))
    checked = set()
    while pending:
        target = pending.pop()
        if target in checked:
            continue
        checked.add(target)
        imports = re.findall(r"DLL Name:\s*(\S+)", run(mingw_bin / "objdump.exe", "-p", target))
        for name in imports:
            source = mingw_bin / name
            destination = bin_dir / name
            if source.is_file():
                if not destination.exists():
                    shutil.copy2(source, destination)
                    pending.append(destination)
                record(source, destination)
            elif not (Path(os.environ["SystemRoot"]) / "System32" / name).is_file() and not name.lower().startswith(("api-ms-", "ext-ms-")):
                raise RuntimeError(f"Unresolved dependency {name} imported by {target}")

    # Eigen's compiled-in headers and compiler runtime sources also belong in provenance.
    used.add("mingw-w64-x86_64-eigen3")
    source_packages = []
    for name in sorted(used):
        package = packages[name]
        version = package["version"].replace(":", "~")
        archive = f"{package['base']}-{version}.src.tar.zst"
        info = {key: value for key, value in package.items() if key != "files"}
        info["source_archive"] = archive
        info["source_url"] = f"https://repo.msys2.org/mingw/sources/{archive}"
        source_packages.append(info)
        for relative in package["files"]:
            if "/share/licenses/" in relative and not relative.endswith("/"):
                source = msys / relative
                dest = bundle / "licenses" / name / relative.split("/share/licenses/", 1)[1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
    # Some MSYS2 packages refer to standard SPDX licenses without installing a copy.
    common = bundle / "licenses/common"
    common.mkdir(parents=True)
    for source in (args.msys_prefix / "share/licenses/qt6-base").glob("*.txt"):
        shutil.copy2(source, common / source.name)
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE-FDFD.txt")
    (bundle / "samples").mkdir()
    for name in ("periodic-2d.h5", "periodic-3d.h5", "periodic-sweep.h5", "scattering.h5", "scattering-sweep.h5"):
        shutil.copy2(args.samples / name, bundle / "samples" / name)

    revision = run("git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD", cwd=ROOT).strip()
    manifest = {"project": "FDFD", "version": VERSION, "architecture": "x86_64",
                "toolchain": "MSYS2 MinGW64", "git_base_revision": revision,
                "source_note": "Native application sources are in the FDFD repository; dependency sources are indexed in SOURCE_INDEX.md.",
                "packages": source_packages, "runtime_owners": origins}
    (bundle / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_bundle_readme(bundle)
    log = qualify(bundle)
    (args.output / "qualification.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"Staged {len(origins)} runtime files from {len(used)} packages at {bundle}")


def finish(args):
    bundle = args.output / BUNDLE_NAME
    manifest_path = bundle / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = args.output / "dependency-sources"
    sources.mkdir(exist_ok=True)
    unique = {item["source_archive"]: item for item in manifest["packages"]}

    def download(item):
        target = sources / item["source_archive"]
        if not target.is_file():
            temporary = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(item["source_url"], headers={"User-Agent": "FDFD-release-packager/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(target)
        print(f"Source {target.name} ({target.stat().st_size // 1024**2} MiB)", flush=True)
        return target.name, digest(target)

    with ThreadPoolExecutor(max_workers=4) as executor:
        hashes = dict(executor.map(download, unique.values()))
    for item in manifest["packages"]:
        item["source_sha256"] = hashes[item["source_archive"]]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    index = ["# Native dependency sources for FDFD 1.0.0", "",
             "The complete Windows wheel contains native applications and dynamically linked",
             "MSYS2 libraries. These exact source archives include upstream sources, patches,",
             "and MSYS2 build recipes. Links and SHA256 hashes were checked during packaging.", "",
             "Application source and packaging code: [FDFD repository](https://github.com/SolverNotConverging/FDFD).",
             "The release notes identify the source commit for the complete wheel.", "",
             "To rebuild a dependency, extract its archive in MSYS2, enter the directory",
             "containing PKGBUILD, and run `MINGW_ARCH=mingw64 makepkg-mingw -sCLf`.",
             "Build FDFD with CMake, the MinGW64 prefix, Release, and VTK enabled.", ""]
    for name, item in sorted(unique.items()):
        index.extend([f"- [{name}]({item['source_url']})", f"  SHA256: `{hashes[name]}`"])
    (bundle / "SOURCE_INDEX.md").write_text("\n".join(index) + "\n")
    shutil.copy2(bundle / "SOURCE_INDEX.md", sources / "SOURCE_INDEX.md")
    write_bundle_readme(bundle)
    log = qualify(bundle)
    (args.output / "qualification.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"Qualified native runtime ready for the single FDFD wheel: {bundle}")


def write_bundle_readme(bundle):
    (bundle / "README.txt").write_text("""FDFD 1.0.0 - bundled Windows x64 native applications

These runtime files are installed by the complete FDFD wheel. Launch the apps:
  python -m fdfd calculator
  python -m fdfd periodic-viewer
  python -m fdfd scattering-viewer
  python -m fdfd calculator-cli

The .exe files in bin can also run directly. Keep the DLLs, qt.conf, and plugin
subdirectories together. The samples directory contains example HDF5 results.
Python result.show() finds these viewers automatically. No compiler or separate
native-app installation is required. The 3D viewport needs an OpenGL driver.

Licenses and source:
FDFD's original source is MIT licensed (LICENSE-FDFD.txt). Bundled libraries retain
their own licenses; see licenses/ and build-manifest.json. The calculator combined
with Gmsh is distributed under GPL-3.0-or-later; the GPL version 3 terms are in
licenses/common/GPL-3.0-only.txt. Qt is dynamically linked under LGPL-3.0.
Users may replace compatible library binaries and debug those modifications.
No additional restrictions are imposed. This software comes without warranty.

SOURCE_INDEX.md provides exact dependency source/build-recipe downloads and hashes.
The same index is published beside the wheel via its GitHub release notes:
https://github.com/SolverNotConverging/FDFD/blob/main/doc/development/native_dependency_sources.md
Application source and packaging scripts are in the FDFD repository; the release
notes identify the corresponding source commit. Only the wheel is needed to run.
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("stage", "finish"))
    parser.add_argument("--build", type=Path, default=ROOT / "outputs/build")
    parser.add_argument("--samples", type=Path, default=ROOT / "outputs/native-qualification")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/native-release-1.0.0")
    parser.add_argument("--msys-prefix", type=Path, default=Path("C:/msys64/mingw64"))
    args = parser.parse_args()
    for name in ("build", "samples", "output", "msys_prefix"):
        setattr(args, name, getattr(args, name).resolve())
    if os.name != "nt":
        parser.error("This packager requires Windows and the MSYS2 MinGW64 toolchain.")
    args.output.mkdir(parents=True, exist_ok=True)
    (stage if args.phase == "stage" else finish)(args)


if __name__ == "__main__":
    main()
