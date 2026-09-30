"""Stage and qualify native runtimes for the complete FDFD Windows wheel.

Run ``--phase stage`` after building/testing the root CMake project with VTK ON.
Then run ``--phase finish`` to preserve exact cached dependency source archives
and their SPDX provenance. Keep vcpkg downloads until packaging is complete. Neither phase publishes artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
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


def verify_recipe(recipe, sbom):
    """Do not package updated ports as the recipes for older installed binaries."""
    for item in sbom.get("files", []):
        if not item["SPDXID"].startswith("SPDXRef-port-file-"):
            continue
        source = recipe / item["fileName"]
        if not source.resolve().is_relative_to(recipe.resolve()):
            raise RuntimeError(f"Recipe path escaped its port: {source}")
        for checksum in item.get("checksums", []):
            with source.open("rb") as stream:
                actual = hashlib.file_digest(stream, checksum["algorithm"].lower()).hexdigest()
            if actual != checksum["checksumValue"].lower():
                raise RuntimeError(f"Recipe changed since this dependency was built: {source}")


def package_database(vcpkg):
    packages = {}
    status = (vcpkg / "installed/vcpkg/status").read_text(encoding="utf-8")
    for record in status.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in record.splitlines() if ": " in line and not line.startswith(" "))
        if fields.get("Architecture") != "x64-windows" or fields.get("Status") != "install ok installed":
            continue
        name = fields["Package"]
        previous = packages.get(name, {})
        dependencies = ", ".join(filter(None, (previous.get("Depends"), fields.get("Depends"))))
        if "Feature" not in fields:
            previous.update(fields)
        previous["Depends"] = dependencies
        packages[name] = previous
    return packages


def clean_environment(bin_dir):
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("QT_", "QML", "PYTHON", "CONDA", "VIRTUAL_ENV"))}
    windows = Path(os.environ["SystemRoot"])
    environment["PATH"] = os.pathsep.join(map(str, (bin_dir, windows / "System32", windows)))
    environment["QT_QPA_PLATFORM"] = "minimal"
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
    run("cmake", "--install", args.build, "--config", "Release", "--prefix", bundle)
    for executables in APPS.values():
        for executable in executables:
            if not (bin_dir / (executable + ".exe")).is_file():
                raise RuntimeError(f"Missing installed executable: {executable}")
            runtime = bin_dir / (executable + ".runtime.txt")
            if not runtime.is_file() or "compiler=MSVC" not in runtime.read_text().splitlines():
                raise RuntimeError(f"Rebuild {executable} with MSVC before packaging.")
    packages = package_database(args.vcpkg_root)
    pending = ["qtbase", "hdf5", "eigen3", "gmsh", "ftxui", "vtk"]
    used = set()
    source_packages = []
    while pending:
        name = pending.pop()
        if name in used:
            continue
        used.add(name)
        package = packages[name]
        pending.extend(dep.strip().split(":")[0] for dep in package.get("Depends", "").split(",") if dep.strip())
        share = args.vcpkg_root / "installed/x64-windows/share" / name
        sbom = share / "vcpkg.spdx.json"
        if not sbom.is_file():
            raise RuntimeError(f"Missing dependency provenance: {sbom}")
        target = bundle / "licenses" / name
        shutil.copytree(share, target)
        recipe = ROOT / "vcpkg-ports" / name
        if not recipe.is_dir():
            recipe = args.vcpkg_root / "ports" / name
        if recipe.is_dir():
            verify_recipe(recipe, json.loads(sbom.read_text(encoding="utf-8")))
            shutil.copytree(recipe, bundle / "recipes" / name)
        source_packages.append({"name": name, "version": package.get("Version", ""),
                                "abi": package.get("Abi", ""),
                                "sbom": sbom.relative_to(args.vcpkg_root).as_posix()})
    shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE-FDFD.txt")
    (bundle / "samples").mkdir()
    for name in ("periodic-2d.h5", "periodic-3d.h5", "periodic-sweep.h5", "scattering.h5", "scattering-sweep.h5"):
        shutil.copy2(args.samples / name, bundle / "samples" / name)

    revision = run("git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD", cwd=ROOT).strip()
    manifest = {"project": "FDFD", "version": VERSION, "architecture": "x86_64",
                "toolchain": "MSVC x64 / vcpkg x64-windows", "git_base_revision": revision,
                "source_note": "Native application sources are in the FDFD repository; dependency sources are indexed in SOURCE_INDEX.md.",
                "packages": source_packages, "vcpkg_revision": run("git", "-c", f"safe.directory={args.vcpkg_root.as_posix()}", "-C", args.vcpkg_root, "rev-parse", "HEAD").strip()}
    (bundle / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_bundle_readme(bundle)
    log = qualify(bundle)
    (args.output / "qualification.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"Staged native runtime and {len(used)} dependency records at {bundle}")


def finish(args):
    bundle = args.output / BUNDLE_NAME
    manifest_path = bundle / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = args.output / "dependency-sources"
    sources.mkdir(exist_ok=True)
    # Preserve the exact cached upstream archives identified by vcpkg's SBOMs.
    # Refuse incomplete provenance instead of inventing source URLs or hashes.
    archives = [path for path in (args.vcpkg_root / "downloads").iterdir() if path.is_file()]
    hashes = {}
    index = ["# MSVC/vcpkg native dependency sources", "",
             "Exact source archives are in the adjacent dependency-sources directory.",
             "Installed SPDX records and licenses are in licenses/; vcpkg recipes are in recipes/.",
             "Rebuild using x64-windows at the vcpkg revision in build-manifest.json.", ""]
    for package in manifest["packages"]:
        sbom = json.loads((bundle / "licenses" / package["name"] / "vcpkg.spdx.json").read_text())
        for resource in sbom["packages"]:
            if not resource["SPDXID"].startswith("SPDXRef-resource-"):
                continue
            checksums = {entry["algorithm"]: entry["checksumValue"].lower() for entry in resource.get("checksums", [])}
            expected = checksums.get("SHA512")
            if "${" in str(resource) or not expected or not re.fullmatch(r"[0-9a-f]{128}", expected):
                if "${" not in str(resource):
                    raise RuntimeError(f"Source resource needs manual archival: {resource}")
                # Some installed vcpkg SPDX records contain unexpanded port-template
                # variables for sources built as part of another port (e.g. Qt).
                # Record that limitation rather than pretending to archive a hash.
                index.extend([f"- {package['name']}: {resource['name']}",
                              f"  Upstream: {resource['downloadLocation']}",
                              "  Archive: unavailable; the installed SPDX record contains an unexpanded source variable."])
                continue
            match = None
            for archive in archives:
                if archive not in hashes:
                    with archive.open("rb") as stream:
                        hashes[archive] = hashlib.file_digest(stream, "sha512").hexdigest()
                if hashes[archive] == expected:
                    match = archive
                    break
            if match is None:
                raise RuntimeError(f"Missing cached source archive for {resource['name']}; restore vcpkg downloads before finishing.")
            shutil.copy2(match, sources / match.name)
            index.extend([f"- {package['name']}: {match.name}",
                          f"  Source: {resource['downloadLocation']}", f"  SHA512: `{expected}`"])
    (bundle / "SOURCE_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    shutil.copy2(bundle / "SOURCE_INDEX.md", sources / "SOURCE_INDEX.md")
    write_bundle_readme(bundle)
    log = qualify(bundle)
    (args.output / "qualification.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"Qualified native runtime ready for the single FDFD wheel: {bundle}")


def write_bundle_readme(bundle):
    (bundle / "README.txt").write_text("""FDFD 1.1.0 - bundled Windows x64 native applications

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
licenses/gmsh/copyright. Qt is dynamically linked under LGPL-3.0.
Users may replace compatible library binaries and debug those modifications.
No additional restrictions are imposed. This software comes without warranty.

SOURCE_INDEX.md provides exact dependency source/build-recipe downloads and hashes.
Current Windows rebuild instructions are in the FDFD repository:
https://github.com/SolverNotConverging/FDFD/blob/main/doc/development/native_dependency_sources.md
Application source and packaging scripts are in the FDFD repository; the release
notes identify the corresponding source commit. Only the wheel is needed to run.
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("stage", "finish"))
    parser.add_argument("--build", type=Path, default=ROOT / "outputs/build-msvc")
    parser.add_argument("--samples", type=Path, default=ROOT / "outputs/native-qualification")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/native-release-1.1.0")
    parser.add_argument("--vcpkg-root", type=Path, default=Path("C:/opt/vcpkg"))
    args = parser.parse_args()
    for name in ("build", "samples", "output", "vcpkg_root"):
        setattr(args, name, getattr(args, name).resolve())
    if os.name != "nt":
        parser.error("This packager requires Windows and the MSVC/vcpkg toolchain.")
    args.output.mkdir(parents=True, exist_ok=True)
    (stage if args.phase == "stage" else finish)(args)


if __name__ == "__main__":
    main()
