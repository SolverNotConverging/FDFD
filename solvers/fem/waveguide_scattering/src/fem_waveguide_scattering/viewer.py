"""Discovery and launch helpers for the standalone native FEM Waveguide Scattering viewer."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .exceptions import ViewerError
from cem_common._native import bundled_executable, bundled_environment


_VIEWER_BASENAME = "fem-waveguide-scattering-viewer"
_BUILD_CONFIGURATIONS = ("Release", "RelWithDebInfo", "Debug")


def _build_runtime_environment(executable: Path) -> dict[str, str] | None:
    """Return the MinGW runtime environment recorded by the build tree."""

    bundled = bundled_environment(executable)
    if bundled is not None:
        return bundled
    if os.name != "nt":
        return None
    for directory in (executable.parent, *executable.parents):
        cache = directory / "CMakeCache.txt"
        if not cache.is_file():
            continue
        try:
            lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        compiler_value = next(
            (
                line.split("=", 1)[1]
                for line in lines
                if line.startswith("CMAKE_CXX_COMPILER:") and "=" in line
            ),
            None,
        )
        if compiler_value is None:
            return None
        runtime = Path(compiler_value).expanduser().resolve().parent
        if not any((runtime / name).is_file() for name in ("Qt6Core.dll", "libstdc++-6.dll")):
            return None
        environment = os.environ.copy()
        existing = environment.get("PATH", "")
        entries = [entry for entry in existing.split(os.pathsep) if entry]
        runtime_text = str(runtime)
        runtime_key = os.path.normcase(os.path.normpath(runtime_text))
        # A Conda IDE environment commonly includes MinGW late in PATH.  It
        # must still be moved to the front or Windows loads Conda's Qt DLLs
        # first and the MinGW viewer dies with STATUS_ENTRYPOINT_NOT_FOUND.
        entries = [
            entry
            for entry in entries
            if os.path.normcase(os.path.normpath(entry)) != runtime_key
        ]
        environment["PATH"] = os.pathsep.join((runtime_text, *entries))
        plugins = runtime.parent / "share" / "qt6" / "plugins"
        if plugins.is_dir():
            environment["QT_PLUGIN_PATH"] = str(plugins)
            platform_plugins = plugins / "platforms"
            if platform_plugins.is_dir():
                environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugins)
        return environment
    return None


def _confirm_viewer_started(process: Any, executable: Path) -> None:
    """Turn an immediate native-loader failure into an actionable exception."""

    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        return_code = wait(timeout=0.35)
    except subprocess.TimeoutExpired:
        return
    if return_code is not None:
        raise ViewerError(
            f"The native FEM Waveguide Scattering viewer exited before opening a window "
            f"(exit code {return_code}): {executable}. Rebuild or install its "
            "matching Qt/HDF5 runtime."
        )


def _repository_root() -> Path | None:
    """Return the source checkout containing ``FEMWaveguideScatteringViewer``, if present."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "apps" / "fem_waveguide_scattering_viewer" / "CMakeLists.txt").is_file():
            return parent
    return None


def _executable_names() -> tuple[str, ...]:
    if os.name == "nt":
        return (f"{_VIEWER_BASENAME}.exe", _VIEWER_BASENAME)
    return (_VIEWER_BASENAME,)


def _directory_executables(directory: Path) -> list[Path]:
    candidates = [directory / name for name in _executable_names()]
    for configuration in _BUILD_CONFIGURATIONS:
        candidates.extend(
            directory / configuration / name for name in _executable_names()
        )
    if os.name != "nt":
        candidates.append(
            directory
            / f"{_VIEWER_BASENAME}.app"
            / "Contents"
            / "MacOS"
            / _VIEWER_BASENAME
        )
    return candidates


def _build_candidates(repository: Path) -> list[Path]:
    """Enumerate standalone and repository-root CMake build layouts."""

    viewer_source = repository / "apps" / "fem_waveguide_scattering_viewer"
    build_directories: list[Path] = []
    for directory in sorted(viewer_source.glob("build*")):
        if directory.is_dir():
            build_directories.append(directory)
    for directory in sorted(repository.glob("build*")):
        if not directory.is_dir():
            continue
        # A root CMake build places this target in its subproject directory.
        build_directories.append(directory / "apps" / "fem_waveguide_scattering_viewer")

    # Keep conventional paths available even before their directories exist;
    # this also makes discovery deterministic in minimally populated checkouts.
    build_directories.extend(
        (
            viewer_source / "build",
            viewer_source / "build-mingw",
            viewer_source / "build-msvc",
            viewer_source / "build-linux",
            viewer_source / "build-macos",
            repository / "outputs" / "build" / "apps" / "fem_waveguide_scattering_viewer",
        )
    )

    candidates: list[Path] = []
    seen: set[Path] = set()
    for directory in build_directories:
        for candidate in _directory_executables(directory):
            normalized = candidate.resolve()
            if normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)
    return candidates


def find_viewer_executable() -> Path:
    """Find the native GUI in an override, build tree, ``PATH``, or install.

    The explicit environment override has highest priority, followed by the native
    application bundled with FDFD, checkout builds, PATH, and local installations.
    """

    candidates: list[Path] = []
    configured = os.environ.get("FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE")
    if configured:
        candidates.append(Path(configured).expanduser())

    bundled = bundled_executable(_VIEWER_BASENAME)
    if bundled is not None:
        candidates.append(bundled)

    repository = _repository_root()
    if repository is not None:
        candidates.extend(_build_candidates(repository))

    for name in _executable_names():
        located = shutil.which(name)
        if located:
            candidates.append(Path(located))

    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        install = Path(os.environ["LOCALAPPDATA"]) / "FEMWaveguideScatteringViewer" / "bin"
        candidates.extend(install / name for name in _executable_names())

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    raise ViewerError(
        "The native fem-waveguide-scattering-viewer executable was not found. Build the "
        "FEMWaveguideScatteringViewer CMake project, install it on PATH, or set "
        "FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE to the executable path."
    )


def _viewer_target(path: str | PathLike[str] | None) -> Path:
    try:
        target = Path.cwd() if path is None else Path(path).expanduser()
        target = target.resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise ViewerError("The viewer path must name an existing HDF5 file or directory.") from exc
    if not target.exists():
        raise ViewerError(f"The viewer path does not exist: {target}")
    if not (target.is_file() or target.is_dir()):
        raise ViewerError(f"The viewer path is not a regular file or directory: {target}")
    return target


def launch_viewer(
    path: str | PathLike[str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch the native viewer for a result file or results directory.

    With ``path=None`` the current directory is opened.  Passing a directory
    lets the native viewer populate its in-window selector with every readable
    ``.h5`` and ``.hdf5`` file there.
    """

    executable = find_viewer_executable()
    target = _viewer_target(path)
    try:
        environment = _build_runtime_environment(executable)
        if environment is None:
            process = subprocess.Popen([str(executable), str(target)])
        else:
            process = subprocess.Popen(
                [str(executable), str(target)], env=environment
            )
    except OSError as exc:
        raise ViewerError(
            f"Could not launch the native FEM Waveguide Scattering viewer {executable}: {exc}"
        ) from exc
    _confirm_viewer_started(process, executable)
    return process




__all__ = ["find_viewer_executable", "launch_viewer"]
