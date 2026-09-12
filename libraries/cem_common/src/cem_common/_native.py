"""Locate complete-release executables without importing GUI dependencies."""
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
import os
from pathlib import Path
from typing import Iterable


def bundled_executable(name: str) -> Path | None:
    """Locate CMake-installed Windows, Unix, or macOS application binaries."""
    spec = find_spec("fdfd")
    if spec is None or spec.origin is None:
        return None
    filename = name + ".exe" if os.name == "nt" and not name.endswith(".exe") else name
    relative_paths = [Path("native/bin") / filename]
    if os.name != "nt":
        relative_paths.append(Path("native") / f"{name}.app" / "Contents/MacOS" / name)
    candidates = [Path(spec.origin).parent / path for path in relative_paths]
    try:
        installed = distribution("fdfd")
        candidates.extend(Path(installed.locate_file(Path("fdfd") / path)) for path in relative_paths)
    except PackageNotFoundError:
        pass
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def bundled_environment(executable: Path) -> dict[str, str] | None:
    """Keep another Python/Qt installation's plugin settings out of this process."""
    manifest = executable.parent.parent / "build-manifest.json"
    app_contents = executable.parent.parent
    macos_bundle = app_contents.name == "Contents" and app_contents.parent.suffix == ".app"
    if not manifest.is_file() and not macos_bundle:
        return None
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("QT_", "QML"))}
    # The caller may deliberately request an offscreen smoke test.
    if "QT_QPA_PLATFORM" in os.environ:
        environment["QT_QPA_PLATFORM"] = os.environ["QT_QPA_PLATFORM"]
    environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
    if macos_bundle:
        plugins = app_contents / "PlugIns"
        environment["QT_PLUGIN_PATH"] = str(plugins)
        environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins / "platforms")
    return environment


def source_build_environment(
    executable: Path, cmake_caches: Iterable[Path]
) -> dict[str, str] | None:
    """Construct the runtime environment for a local Windows source build."""

    if os.name != "nt":
        return None

    # An app-specific CMake record is authoritative: never substitute MinGW
    # libraries for an MSVC build, or Release libraries for a Debug build.
    manifest = executable.with_suffix(".runtime.txt")
    if manifest.is_file():
        directories = [executable.parent]
        plugins = None
        section = ""
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("["):
                section = line
            elif line and section == "[dlls]":
                directories.append(Path(line).parent)
            elif line and section == "[directories]":
                directories.append(Path(line))
            elif line and section == "[platform-plugin]":
                plugins = Path(line).parent
        environment = {
            key: value for key, value in os.environ.items()
            if not key.upper().startswith(("QT_", "QML"))
        }
        if "QT_QPA_PLATFORM" in os.environ:
            environment["QT_QPA_PLATFORM"] = os.environ["QT_QPA_PLATFORM"]
        paths = list(dict.fromkeys(str(path) for path in directories if path.is_dir()))
        environment["PATH"] = os.pathsep.join((*paths, environment.get("PATH", "")))
        if plugins is not None:
            environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins)
            environment["QT_PLUGIN_PATH"] = str(plugins.parent)
        return environment

    runtimes: list[Path] = []
    runtime_hint = executable.parent / "fdfd-native-runtime.txt"
    if runtime_hint.is_file():
        try:
            runtimes.append(
                Path(runtime_hint.read_text(encoding="utf-8").strip()).expanduser()
            )
        except OSError:
            pass

    for cache in cmake_caches:
        if not cache.is_file():
            continue
        try:
            lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        compiler_value = next(
            (
                line.split("=", 1)[1]
                for line in lines
                if line.startswith("CMAKE_CXX_COMPILER:") and "=" in line
            ),
            None,
        )
        if compiler_value:
            runtimes.append(Path(compiler_value).expanduser().parent)

    for variable in ("MINGW_PREFIX", "MSYSTEM_PREFIX"):
        if os.environ.get(variable):
            runtimes.append(Path(os.environ[variable]) / "bin")
    runtimes.extend(
        Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry
    )
    system_drive = Path(os.environ.get("SystemDrive", "C:") + "\\")
    runtimes.extend(
        system_drive / "msys64" / prefix / "bin"
        for prefix in ("mingw64", "ucrt64", "clang64")
    )

    seen: set[str] = set()
    for candidate in runtimes:
        runtime = candidate.resolve()
        runtime_key = os.path.normcase(os.path.normpath(str(runtime)))
        if runtime_key in seen:
            continue
        seen.add(runtime_key)
        if not any(
            (runtime / name).is_file() for name in ("Qt6Core.dll", "libstdc++-6.dll")
        ):
            continue

        environment = os.environ.copy()
        entries = [
            entry
            for entry in environment.get("PATH", "").split(os.pathsep)
            if entry
            and os.path.normcase(os.path.normpath(entry)) != runtime_key
        ]
        environment["PATH"] = os.pathsep.join((str(runtime), *entries))
        plugins = runtime.parent / "share" / "qt6" / "plugins"
        if plugins.is_dir():
            environment["QT_PLUGIN_PATH"] = str(plugins)
            platform_plugins = plugins / "platforms"
            if platform_plugins.is_dir():
                environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugins)
        return environment
    return None
