"""Locate complete-release executables without importing GUI dependencies."""
from importlib.util import find_spec
import os
from pathlib import Path


def bundled_executable(name: str) -> Path | None:
    if os.name != "nt":
        return None
    spec = find_spec("fdfd")
    if spec is None or spec.origin is None:
        return None
    filename = name if name.endswith(".exe") else name + ".exe"
    candidate = Path(spec.origin).parent / "native/bin" / filename
    return candidate if candidate.is_file() else None


def bundled_environment(executable: Path) -> dict[str, str] | None:
    """Keep another Python/Qt installation's plugin settings out of this process."""
    if not (executable.parent.parent / "build-manifest.json").is_file():
        return None
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("QT_", "QML"))}
    # The caller may deliberately request an offscreen smoke test.
    if "QT_QPA_PLATFORM" in os.environ:
        environment["QT_QPA_PLATFORM"] = os.environ["QT_QPA_PLATFORM"]
    environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
    return environment
