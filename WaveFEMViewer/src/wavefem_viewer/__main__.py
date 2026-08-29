"""Module entry point for ``python -m wavefem_viewer``."""

from .app import main


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    raise SystemExit(main())
