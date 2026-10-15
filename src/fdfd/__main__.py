"""``python -m fdfd info``: report the installation and optional accelerators."""

import argparse


def main():
    from . import __version__
    from .core._refined import native_kernels_available

    parser = argparse.ArgumentParser(prog="fdfd", description="FDFD electromagnetic solvers")
    parser.add_argument("--version", action="version", version=f"fdfd {__version__}")
    parser.add_argument("command", choices=("info",))
    parser.parse_args()
    print(f"fdfd {__version__}")
    print(f"Compiled Arnoldi kernels: {native_kernels_available()}")
    try:
        import cupy

        print(f"CuPy: {cupy.__version__}")
    except ImportError:
        print("CuPy: not installed (device='cuda' unavailable)")
    else:
        try:
            from .core._cudss import find_cudss_library

            print(f"cuDSS: {find_cudss_library()}")
        except ImportError as exc:
            print(f"cuDSS: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
