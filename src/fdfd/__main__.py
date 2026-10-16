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
    print(f"Compiled CPU Arnoldi kernels: {native_kernels_available()}")
    try:
        import cupy

        print(f"CuPy: {cupy.__version__}")
    except ImportError:
        print("CuPy: not installed (device='cuda' unavailable)")
    else:
        try:
            from .core._cudss import find_cudss_library

            print(f"cuDSS: {find_cudss_library()}")
            from .core.cuda import probe_eigen_available, probe_hybrid_available, require_available

            require_available()
            print("cuDSS direct GPU solves: available")
        except Exception as exc:
            print(f"CUDA/cuDSS unavailable: {exc}")
        else:
            for label, probe in (
                ("Hybrid CUDA Arnoldi / CPU Ritz", probe_hybrid_available),
                ("Resident CUDA Arnoldi / Ritz / SVD", probe_eigen_available),
            ):
                try:
                    probe()
                    print(f"{label}: available")
                except Exception as exc:
                    print(f"{label} unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
