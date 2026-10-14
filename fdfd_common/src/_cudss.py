"""Small ctypes binding to cuDSS's public C API (optional runtime).

No NVIDIA binaries are bundled. CUDSS_LIBRARY overrides automatic discovery
in system paths, the Python environment, and repository build/cuda-deps.
Numeric factorization and triangular solves stay on the GPU; cuDSS performs
its own host-side symbolic analysis. Buffers remain owned by CuPy.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import os
import sys
from pathlib import Path

# CUDA's compiler loads helper DLLs lazily, even after a factor is destroyed.
# Keep the process DLL search directories alive across repeated solves.
_dll_directories = {}
_compiler_libraries = {}


def find_cudss_library():
    """Prefer an explicit override, then system and local dependency installs."""
    override = os.environ.get("CUDSS_LIBRARY")
    if override:
        return override
    system_library = ctypes.util.find_library("cudss")
    if system_library:
        return system_library
    filename = "cudss64_0.dll" if os.name == "nt" else "libcudss.so*"
    # Windows DLLs can be on PATH without find_library resolving them.
    if os.name == "nt":
        for directory in os.get_exec_path():
            candidate = Path(directory) / filename
            if candidate.is_file():
                return str(candidate.resolve())
    repository = Path(__file__).resolve().parents[2]
    roots = [repository / "build" / "cuda-deps"]
    # NVIDIA Python runtime packages use nvidia/cu12 or cu13/{bin,lib}.
    roots.extend(Path(directory) / "nvidia" for directory in sys.path
                 if directory and Path(directory).name in ("site-packages", "dist-packages"))
    roots.extend([Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
                  Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"])
    for root in roots:
        if root.is_dir():
            candidates = sorted(root.rglob(filename), reverse=True)
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate.resolve())
    raise ImportError(
        "cuDSS was not found in system paths, the Python environment, or "
        "build/cuda-deps. Set CUDSS_LIBRARY to cudss64_0.dll or libcudss.so "
        "(cuDSS >= 0.8)."
    )


def configure_dll_search():
    if os.name != "nt":
        return
    import cupy as cp
    cuda_path = cp.cuda.get_cuda_path()
    cudss_path = find_cudss_library()
    directories = []
    if cuda_path:
        directories.append(str(Path(cuda_path) / "bin"))
        directories.append(str(Path(cuda_path) / "bin" / "x64"))
    if cudss_path:
        directories.append(str(Path(cudss_path).resolve().parent))
    for directory in directories:
        if Path(directory).is_dir() and directory not in _dll_directories:
            _dll_directories[directory] = os.add_dll_directory(directory)
        # NVRTC uses a legacy LoadLibrary call for its builtins on Windows.
        # Explicitly load the companion DLL from the selected CUDA toolkit.
        for builtin in Path(directory).glob("nvrtc-builtins64_*.dll"):
            if str(builtin) not in _compiler_libraries:
                _compiler_libraries[str(builtin)] = ct.WinDLL(str(builtin))


class CudssFactor:
    def __init__(self, matrix):
        import cupy as cp

        self.cp = cp
        self.stream = cp.cuda.get_current_stream()
        self.device = cp.cuda.Device()
        configure_dll_search()
        path = find_cudss_library()
        self.lib = (ct.WinDLL if os.name == "nt" else ct.CDLL)(path)
        p, i, n = ct.c_void_p, ct.c_int, ct.c_int64
        signatures = {
            "cudssCreate": [ct.POINTER(p)], "cudssDestroy": [p],
            "cudssConfigCreate": [ct.POINTER(p)], "cudssConfigDestroy": [p],
            "cudssDataCreate": [p, ct.POINTER(p)], "cudssDataDestroy": [p, p],
            "cudssSetStream": [p, p],
            "cudssMatrixCreateCsr": [ct.POINTER(p), n, n, n, p, p, p, p, i, i, i, i, i, i],
            "cudssMatrixCreateDn": [ct.POINTER(p), n, n, n, p, i, i],
            "cudssMatrixDestroy": [p],
            "cudssExecute": [p, i, p, p, p, p, p],
            "cudssDataGet": [p, p, i, p, ct.c_size_t, ct.POINTER(ct.c_size_t)],
            "cudssGetProperty": [i, ct.POINTER(i)],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, i
        version = []
        for property_type in range(3):
            value = i()
            self._call("cudssGetProperty", property_type, ct.byref(value))
            version.append(value.value)
        if tuple(version) < (0, 8, 0):
            raise ImportError("CUDA Arnoldi requires cuDSS >= 0.8 (updated CSR descriptor ABI).")
        self.handle, self.config, self.data = p(), p(), p()
        self.a, self.x, self.b = p(), p(), p()
        self.matrix = matrix.astype(cp.complex128).tocsr()
        self.matrix.sum_duplicates()
        self.matrix.sort_indices()
        size = self.matrix.shape[0]
        self.rhs = cp.empty(size, dtype=cp.complex128)
        self.solution = cp.empty_like(self.rhs)
        try:
            self._call("cudssCreate", ct.byref(self.handle))
            self._call("cudssSetStream", self.handle, self.stream.ptr)
            self._call("cudssConfigCreate", ct.byref(self.config))
            self._call("cudssDataCreate", self.handle, ct.byref(self.data))
            # cudaDataType: int32=10, int64=24, complex128=5. General,
            # full CSR, zero-based indexing; dense column-major vectors.
            offset_type = 10 if self.matrix.indptr.dtype.itemsize == 4 else 24
            index_type = 10 if self.matrix.indices.dtype.itemsize == 4 else 24
            self._call("cudssMatrixCreateCsr", ct.byref(self.a), size, size,
                       self.matrix.nnz, self.matrix.indptr.data.ptr, None,
                       self.matrix.indices.data.ptr, self.matrix.data.data.ptr,
                       offset_type, index_type, 5, 0, 0, 0)
            for wrapper, buffer in ((self.x, self.solution), (self.b, self.rhs)):
                self._call("cudssMatrixCreateDn", ct.byref(wrapper), size, 1,
                           size, buffer.data.ptr, 5, 0)
            self._execute(3)  # symbolic analysis
            self._execute(4)  # numeric factorization
            self.stream.synchronize()
            info, written = ct.c_int(), ct.c_size_t()
            self._call("cudssDataGet", self.handle, self.data, 0, ct.byref(info),
                       ct.sizeof(info), ct.byref(written))
            if info.value:
                raise ValueError(f"cuDSS factorization reports numerical failure {info.value}; change sigma.")
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with cuDSS status {status}.")

    def _execute(self, phase):
        self._call("cudssExecute", self.handle, phase, self.config, self.data,
                   self.a, self.x, self.b)

    def solve(self, rhs):
        if self.cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise RuntimeError("cuDSS factor must be used on its original CUDA stream.")
        self.rhs[...] = rhs
        self._execute(1008)  # forward/back substitution and refinement
        return self.solution.copy()

    def close(self):
        # Synchronize before releasing descriptors or buffers used asynchronously.
        self.stream.synchronize()
        for wrapper in (self.a, self.x, self.b):
            if wrapper.value:
                self.lib.cudssMatrixDestroy(wrapper)
                wrapper.value = None
        if self.data.value:
            self.lib.cudssDataDestroy(self.handle, self.data)
            self.data.value = None
        if self.config.value:
            self.lib.cudssConfigDestroy(self.config)
            self.config.value = None
        if self.handle.value:
            self.lib.cudssDestroy(self.handle)
            self.handle.value = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
