"""Typed periodic result loading with lazy sweep cases."""
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from .persistence import open_periodic_h5, save_periodic_h5, save_periodic_sweep_h5


class PeriodicSweepResult(Sequence):
    """Returned multi-case archive; indexing loads only the requested case."""
    def __init__(self, archive):
        self._archive = archive
        self._results = None
        self.frequencies = archive.frequency_hz
        self.metadata = MappingProxyType({"time_convention": "exp(+i*omega*t)", "field_representation": "periodic-envelope"})

    @classmethod
    def from_results(cls, results):
        """Combine solved periodic mode sets into a frequency or parameter sweep."""
        import numpy as np
        from .results import PeriodicModeSet
        cases = tuple(results)
        if not cases or any(not isinstance(case, PeriodicModeSet) for case in cases):
            raise ValueError("results must contain one or more periodic mode sets.")
        sweep = cls.__new__(cls)
        sweep._archive, sweep._results = None, cases
        sweep.frequencies = np.asarray([case.frequency for case in cases])
        sweep.frequencies.flags.writeable = False
        sweep.metadata = MappingProxyType({"time_convention": "exp(+i*omega*t)", "field_representation": "periodic-envelope"})
        return sweep

    def __len__(self): return len(self._results) if self._results is not None else self._archive.case_count

    @property
    def solve_info(self):
        return tuple(case.solve_info for case in self)

    @property
    def mesh_data(self):
        return tuple(case.mesh_data for case in self)

    def __getitem__(self, case):
        if isinstance(case, slice):
            return tuple(self[index] for index in range(*case.indices(len(self))))
        import numpy as np
        if isinstance(case, bool) or not isinstance(case, (int, np.integer)) or not 0 <= case < len(self):
            raise IndexError("case must be a zero-based index.")
        return self._results[case] if self._results is not None else self._archive.load_case(case)

    def save(self, path): return save_periodic_sweep_h5(self, path)

    def plot(self, *, case=0, component="Ey", quantity="real", mode=0):
        return self[case].plot(component=component, quantity=quantity, mode=mode)

    def show(self, *, block=True):
        if not isinstance(block, bool):
            raise ValueError("block must be a boolean.")
        from .persistence import launch_viewer
        if self._archive is None:
            import os, tempfile
            descriptor, name = tempfile.mkstemp(prefix="cem-periodic-sweep-", suffix=".h5")
            os.close(descriptor)
            try:
                self.save(name)
                process = launch_viewer(name, _remove_on_exit=True)
            except Exception:
                Path(name).unlink(missing_ok=True)
                raise
            if block: process.wait()
            return process
        process = launch_viewer(self._archive.path)
        if block: process.wait()
        return process


def load_result(path):
    archive = open_periodic_h5(path)
    return archive.load_case(0) if archive.case_count == 1 else PeriodicSweepResult(archive)


def save_result(result, path): return save_periodic_h5(result, path)


def plot_result(result, *, component=None, quantity="real", mode=0):
    from .visualization import visualize
    from matplotlib.figure import Figure
    figure = Figure()
    axis = figure.add_subplot(111, projection="3d") if result.dimension == 3 else figure.subplots()
    visualize(result.mode(mode), component=component or "Ey", quantity=quantity, show=False, ax=axis)
    return figure


def show_result(result, *, block=True):
    from .visualization import visualize_with_gui
    process = visualize_with_gui(result)
    if block: process.wait()
    return process
