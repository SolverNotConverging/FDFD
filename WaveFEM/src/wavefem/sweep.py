"""Frequency-sweep scattering observables and lazy HDF5 persistence."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _result_frequency_hz(value: object, index: int) -> float:
    """Validate optional per-result frequency metadata without truncation."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"results[{index}].frequency_hz must be a finite positive real scalar."
        ) from exc
    if (
        array.shape != ()
        or np.iscomplexobj(array)
        or isinstance(value, (bool, str, bytes))
    ):
        raise ValueError(
            f"results[{index}].frequency_hz must be a finite positive real scalar."
        )
    try:
        frequency_hz = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"results[{index}].frequency_hz must be a finite positive real scalar."
        ) from exc
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError(
            f"results[{index}].frequency_hz must be a finite positive real scalar."
        )
    return frequency_hz


@dataclass(frozen=True, slots=True)
class FrequencySweepResult:
    """Ordered scattering results evaluated at increasing frequencies.

    Parameters
    ----------
    frequencies_hz:
        Strictly increasing positive ordinary frequencies in hertz.
    results:
        One result per frequency.  Each result is expected to expose ``S`` and
        the scalar power-ratio properties used below.  If a result also exposes
        ``frequency_hz``, that value is checked against the corresponding sweep
        frequency.
    h5_path:
        Optional path associated with an already persisted sweep.  Calling
        :meth:`save_h5` returns the written path and does not mutate this frozen
        object.
    """

    frequencies_hz: FloatArray
    results: tuple[Any, ...]
    h5_path: Path | None = None

    def __post_init__(self) -> None:
        try:
            raw_frequencies = np.asarray(self.frequencies_hz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "frequencies_hz must contain finite positive real values."
            ) from exc
        if raw_frequencies.ndim != 1 or raw_frequencies.size == 0:
            raise ValueError("frequencies_hz must be a nonempty one-dimensional array.")
        if np.iscomplexobj(raw_frequencies) or raw_frequencies.dtype.kind == "b":
            raise ValueError("frequencies_hz must contain finite positive real values.")
        try:
            frequencies = np.array(raw_frequencies, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "frequencies_hz must contain finite positive real values."
            ) from exc
        if not np.isfinite(frequencies).all() or np.any(frequencies <= 0.0):
            raise ValueError("frequencies_hz must contain finite positive real values.")
        if np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("frequencies_hz must be strictly increasing.")

        try:
            results = tuple(self.results)
        except TypeError as exc:
            raise ValueError(
                "results must be an iterable with one entry per frequency."
            ) from exc
        if len(results) != frequencies.size:
            raise ValueError(
                "results must contain exactly one entry for each frequency; "
                f"received {len(results)} result(s) for {frequencies.size} frequencies."
            )

        for index, (frequency_hz, result) in enumerate(
            zip(frequencies, results, strict=True)
        ):
            if not hasattr(result, "frequency_hz"):
                continue
            result_frequency = _result_frequency_hz(result.frequency_hz, index)
            if not np.isclose(
                result_frequency,
                frequency_hz,
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(
                    f"results[{index}].frequency_hz={result_frequency:.16g} does not "
                    f"match frequencies_hz[{index}]={frequency_hz:.16g}."
                )

        frequencies.setflags(write=False)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "results", results)
        if self.h5_path is not None:
            try:
                path = Path(self.h5_path)
            except TypeError as exc:
                raise ValueError("h5_path must be path-like or None.") from exc
            object.__setattr__(self, "h5_path", path)

    def S(
        self,
        side: str,
        *,
        out_mode: int = 0,
        in_mode: int = 0,
    ) -> ComplexArray:
        """Return one indexed modal amplitude across the sweep."""

        return np.asarray(
            [
                result.S(side, out_mode=out_mode, in_mode=in_mode)
                for result in self.results
            ],
            dtype=np.complex128,
        )

    @property
    def S11(self) -> ComplexArray:
        """Fundamental reflected modal amplitude at every frequency."""

        return self.S("left", out_mode=0, in_mode=0)

    @property
    def S21(self) -> ComplexArray:
        """Fundamental transmitted modal amplitude at every frequency."""

        return self.S("right", out_mode=0, in_mode=0)

    @property
    def reflection(self) -> FloatArray:
        """Total reflected-power ratio at every frequency."""

        return np.asarray(
            [result.reflection for result in self.results], dtype=np.float64
        )

    @property
    def transmission(self) -> FloatArray:
        """Total transmitted-power ratio at every frequency."""

        return np.asarray(
            [result.transmission for result in self.results], dtype=np.float64
        )

    @property
    def power_balance_error(self) -> FloatArray:
        """Dimensionless power-balance error at every frequency."""

        return np.asarray(
            [result.power_balance_error for result in self.results],
            dtype=np.float64,
        )

    @property
    def incident_power(self) -> FloatArray:
        """Incident modal power at every frequency."""

        return np.asarray(
            [result.incident_power for result in self.results], dtype=np.float64
        )

    @property
    def radiated_power(self) -> FloatArray:
        """Outward radiated power at every frequency."""

        return np.asarray(
            [result.radiated_power for result in self.results], dtype=np.float64
        )

    @property
    def absorbed_power(self) -> FloatArray:
        """Material-absorbed power at every frequency."""

        return np.asarray(
            [result.absorbed_power for result in self.results], dtype=np.float64
        )

    @property
    def power_balance(self) -> FloatArray:
        """Accounted output-power fraction at every frequency."""

        return np.asarray(
            [result.power_balance for result in self.results], dtype=np.float64
        )

    def save_h5(self, path: str | PathLike[str]) -> Path:
        """Persist this sweep and return the HDF5 path.

        The local import keeps HDF5 support optional for users who only need
        in-memory sweeps.
        """

        from .hdf5 import save_sweep_h5

        written = save_sweep_h5(
            self.frequencies_hz,
            self.results,
            path,
            modes_per_result=[getattr(result, "modes", ()) for result in self.results],
        )
        return Path(written)


__all__ = ["FrequencySweepResult"]
