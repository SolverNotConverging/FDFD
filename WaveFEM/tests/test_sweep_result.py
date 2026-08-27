from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest

import wavefem
from wavefem.sweep import FrequencySweepResult


@dataclass
class DummyResult:
    frequency_hz: float
    offset: float
    reflection: float
    transmission: float
    power_balance_error: float
    modes: tuple[str, ...] = ()
    incident_power: float = 1.0
    radiated_power: float = 0.05
    absorbed_power: float = 0.05

    @property
    def power_balance(self) -> float:
        return 1.0 - self.power_balance_error

    def S(self, side: str, *, out_mode: int = 0, in_mode: int = 0) -> complex:
        side_offset = 0.0 if side.lower() == "left" else 10.0
        return complex(self.offset + side_offset + out_mode, in_mode)


@dataclass
class ResultWithoutFrequency:
    offset: float
    reflection: float = 0.1
    transmission: float = 0.8
    power_balance_error: float = 0.1

    def S(self, side: str, *, out_mode: int = 0, in_mode: int = 0) -> complex:
        del side
        return complex(self.offset + out_mode, in_mode)


def make_sweep() -> FrequencySweepResult:
    return FrequencySweepResult(
        frequencies_hz=np.asarray((1.0e9, 2.0e9)),
        results=(
            DummyResult(1.0e9, 1.0, 0.1, 0.8, 0.1, ("mode-a",)),
            DummyResult(2.0e9, 2.0, 0.2, 0.7, 0.1, ("mode-b",)),
        ),
    )


def test_stores_an_immutable_frequency_copy_and_result_tuple() -> None:
    source = np.asarray((1.0e9, 2.0e9))
    first = ResultWithoutFrequency(1.0)
    second = ResultWithoutFrequency(2.0)
    sweep = FrequencySweepResult(source, [first, second], h5_path="saved/sweep.h5")

    source[0] = 3.0e9

    np.testing.assert_array_equal(sweep.frequencies_hz, (1.0e9, 2.0e9))
    assert not sweep.frequencies_hz.flags.writeable
    assert sweep.results == (first, second)
    assert sweep.h5_path == Path("saved/sweep.h5")


@pytest.mark.parametrize(
    "frequencies",
    [
        np.asarray([]),
        np.asarray([[1.0e9, 2.0e9]]),
        np.asarray((0.0, 1.0e9)),
        np.asarray((-1.0, 1.0e9)),
        np.asarray((1.0e9, np.nan)),
        np.asarray((1.0e9, np.inf)),
        np.asarray((1.0e9, 1.0e9)),
        np.asarray((2.0e9, 1.0e9)),
        np.asarray((1.0e9 + 0.0j, 2.0e9 + 0.0j)),
        np.asarray((False, True)),
    ],
)
def test_rejects_invalid_frequency_grids(frequencies: np.ndarray) -> None:
    with pytest.raises(ValueError, match="frequencies_hz"):
        FrequencySweepResult(frequencies, tuple(None for _ in range(frequencies.size)))


def test_rejects_result_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry for each frequency"):
        FrequencySweepResult(np.asarray((1.0e9, 2.0e9)), (ResultWithoutFrequency(1.0),))


def test_validates_exposed_result_frequency_with_roundoff_tolerance() -> None:
    accepted = DummyResult(1.0e9 * (1.0 + 5.0e-13), 1.0, 0.1, 0.8, 0.1)
    FrequencySweepResult(np.asarray((1.0e9,)), (accepted,))

    mismatched = DummyResult(1.01e9, 1.0, 0.1, 0.8, 0.1)
    with pytest.raises(ValueError, match="does not match"):
        FrequencySweepResult(np.asarray((1.0e9,)), (mismatched,))


@pytest.mark.parametrize("frequency_hz", [np.nan, 0.0, 1.0e9 + 0.0j, [1.0e9]])
def test_rejects_invalid_exposed_result_frequency(frequency_hz: object) -> None:
    result = DummyResult(1.0e9, 1.0, 0.1, 0.8, 0.1)
    result.frequency_hz = frequency_hz  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r"results\[0\]\.frequency_hz"):
        FrequencySweepResult(np.asarray((1.0e9,)), (result,))


def test_collects_modal_and_power_observables_in_frequency_order() -> None:
    sweep = make_sweep()

    np.testing.assert_array_equal(sweep.S("left", out_mode=2, in_mode=3), (3.0 + 3.0j, 4.0 + 3.0j))
    np.testing.assert_array_equal(sweep.S11, (1.0 + 0.0j, 2.0 + 0.0j))
    np.testing.assert_array_equal(sweep.S21, (11.0 + 0.0j, 12.0 + 0.0j))
    np.testing.assert_array_equal(sweep.reflection, (0.1, 0.2))
    np.testing.assert_array_equal(sweep.transmission, (0.8, 0.7))
    np.testing.assert_array_equal(sweep.power_balance_error, (0.1, 0.1))
    np.testing.assert_array_equal(sweep.incident_power, (1.0, 1.0))
    np.testing.assert_array_equal(sweep.radiated_power, (0.05, 0.05))
    np.testing.assert_array_equal(sweep.absorbed_power, (0.05, 0.05))
    np.testing.assert_array_equal(sweep.power_balance, (0.9, 0.9))

    assert sweep.S11.dtype == np.complex128
    assert sweep.reflection.dtype == np.float64


def test_save_h5_uses_lazy_helper_and_passes_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_module = ModuleType("wavefem.hdf5")

    def fake_save_sweep_h5(
        frequencies_hz: np.ndarray,
        results: tuple[DummyResult, ...],
        path: str | Path,
        *,
        modes_per_result: list[tuple[str, ...]],
    ) -> Path:
        captured.update(
            frequencies_hz=np.array(frequencies_hz, copy=True),
            results=results,
            path=path,
            modes_per_result=modes_per_result,
        )
        return Path(path)

    fake_module.save_sweep_h5 = fake_save_sweep_h5  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wavefem.hdf5", fake_module)
    monkeypatch.setattr(wavefem, "hdf5", fake_module, raising=False)
    destination = Path("sweep.h5")
    sweep = make_sweep()

    written = sweep.save_h5(destination)

    assert written == destination
    np.testing.assert_array_equal(captured["frequencies_hz"], sweep.frequencies_hz)
    assert captured["results"] is sweep.results
    assert captured["path"] == destination
    assert captured["modes_per_result"] == [("mode-a",), ("mode-b",)]
