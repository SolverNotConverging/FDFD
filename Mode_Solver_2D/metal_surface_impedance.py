"""Compatibility re-export of the repository-wide metal preset registry."""

if __package__:
    from metal_surface_impedance import (
        METAL_RESISTIVITIES_OHM_M,
        MU_0_H_PER_M,
        canonical_metal_name,
        good_conductor_surface_impedance,
        metal_conductivity,
        metal_resistivity,
    )
else:
    # A direct ``import metal_surface_impedance`` from this directory gives
    # this compatibility file the same module name as the shared root file.
    # Load the shared source under a private name to avoid importing ourselves.
    import importlib.util
    import sys
    from pathlib import Path

    _SHARED_NAME = "_fdfd_shared_metal_surface_impedance"
    _shared = sys.modules.get(_SHARED_NAME)
    if _shared is None:
        _shared_path = Path(__file__).resolve().parent.parent / "metal_surface_impedance.py"
        _spec = importlib.util.spec_from_file_location(_SHARED_NAME, _shared_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Cannot load shared metal presets from {_shared_path}.")
        _shared = importlib.util.module_from_spec(_spec)
        sys.modules[_SHARED_NAME] = _shared
        _spec.loader.exec_module(_shared)

    METAL_RESISTIVITIES_OHM_M = _shared.METAL_RESISTIVITIES_OHM_M
    MU_0_H_PER_M = _shared.MU_0_H_PER_M
    canonical_metal_name = _shared.canonical_metal_name
    good_conductor_surface_impedance = _shared.good_conductor_surface_impedance
    metal_conductivity = _shared.metal_conductivity
    metal_resistivity = _shared.metal_resistivity

__all__ = [
    "METAL_RESISTIVITIES_OHM_M",
    "MU_0_H_PER_M",
    "canonical_metal_name",
    "good_conductor_surface_impedance",
    "metal_conductivity",
    "metal_resistivity",
]
