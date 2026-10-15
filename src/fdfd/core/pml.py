"""One uniaxial PML shared by every solver that supports absorbing boundaries.

The layer is a complex coordinate stretch sampled at cell centres,

    s(d) = 1 - i * strength * (d / thickness) ** order,

where ``d`` is the depth into a layer of physical ``thickness`` (metres). The
stretch is dimensionless and does not depend on frequency or on the mesh: the
thickness is not rounded to whole cells. With ``exp(+i*omega*t)`` the negative
imaginary part makes outgoing waves decay inside the layer. The stretch enters
the cell tensors as ``Lambda = diag(sy*sz/sx, sx*sz/sy, sx*sy/sz)``.
"""

from math import log

import numpy as np

from .constants import C0
from .errors import BackendCapabilityError, ConfigurationError
from .materials import _positive


def normalize_sides(sides, axes):
    """Expand ``"all"``, ``"x"`` or ``"x+"`` (or a sequence of them) to explicit sides."""
    requested = (sides,) if isinstance(sides, str) else tuple(sides)
    if not requested:
        raise ConfigurationError("sides must name at least one PML side.")
    selected = []
    for item in requested:
        if item == "all":
            names = [axis + side for axis in axes for side in "-+"]
        elif item in axes:
            names = [item + "-", item + "+"]
        elif isinstance(item, str) and len(item) == 2 and item[0] in axes and item[1] in "-+":
            names = [item]
        else:
            choices = ("all", *axes, *(a + s for a in axes for s in "-+"))
            raise BackendCapabilityError(f"PML sides must be drawn from {choices}; received {item!r}.")
        selected.extend(name for name in names if name not in selected)
    return tuple(selected)


def validate(*, thickness, sides, strength, order, axes, ranges, all_axes):
    """Check a PML request and return it as a plain dictionary."""
    thickness = _positive(thickness, "thickness")
    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ConfigurationError("PML order must be a positive integer.")
    if isinstance(strength, bool) or not np.isfinite(strength) or strength < 0:
        raise ConfigurationError("PML strength must be finite and nonnegative.")
    sides = normalize_sides(sides, axes)
    for axis in {side[0] for side in sides}:
        lo, hi = ranges[all_axes.index(axis)]
        layers = sum(side[0] == axis for side in sides)
        if layers * thickness >= hi - lo:
            raise ConfigurationError(f"PML along {axis} leaves no physical interior.")
    return {"thickness": thickness, "sides": sides, "strength": float(strength), "order": int(order)}


def stretches(specs, *, axes, ranges, resolution):
    """Per-axis complex stretch at cell centres and the mask of PML cells.

    Returns ``(stretch, mask)``: ``stretch`` maps each physical axis to an
    array broadcastable to the cell grid; ``mask`` marks cells with loss.
    """
    shape = tuple(resolution)
    stretch = {}
    mask = np.zeros(shape, dtype=bool)
    for index, axis in enumerate(axes):
        lo, hi = ranges[index]
        centres = lo + (np.arange(shape[index]) + 0.5) * (hi - lo) / shape[index]
        loss = np.zeros(shape[index])
        for spec in specs:
            for side in spec["sides"]:
                if side[0] != axis:
                    continue
                depth = (
                    lo + spec["thickness"] - centres if side[1] == "-" else centres - hi + spec["thickness"]
                )
                depth = np.clip(depth / spec["thickness"], 0.0, None)
                loss += spec["strength"] * depth ** spec["order"]
        view = [1] * len(shape)
        view[index] = shape[index]
        stretch[axis] = (1.0 - 1j * loss).reshape(view)
        mask |= np.broadcast_to((loss > 0).reshape(view), shape)
    return stretch, mask


def tensor_factors(stretch, shape):
    """Return the ``(xx, yy, zz)`` factors of ``diag(sy*sz/sx, sx*sz/sy, sx*sy/sz)``."""
    sx, sy, sz = (np.broadcast_to(stretch.get(axis, 1.0), shape) for axis in "xyz")
    return sy * sz / sx, sx * sz / sy, sx * sy / sz


def suggest_strength(*, thickness, frequency, reflection=1e-8, order=3, index=1.0):
    """PML ``strength`` for a target normal-incidence reflection.

    A plane wave crossing the layer twice is attenuated by
    ``exp(-2 * k * strength * thickness / (order + 1))`` with
    ``k = 2*pi*frequency*index/c``. Discretization reflection is not included,
    so resolve the layer with several cells and prefer moderate strengths.

    Parameters
    ----------
    thickness : float
        Layer thickness in metres.
    frequency : float
        Frequency in hertz.
    reflection : float
        Target reflection magnitude, between 0 and 1.
    order : int
        Polynomial grading order of the stretch.
    index : float
        Refractive index of the medium next to the layer.
    """
    thickness = _positive(thickness, "thickness")
    k = 2 * np.pi * _positive(frequency, "frequency") * _positive(index, "index") / C0
    if not 0 < reflection < 1:
        raise ConfigurationError("reflection must lie strictly between 0 and 1.")
    return -(order + 1) * log(reflection) / (2 * k * thickness)
