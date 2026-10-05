"""Reusable continuous geometry in metres.

2D coordinates are ordered in the solver's declared physical plane (xy or xz).
Bounds are flattened pairs, e.g. (xmin, xmax, ymin, ymax) in an xy plane.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from .errors import GeometryError


def _point(value, dimension=None):
    p = tuple(float(v) for v in value)
    if (dimension is not None and len(p) != dimension) or not np.isfinite(p).all():
        raise GeometryError('Coordinates must be finite and match the shape dimension.')
    return p


def _positive(value):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise GeometryError('Shape lengths must be finite and positive.')
    return float(value)


def _bounds(value, dim):
    raw = np.asarray(value, dtype=float)
    if raw.shape == (dim, 2):
        raw = raw.ravel()
    if raw.shape != (2*dim,) or not np.isfinite(raw).all() or np.any(raw[1::2] <= raw[::2]):
        raise GeometryError(f'Expected {dim} finite increasing pairs of bounds.')
    return tuple(raw)


class Shape:
    @property
    def dimension(self):
        return len(self.bounds)//2

    def translated(self, *, offset):
        return Transformed(shape=self, offset=_point(offset, self.dimension), angle=0.)

    def rotated(self, *, angle, center=None, axis=(0, 0, 1)):
        """Rotate in degrees in 2D, or around the supplied 3D axis."""
        return Transformed(shape=self, offset=(0.,)*self.dimension, angle=float(angle),
                           center=(0.,)*self.dimension if center is None else center, axis=axis)


@dataclass(frozen=True, kw_only=True)
class Interval(Shape):
    bounds: tuple
    def __post_init__(self):
        object.__setattr__(self, 'bounds', _bounds(self.bounds, 1))
    def contains(self, x):
        a = np.asarray(x)
        return (a >= self.bounds[0]) & (a <= self.bounds[1])


@dataclass(frozen=True, kw_only=True)
class Rectangle(Shape):
    bounds: tuple
    def __post_init__(self):
        object.__setattr__(self, 'bounds', _bounds(self.bounds, 2))
    def contains(self, *coordinates):
        arrays = np.broadcast_arrays(*coordinates)
        if len(arrays) != self.dimension:
            raise GeometryError('Coordinates do not match shape dimension.')
        return np.logical_and.reduce([(a >= lo) & (a <= hi)
                                      for a, lo, hi in zip(arrays, self.bounds[::2], self.bounds[1::2])])


@dataclass(frozen=True, kw_only=True)
class Box(Rectangle):
    def __post_init__(self):
        object.__setattr__(self, 'bounds', _bounds(self.bounds, 3))


@dataclass(frozen=True, kw_only=True)
class Circle(Shape):
    center: tuple
    radius: float
    def __post_init__(self):
        object.__setattr__(self, 'center', _point(self.center, 2))
        object.__setattr__(self, 'radius', _positive(self.radius))
    @property
    def bounds(self):
        return tuple(v for c in self.center for v in (c-self.radius, c+self.radius))
    def contains(self, *coordinates):
        return sum((np.asarray(a)-c)**2 for a, c in zip(coordinates, self.center)) <= self.radius**2


@dataclass(frozen=True, kw_only=True)
class Sphere(Circle):
    def __post_init__(self):
        object.__setattr__(self, 'center', _point(self.center, 3))
        object.__setattr__(self, 'radius', _positive(self.radius))


@dataclass(frozen=True, kw_only=True)
class Annulus(Shape):
    center: tuple
    inner_radius: float
    outer_radius: float
    def __post_init__(self):
        object.__setattr__(self, 'center', _point(self.center, 2))
        object.__setattr__(self, 'inner_radius', _positive(self.inner_radius))
        object.__setattr__(self, 'outer_radius', _positive(self.outer_radius))
        if self.inner_radius >= self.outer_radius:
            raise GeometryError('inner_radius must be smaller than outer_radius.')
    @property
    def bounds(self):
        return Circle(center=self.center, radius=self.outer_radius).bounds
    def contains(self, x, y):
        r2 = (np.asarray(x)-self.center[0])**2 + (np.asarray(y)-self.center[1])**2
        return (r2 >= self.inner_radius**2) & (r2 <= self.outer_radius**2)


@dataclass(frozen=True, kw_only=True)
class Ellipse(Shape):
    center: tuple
    radii: tuple
    def __post_init__(self):
        object.__setattr__(self, 'center', _point(self.center, 2))
        object.__setattr__(self, 'radii', tuple(_positive(v) for v in _point(self.radii, 2)))
    @property
    def bounds(self):
        return tuple(v for c, r in zip(self.center, self.radii) for v in (c-r, c+r))
    def contains(self, *coordinates):
        return sum(((np.asarray(a)-c)/r)**2 for a, c, r in zip(coordinates, self.center, self.radii)) <= 1


@dataclass(frozen=True, kw_only=True)
class Ellipsoid(Ellipse):
    def __post_init__(self):
        object.__setattr__(self, 'center', _point(self.center, 3))
        object.__setattr__(self, 'radii', tuple(_positive(v) for v in _point(self.radii, 3)))


@dataclass(frozen=True, kw_only=True)
class Polygon(Shape):
    points: tuple
    def __post_init__(self):
        points = tuple(_point(p, 2) for p in self.points)
        if len(points) < 3 or len(set(points)) != len(points):
            raise GeometryError('A polygon requires at least three distinct vertices.')
        p = np.asarray(points)
        if abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1))) < np.finfo(float).eps*max(np.ptp(p, axis=0))**2:
            raise GeometryError('A polygon must have nonzero area.')
        object.__setattr__(self, 'points', points)
    @property
    def bounds(self):
        p = np.asarray(self.points)
        return (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())
    def contains(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        inside = np.zeros(x.shape, dtype=bool)
        for a, b in zip(self.points, (*self.points[1:], self.points[0])):
            if b[1] != a[1]:
                inside ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        return inside


@dataclass(frozen=True, kw_only=True)
class RoundedRectangle(Rectangle):
    radius: float
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'radius', _positive(self.radius))
        if 2*self.radius > min(np.subtract(self.bounds[1::2], self.bounds[::2])):
            raise GeometryError('Corner radius exceeds half the smaller side.')
    def contains(self, x, y):
        x0, x1, y0, y1 = self.bounds
        dx = np.maximum(np.maximum(x0+self.radius-np.asarray(x), np.asarray(x)-x1+self.radius), 0.)
        dy = np.maximum(np.maximum(y0+self.radius-np.asarray(y), np.asarray(y)-y1+self.radius), 0.)
        return dx*dx+dy*dy <= self.radius**2


@dataclass(frozen=True, kw_only=True)
class Cylinder(Shape):
    center: tuple
    radius: float
    z_range: tuple
    def __post_init__(self):
        object.__setattr__(self, 'center', _point(self.center, 2))
        object.__setattr__(self, 'radius', _positive(self.radius))
        object.__setattr__(self, 'z_range', _bounds(self.z_range, 1))
    @property
    def bounds(self):
        return (*Circle(center=self.center, radius=self.radius).bounds, *self.z_range)
    def contains(self, x, y, z):
        return Circle(center=self.center, radius=self.radius).contains(x, y) & Interval(bounds=self.z_range).contains(z)


@dataclass(frozen=True, kw_only=True)
class Extrusion(Shape):
    shape: Shape
    z_range: tuple
    def __post_init__(self):
        if self.shape.dimension != 2:
            raise GeometryError('Extrusion requires a two-dimensional cross-section.')
        object.__setattr__(self, 'z_range', _bounds(self.z_range, 1))
    @property
    def bounds(self):
        return (*self.shape.bounds, *self.z_range)
    def contains(self, x, y, z):
        return self.shape.contains(x, y) & Interval(bounds=self.z_range).contains(z)


@dataclass(frozen=True, kw_only=True)
class Union(Shape):
    shapes: tuple
    def __post_init__(self):
        shapes = tuple(self.shapes)
        if len(shapes) < 2 or len({s.dimension for s in shapes}) != 1:
            raise GeometryError('Boolean geometry requires at least two shapes of the same dimension.')
        object.__setattr__(self, 'shapes', shapes)
    @property
    def bounds(self):
        b = np.array([s.bounds for s in self.shapes])
        return tuple(v for lo, hi in zip(b[:, ::2].min(axis=0), b[:, 1::2].max(axis=0)) for v in (lo, hi))
    def contains(self, *coordinates):
        return np.logical_or.reduce([s.contains(*coordinates) for s in self.shapes])


@dataclass(frozen=True, kw_only=True)
class Intersection(Union):
    @property
    def bounds(self):
        b = np.array([s.bounds for s in self.shapes])
        return tuple(v for lo, hi in zip(b[:, ::2].max(axis=0), b[:, 1::2].min(axis=0)) for v in (lo, hi))
    def contains(self, *coordinates):
        return np.logical_and.reduce([s.contains(*coordinates) for s in self.shapes])


@dataclass(frozen=True, kw_only=True)
class Difference(Shape):
    shape: Shape
    tool: Shape
    def __post_init__(self):
        if self.shape.dimension != self.tool.dimension:
            raise GeometryError('Difference shapes must have the same dimension.')
    @property
    def bounds(self):
        return self.shape.bounds
    def contains(self, *coordinates):
        return self.shape.contains(*coordinates) & ~self.tool.contains(*coordinates)


@dataclass(frozen=True, kw_only=True)
class Transformed(Shape):
    shape: Shape
    offset: tuple
    angle: float = 0.
    center: tuple | None = None
    axis: tuple = (0., 0., 1.)
    def __post_init__(self):
        dim = self.shape.dimension
        object.__setattr__(self, 'offset', _point(self.offset, dim))
        object.__setattr__(self, 'center', (0.,)*dim if self.center is None else _point(self.center, dim))
        axis = _point(self.axis, 3)
        if not np.isfinite(self.angle) or np.linalg.norm(axis) == 0 or (dim == 1 and self.angle != 0):
            raise GeometryError('Invalid rotation angle or axis.')
        object.__setattr__(self, 'axis', axis)
    @property
    def matrix(self):
        a = np.deg2rad(self.angle)
        if self.shape.dimension == 1:
            return np.ones((1, 1))
        if self.shape.dimension == 2:
            return np.array(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
        v = np.array(self.axis)/np.linalg.norm(self.axis)
        k = np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))
        return np.eye(3)*np.cos(a)+(1-np.cos(a))*np.outer(v, v)+np.sin(a)*k
    @property
    def bounds(self):
        b = self.shape.bounds
        corners = np.array(list(product(*zip(b[::2], b[1::2]))))
        p = (corners-self.center) @ self.matrix.T+self.center+self.offset
        return tuple(v for lo, hi in zip(p.min(axis=0), p.max(axis=0)) for v in (lo, hi))
    def contains(self, *coordinates):
        arrays = np.broadcast_arrays(*coordinates)
        p = np.stack(arrays, axis=-1)
        q = (p-self.offset-self.center) @ self.matrix+self.center
        return self.shape.contains(*(q[..., i] for i in range(self.dimension)))


@dataclass(frozen=True, kw_only=True)
class Segment(Shape):
    start: tuple
    end: tuple
    def __post_init__(self):
        object.__setattr__(self, 'start', _point(self.start, 2))
        object.__setattr__(self, 'end', _point(self.end, 2))
        if self.start == self.end:
            raise GeometryError('A segment must have nonzero length.')
    @property
    def bounds(self):
        return tuple(v for a, b in zip(self.start, self.end) for v in (min(a, b), max(a, b)))


__all__ = ['Interval', 'Rectangle', 'Circle', 'Annulus', 'Ellipse', 'Polygon',
           'RoundedRectangle', 'Box', 'Sphere', 'Cylinder', 'Ellipsoid', 'Extrusion',
           'Union', 'Difference', 'Intersection', 'Transformed', 'Segment']
