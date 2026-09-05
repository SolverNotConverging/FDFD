"""Translate common primitives to the existing private mesher records."""
from . import shapes as s


def native_shape(shape, module):
    if type(shape) is s.Interval:
        return module.Interval(shape.bounds)
    if type(shape) is s.Rectangle:
        return module.Rectangle(shape.bounds[:2], shape.bounds[2:])
    if type(shape) is s.Box:
        return module.Box(shape.bounds[:2], shape.bounds[2:4], shape.bounds[4:])
    if type(shape) is s.Circle:
        return module.Circle(shape.center, shape.radius)
    if type(shape) is s.Annulus and module.__name__ != 'fem_waveguide_scattering.geometry':
        return module.Circle(shape.center, shape.outer_radius, shape.inner_radius)
    if type(shape) is s.Polygon:
        return module.Polygon(shape.points)
    if type(shape) is s.Sphere:
        return module.Sphere(shape.center, shape.radius)
    if type(shape) is s.Cylinder:
        return module.Cylinder(shape.center, shape.radius, shape.z_range)
    return shape
