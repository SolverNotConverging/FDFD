"""Optional CAD construction for shared shapes; no assembly or mesh policy."""
from . import shapes as s
from .errors import BackendCapabilityError, GeometryError


def add_shape(gmsh, shape, origin, scale):
    """Return one connected OCC face/solid in the mesher's local frame."""
    occ = gmsh.model.occ
    dim = shape.dimension
    origin = tuple(origin)
    def point(p): return tuple((v-o)*scale for v, o in zip(p, origin))
    def one(items):
        found = [item for item in items if item[0] == dim]
        if len(found) != 1:
            raise BackendCapabilityError('This FEM mesher requires each Boolean object to be one connected face/solid; add disconnected parts separately.')
        return found[0]
    if isinstance(shape, s.Transformed):
        item = add_shape(gmsh, shape.shape, origin, scale)
        if shape.angle:
            from math import radians
            c = point(shape.center)
            occ.rotate([item], *(*c, 0.)[:3] if dim == 2 else c,
                       *(shape.axis if dim == 3 else (0., 0., 1.)), radians(shape.angle))
        offset = tuple(v*scale for v in shape.offset)
        occ.translate([item], *(*offset, 0.)[:3] if dim == 2 else offset)
        return item
    if isinstance(shape, s.Difference):
        a, b = add_shape(gmsh, shape.shape, origin, scale), add_shape(gmsh, shape.tool, origin, scale)
        return one(occ.cut([a], [b])[0])
    if isinstance(shape, s.Union):
        items = [add_shape(gmsh, part, origin, scale) for part in shape.shapes]
        operation = occ.intersect if isinstance(shape, s.Intersection) else occ.fuse
        result = items[0]
        for tool in items[1:]:
            result = one(operation([result], [tool])[0])
        return result
    if isinstance(shape, s.Extrusion):
        item = add_shape(gmsh, shape.shape, origin[:2], scale)
        occ.translate([item], 0., 0., (shape.z_range[0]-origin[2])*scale)
        return one(occ.extrude([item], 0., 0., (shape.z_range[1]-shape.z_range[0])*scale))
    if isinstance(shape, s.Cylinder):
        x, y, z = point((*shape.center, shape.z_range[0]))
        return 3, occ.addCylinder(x, y, z, 0., 0., (shape.z_range[1]-shape.z_range[0])*scale, shape.radius*scale)
    if isinstance(shape, s.Box):
        return 3, occ.addBox(*point(shape.bounds[::2]), *(scale*(b-a) for a, b in zip(shape.bounds[::2], shape.bounds[1::2])))
    if isinstance(shape, s.Rectangle):
        x, y = point(shape.bounds[::2])
        dx, dy = (scale*(b-a) for a, b in zip(shape.bounds[::2], shape.bounds[1::2]))
        radius = shape.radius*scale if isinstance(shape, s.RoundedRectangle) else 0.
        return 2, occ.addRectangle(x, y, 0., dx, dy, roundedRadius=radius)
    if isinstance(shape, s.Sphere):
        return 3, occ.addSphere(*point(shape.center), shape.radius*scale)
    if isinstance(shape, s.Ellipsoid):
        center = point(shape.center)
        item = (3, occ.addSphere(*center, 1.))
        occ.dilate([item], *center, *(r*scale for r in shape.radii))
        return item
    if isinstance(shape, (s.Circle, s.Ellipse, s.Annulus)):
        x, y = point(shape.center)
        radii = shape.radii if isinstance(shape, s.Ellipse) else (getattr(shape, 'radius', getattr(shape, 'outer_radius', 0.)),)*2
        item = 2, occ.addDisk(x, y, 0., radii[0]*scale, radii[1]*scale)
        if isinstance(shape, s.Annulus):
            inner = (2, occ.addDisk(x, y, 0., shape.inner_radius*scale, shape.inner_radius*scale))
            return one(occ.cut([item], [inner])[0])
        return item
    if isinstance(shape, s.Polygon):
        points = [occ.addPoint(*point(p), 0.) for p in shape.points]
        edges = [occ.addLine(a, b) for a, b in zip(points, (*points[1:], points[0]))]
        return 2, occ.addPlaneSurface([occ.addCurveLoop(edges)])
    raise GeometryError(f'No CAD representation for {type(shape).__name__}.')
