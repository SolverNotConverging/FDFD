"""Closed-contour 2.5D near-to-far transformation by Lorentz reciprocity.

Conventions: exp(+i omega t), exp(-i ky y), theta from +x towards +z.
The reciprocal test solution has (-ky, -kz) and unit electric incidence from
the observation half-space. Its layered reflected/transmitted fields make
the *entire closed contour*, including its waveguide crossings, meaningful.
No guided-mode subtraction or omitted port segments are used.
"""
from dataclasses import dataclass, replace

import numpy as np

from .constants import C0, EPSILON_0, MU_0
from .exceptions import ConfigurationError


def _real(value, name, positive=False):
    if isinstance(value, (bool, str)) or np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ConfigurationError(f"{name} must be a finite real scalar.")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ConfigurationError(f"{name} must be finite" + (" and positive." if positive else "."))
    return value


@dataclass(frozen=True)
class LayeredExterior:
    """Isotropic x-stratified infinite background; materials have length N+1.

    ``interfaces`` contains N increasing x coordinates in metres. A true entry
    of ``pec`` places an infinite PEC sheet at that interface. Outer half-spaces
    must be lossless with positive epsilon and mu. Finite layers may be lossy.
    """
    interfaces: np.ndarray
    epsilon: np.ndarray
    mu: np.ndarray
    pec: np.ndarray

    def __post_init__(self):
        if np.iscomplexobj(self.interfaces):
            raise ConfigurationError("Exterior interfaces must be real.")
        x = np.asarray(self.interfaces, dtype=float)
        eps, mu = (np.asarray(v, dtype=complex) for v in (self.epsilon, self.mu))
        raw_pec = np.asarray(self.pec)
        if raw_pec.size and raw_pec.dtype.kind != "b":
            raise ConfigurationError("Exterior PEC flags must be booleans.")
        pec = np.asarray(self.pec, dtype=bool)
        if x.ndim != 1 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
            raise ConfigurationError("Exterior interfaces must be finite and strictly increasing.")
        if eps.shape != (len(x) + 1,) or mu.shape != eps.shape or pec.shape != x.shape:
            raise ConfigurationError("Exterior needs N+1 materials and N PEC flags for N interfaces.")
        if not np.isfinite(eps).all() or not np.isfinite(mu).all() or np.any(abs(eps * mu) == 0):
            raise ConfigurationError("Exterior materials must be finite and nonzero.")
        if np.any(eps.imag > 0) or np.any(mu.imag > 0):
            raise ConfigurationError("Exterior materials must be passive.")
        for a in (eps, mu):
            if np.any(a[[0, -1]].imag != 0) or np.any(a[[0, -1]].real <= 0):
                raise ConfigurationError("Radiation half-spaces must be lossless positive-index media.")
        for name, value in (("interfaces", x), ("epsilon", eps), ("mu", mu), ("pec", pec)):
            value = np.array(value, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def reciprocal_fields(self, x, z, *, omega, ky, kz, side, polarization):
        """Unit incident TE/TM test field, including all background reflections."""
        k0 = omega / C0
        outer = 0 if side < 0 else len(self.interfaces)
        eps_ext, mu_ext = self.epsilon[outer].real, self.mu[outer].real
        k = k0 * np.sqrt(eps_ext * mu_ext)
        kt = float(np.hypot(ky, kz))
        # A fixed basis at normal incidence avoids an undefined azimuth.
        s = np.array((0., -kz / kt, ky / kt)) if kt > 1e-14 * k else np.array((0., 1., 0.))
        kt_vector = np.array((0., -ky, -kz))
        p = np.sqrt((k0**2 * self.epsilon * self.mu - kt**2).astype(complex))
        p = np.where(p.imag > 0, -p, p)
        if p[outer].real <= 1e-10 * k or abs(p[outer].imag) > 1e-12 * k:
            raise ConfigurationError("Observation direction is grazing or non-propagating.")
        # PEC sheets disconnect the reciprocal problem. Only the compartment
        # connected to the observation half-space is illuminated.
        sheets = np.flatnonzero(self.pec)
        lo, hi = 0, len(p) - 1
        if sheets.size:
            if side < 0:
                hi = int(sheets[0])
            else:
                lo = int(sheets[-1]) + 1
        if np.any(abs(p[lo:hi+1]) <= 1e-12*k):
            raise ConfigurationError("An exterior layer is exactly at its critical angle; offset the observation angle.")
        count = hi - lo + 1
        anchors_l = np.r_[self.interfaces[:1], self.interfaces] if len(self.interfaces) else np.array([0.])
        anchors_r = np.r_[self.interfaces, self.interfaces[-1:]] if len(self.interfaces) else np.array([0.])
        material = self.mu if polarization == "s" else self.epsilon

        def traces(layer, at):
            a = np.exp(-1j * p[layer] * (at - anchors_l[layer]))
            b = np.exp(+1j * p[layer] * (at - anchors_r[layer]))
            return np.array([a, b]), (-1j * p[layer] / material[layer]) * np.array([a, -b])

        matrix = np.zeros((2 * count, 2 * count), complex)
        rhs = np.zeros(2 * count, complex)
        row = 0
        # Unit plane wave referenced to global x=0, not to the stack surface.
        if lo == 0:
            matrix[row, 0] = 1
            rhs[row] = np.exp(-1j * p[outer] * anchors_l[0]) if side < 0 else 0
        else:
            u, du = traces(lo, self.interfaces[lo - 1])
            matrix[row, :2] = u if polarization == "s" else du
        row += 1
        for layer in range(lo, hi):
            u0, d0 = traces(layer, self.interfaces[layer])
            u1, d1 = traces(layer + 1, self.interfaces[layer])
            col = 2 * (layer - lo)
            matrix[row, col:col+4] = np.r_[u0, -u1]
            matrix[row+1, col:col+4] = np.r_[d0, -d1]
            row += 2
        if hi == len(p) - 1:
            matrix[row, -1] = 1
            rhs[row] = np.exp(1j * p[outer] * anchors_r[-1]) if side > 0 else 0
        else:
            u, du = traces(hi, self.interfaces[hi])
            matrix[row, -2:] = u if polarization == "s" else du
        # Scale derivative rows for conditioning across SI length scales.
        scale = np.max(abs(matrix), axis=1)
        try:
            coefficients = np.linalg.solve(matrix / scale[:, None], rhs / scale)
        except np.linalg.LinAlgError as exc:
            raise ConfigurationError("Reciprocal layer problem is singular at this direction.") from exc
        if not np.isfinite(coefficients).all():
            raise ConfigurationError("Reciprocal layer problem returned nonfinite amplitudes.")
        layers = np.searchsorted(self.interfaces, x, side="right")
        E, H = np.zeros((3, len(x)), complex), np.zeros((3, len(x)), complex)
        phase = np.exp(1j * kz * z)
        impedance = np.sqrt(MU_0 * mu_ext / (EPSILON_0 * eps_ext))
        for layer in range(lo, hi + 1):
            mask = layers == layer
            if not np.any(mask):
                continue
            for direction, anchor, coefficient in (
                (1, anchors_l[layer], coefficients[2*(layer-lo)]),
                (-1, anchors_r[layer], coefficients[2*(layer-lo)+1]),
            ):
                amplitude = coefficient * np.exp(-1j * direction * p[layer] * (x[mask] - anchor)) * phase[mask]
                wavevector = kt_vector.astype(complex)
                wavevector[0] = direction * p[layer]
                if polarization == "s":
                    e = s
                    h = np.cross(wavevector, s) / (omega * MU_0 * self.mu[layer])
                else:
                    h = s / impedance
                    e = -np.cross(wavevector, h) / (omega * EPSILON_0 * self.epsilon[layer])
                E[:, mask] += e[:, None] * amplitude
                H[:, mask] += h[:, None] * amplitude
        return E, H, s


@dataclass(frozen=True)
class FarFieldResult:
    """E ~ amplitude exp(-i q rho-i ky y)/sqrt(rho), SI metres.

    ``power_density`` is dP/(dy dtheta), W/m/radian. In stratified exteriors
    angles in opposite half-spaces can have different q and impedance.
    """
    theta: np.ndarray
    amplitude: np.ndarray
    s_amplitude: np.ndarray
    p_amplitude: np.ndarray
    power_density: np.ndarray
    transverse_wavenumber: np.ndarray
    directivity: np.ndarray | None = None
    gain: np.ndarray | None = None
    realized_gain: np.ndarray | None = None
    radiated_power: float | None = None
    accepted_power: float | None = None
    incident_power: float | None = None

    def with_feed_powers(self, *, radiated_power, incident_power, reflected_power):
        """Add 2D directivity, gain and realized gain to this angular pattern.

        D=2πU/Prad, G=2πU/(Pinc-Pref), Gr=2πU/Pinc. U is W/m/rad,
        and the three powers are W/m. Zero radiation leaves D undefined (NaN).
        A nonpositive accepted power leaves G undefined (NaN).
        """
        radiated = _real(radiated_power, "radiated_power")
        incident = _real(incident_power, "incident_power", True)
        reflected = _real(reflected_power, "reflected_power")
        if radiated < 0 or reflected < 0:
            raise ConfigurationError("Radiated and reflected powers must be nonnegative.")
        accepted = incident - reflected
        numerator = 2*np.pi*self.power_density
        return replace(self,
            directivity=numerator/radiated if radiated > 0 else np.full_like(numerator, np.nan),
            gain=numerator/accepted if accepted > 0 else np.full_like(numerator, np.nan),
            realized_gain=numerator/incident,
            radiated_power=radiated, accepted_power=accepted, incident_power=incident)

    def integrated_power(self):
        """Periodic trapezoidal angular integral; requires a full-circle grid."""
        a = self.theta
        if len(a) < 4 or np.any(np.diff(a) <= 0) or a[-1] - a[0] >= 2*np.pi:
            raise ConfigurationError("Power integration needs increasing full-circle angles without a repeated endpoint.")
        gaps = np.diff(np.r_[a, a[0] + 2*np.pi])
        if np.max(gaps) > 2.1 * np.min(gaps):
            raise ConfigurationError("Power integration requires a reasonably uniform full-circle grid.")
        return float(np.sum(gaps * (self.power_density + np.roll(self.power_density, -1)) / 2))


@dataclass(frozen=True)
class ClosedContourFields:
    """Four-sided rectangular Huygens contour, with no omitted port segments.

    Coordinates/normals have shape (2,N) in x,z order; E/H have shape (3,N).
    Positive weights are physical line quadrature weights in metres. Store
    scattered fields; the reciprocal identity cancels bound guide channels.
    """
    x_range: tuple
    z_range: tuple
    coordinates: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    E: np.ndarray
    H: np.ndarray
    frequency_hz: float
    ky: float
    exterior: LayeredExterior

    def __post_init__(self):
        for name in ("x_range", "z_range"):
            span = np.asarray(getattr(self, name), dtype=float)
            if span.shape != (2,) or not np.isfinite(span).all() or span[1] <= span[0]:
                raise ConfigurationError("Contour spans must be finite and increasing.")
            object.__setattr__(self, name, tuple(span))
        n = np.size(self.weights)
        for name, shape, dtype in (("coordinates", (2,n), float), ("normals", (2,n), float),
                                  ("weights", (n,), float), ("E", (3,n), complex), ("H", (3,n), complex)):
            raw = np.asarray(getattr(self, name))
            if dtype is float and np.iscomplexobj(raw):
                raise ConfigurationError(f"Contour {name} must be real.")
            v = np.array(raw, dtype=dtype, copy=True)
            if v.shape != shape or not np.isfinite(v).all():
                raise ConfigurationError(f"Contour {name} must be finite with shape {shape}.")
            v.flags.writeable = False
            object.__setattr__(self, name, v)
        if n == 0 or np.any(self.weights <= 0):
            raise ConfigurationError("Closed contour needs positive quadrature weights on all four sides.")
        x0, x1 = self.x_range
        z0, z1 = self.z_range
        tol = 1e-9 * max(x1-x0, z1-z0)
        assigned = np.zeros(n, bool)
        for axis, value, normal, other_span in (
            (0,x0,(-1,0),self.z_range), (0,x1,(1,0),self.z_range),
            (1,z0,(0,-1),self.x_range), (1,z1,(0,1),self.x_range),
        ):
            mask = np.all(self.normals == np.array(normal)[:,None], axis=0)
            if not np.any(mask) or not np.isclose(self.weights[mask].sum(), other_span[1]-other_span[0], rtol=1e-9, atol=tol):
                raise ConfigurationError("NF2FF contour must be closed: all four complete sides are required.")
            if np.any(abs(self.coordinates[axis,mask]-value) > tol) or np.any(self.coordinates[1-axis,mask] < other_span[0]-tol) or np.any(self.coordinates[1-axis,mask] > other_span[1]+tol):
                raise ConfigurationError("Contour samples do not lie on their declared sides.")
            assigned |= mask
        if not assigned.all():
            raise ConfigurationError("Contour normals must be outward axis unit normals.")
        object.__setattr__(self, "frequency_hz", _real(self.frequency_hz, "frequency_hz", True))
        object.__setattr__(self, "ky", _real(self.ky, "ky"))
        if not isinstance(self.exterior, LayeredExterior):
            raise ConfigurationError("Contour requires a LayeredExterior.")

    def far_field(self, theta):
        """Transform all four sides at angles in radians; exact grazing is excluded."""
        raw = np.asarray(theta)
        if np.iscomplexobj(raw):
            raise ConfigurationError("theta must be real radians.")
        angles = np.atleast_1d(np.asarray(theta, dtype=float))
        if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
            raise ConfigurationError("theta must be a nonempty finite 1D array in radians.")
        if np.any(abs(np.cos(angles)) < 1e-8):
            raise ConfigurationError("Exact grazing directions are excluded; offset the angular grid from +/-pi/2.")
        omega = 2*np.pi*self.frequency_hz
        result = np.empty((3,len(angles)), complex)
        amplitudes = np.empty((2,len(angles)), complex)
        power, qs = np.empty(len(angles)), np.empty(len(angles))
        normals = np.vstack((self.normals[0], np.zeros(len(self.weights)), self.normals[1]))
        for j, angle in enumerate(angles):
            side = 1 if np.cos(angle) > 0 else -1
            index = -1 if side > 0 else 0
            eps, mu = self.exterior.epsilon[index].real, self.exterior.mu[index].real
            k = omega/C0 * np.sqrt(eps*mu)
            if abs(self.ky) >= k:
                raise ConfigurationError("No propagating cylindrical far field exists for this ky in the observation medium.")
            q = np.sqrt(k*k-self.ky*self.ky)
            qs[j] = q
            wavevector = np.array((q*np.cos(angle), self.ky, q*np.sin(angle)))
            for pol_index, pol in enumerate(("s", "p")):
                et, ht, s = self.exterior.reciprocal_fields(*self.coordinates,
                    omega=omega, ky=self.ky, kz=wavevector[2], side=side, polarization=pol)
                integrand = np.cross(self.E.T, ht.T) - np.cross(et.T, self.H.T)
                integral = np.sum(self.weights * np.sum(integrand * normals.T, axis=1))
                amplitudes[pol_index,j] = -1j*omega*MU_0*mu*np.exp(-1j*np.pi/4)/np.sqrt(8*np.pi*q) * integral
            p = np.cross(wavevector/k, s)
            result[:,j] = s*amplitudes[0,j] + p*amplitudes[1,j]
            impedance = np.sqrt(MU_0*mu/(EPSILON_0*eps))
            power[j] = q/k * np.sum(abs(amplitudes[:,j])**2)/(2*impedance)
        return FarFieldResult(angles.copy(), result, amplitudes[0], amplitudes[1], power, qs)

    @property
    def outward_power(self):
        """Net scattered-field Poynting flux, including outgoing guided power."""
        flux = .5*np.real(np.cross(self.E.T, self.H.conj().T))
        return float(np.sum(self.weights * (flux[:,0]*self.normals[0] + flux[:,2]*self.normals[1])))
