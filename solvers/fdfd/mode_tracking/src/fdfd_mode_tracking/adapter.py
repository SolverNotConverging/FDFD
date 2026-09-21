"""Waveguide adapter and independent transverse-confinement evidence."""
from dataclasses import replace
import hashlib
import numpy as np
from .contracts import CandidateSet
from .metrics import normalize_fields, edge_fractions, resampled_vectors, align_subspaces


def material_index_guess(solver):
    """Highest material index magnitude; diagonal media use a conservative bound.

    For scalar media this selects sqrt(epsilon_r * mu_r) with greatest magnitude.
    For diagonal anisotropy all principal epsilon/mu pairs are considered, including the
    transverse crossed pairs. Conductors and surface impedances are excluded.
    """
    from cem_common.materials import Material, bulk_values
    media = [solver.background_material]
    media.extend(record.material for record, _ in solver._objects.values()
                 if isinstance(record.material, Material))
    indices = []
    for material in media:
        epsilon, mu = bulk_values(material)
        products = np.asarray(epsilon).reshape(-1, 1) * np.asarray(mu).reshape(1, -1)
        indices.extend(np.sqrt(products.astype(complex)).ravel())
    return complex(max(indices, key=abs))


def require_no_pml(solver):
    if not hasattr(solver, '_pmls') or solver._pmls:
        raise ValueError('Port eigenproblems require explicit no-PML provenance.')


def candidates_from_result(result, config):
    """Normalize an archived/live ModeSet; confinement starts unresolved."""
    if result.family != 'fdfd_waveguide_modes':
        raise ValueError('Only FDFD waveguide ModeSet results are supported.')
    if 'pml' not in result.metadata or result.metadata['pml']:
        raise ValueError('PML or missing no-PML provenance prevents port tracking.')
    if result.metadata.get('field_normalization') != 'native eigenvector normalization; H_num=-i*eta0*H':
        raise ValueError('Unknown magnetic field convention.')
    if not np.isfinite(result.metadata.get('eta0', np.nan)) or result.metadata['eta0'] <= 0:
        raise ValueError('Missing physical impedance convention.')
    fields, vectors, scales, power, finite = normalize_fields(result)
    residuals = np.asarray(result.solve_info.get('residuals', np.full(len(result), np.inf)))
    field_residuals = np.asarray(result.solve_info.get('field_residuals', np.full(len(result), np.inf)))
    if residuals.shape != (len(result),):
        raise ValueError('Residuals must follow the returned candidate ordering.')
    cutoff = abs(result.neff) <= config.cutoff_neff
    valid = finite & np.isfinite(residuals) & (residuals <= config.residual_tolerance) & ~cutoff
    valid &= np.isfinite(field_residuals) & (field_residuals <= config.residual_tolerance)
    edge = edge_fractions(result, fields)
    propagation, evidence = [], []
    for i, n in enumerate(result.neff):
        tol = 1e-8*max(1., abs(n))
        state = ('cutoff_unresolved' if cutoff[i] else
                 'evanescent' if abs(n.real) <= tol and n.imag < -tol else
                 'propagating' if abs(n.imag) <= tol else 'complex')
        reasons = []
        if cutoff[i]: reasons.append('cutoff_reconstruction_unresolved')
        if not finite[i]: reasons.append('nonfinite_or_zero_fields')
        if not np.isfinite(residuals[i]) or residuals[i] > config.residual_tolerance:
            reasons.append('eigenpair_residual_failed')
        if not np.isfinite(field_residuals[i]) or field_residuals[i] > config.residual_tolerance:
            reasons.append('field_equation_residual_failed')
        propagation.append(state)
        evidence.append({'residual': float(residuals[i]), 'field_residual': float(field_residuals[i]),
                         'edge_fraction': float(edge[i]),
                         'reasons': tuple(reasons), 'verification': ()})
    return CandidateSet(result, fields, vectors, scales, power, valid,
                        tuple(propagation), ('unresolved',)*len(result), tuple(evidence))


def fingerprint(result):
    digest = hashlib.sha256()
    digest.update(repr((result.mesh_data.bounds, result.mesh_data.resolution,
                        result.metadata.get('context'), result.metadata.get('boundaries'),
                        result.metadata.get('pml'))).encode())
    return digest.hexdigest()


def verify_candidates(base, variants, port, config):
    """variants maps mesh/padding_1/padding_2 to independent CandidateSets."""
    enclosed = port.boundary == 'enclosed'
    physical_wall = base.result.metadata.get('boundaries', {}).get('closed_conductor_boundary', False)
    required = ('mesh',) if enclosed else ('mesh', 'padding_1', 'padding_2')
    confinement, evidence = [], []
    comparisons = {}
    for key in required:
        if key not in variants:
            continue
        candidate = variants[key]
        br, vr = base.result, candidate.result
        old_bounds, new_bounds = np.array(br.mesh_data.bounds), np.array(vr.mesh_data.bounds)
        old_step = np.diff(old_bounds).ravel()/np.array(br.mesh_data.resolution)
        new_step = np.diff(new_bounds).ravel()/np.array(vr.mesh_data.resolution)
        if key == 'mesh':
            if not np.allclose(old_bounds, new_bounds, rtol=0, atol=1e-14) or not np.all(new_step < old_step*0.75):
                raise ValueError('Mesh verification must retain bounds and refine every axis.')
        elif not (np.all(new_bounds[:, 0] < old_bounds[:, 0]) and
                  np.all(new_bounds[:, 1] > old_bounds[:, 1]) and
                  np.allclose(old_step, new_step, rtol=1e-8, atol=1e-14)):
            raise ValueError('Padding verification must enlarge every artificial edge at fixed spacing.')
        vectors = resampled_vectors(br, vr, candidate.fields)
        overlaps = abs(base.vectors.conj().T @ vectors)**2
        comparisons[key] = (candidate, overlaps, vectors)
    for i in range(len(base.result)):
        record = dict(base.evidence[i])
        cutoff = base.propagation[i] == 'cutoff_unresolved'
        # At exact longitudinal cutoff the beta-based field reconstruction can
        # be singular even though a physical enclosure already proves
        # transverse confinement. Keep numerical/export validity separate from
        # that confinement fact. Open boundaries still require field-based
        # enlargement evidence and therefore remain unresolved here.
        if cutoff and enclosed:
            walls = [physical_wall]
            checks = []
            for key in required:
                other = variants.get(key)
                wall = bool(other is not None and other.result.metadata.get(
                    'boundaries', {}).get('closed_conductor_boundary', False))
                walls.append(wall)
                checks.append({'kind': key, 'overlap': np.nan, 'beta_drift': np.nan,
                               'candidate': None, 'passed': wall,
                               'fingerprint': None if other is None else fingerprint(other.result)})
            residual_ok = (np.isfinite(record['residual'])
                           and record['residual'] <= config.residual_tolerance)
            complete = len(checks) == len(required) and all(walls)
            reasons = list(record['reasons'])
            if not physical_wall:
                reasons.append('physical_enclosure_not_present')
            if not complete:
                reasons.append('verification_missing')
            if complete and not residual_ok:
                reasons.append('confinement_verification_failed')
            state = 'bound' if complete and residual_ok else 'unresolved'
            record.update(reasons=tuple(reasons), verification=tuple(checks),
                          cutoff_confinement='physical_enclosure')
            confinement.append(state)
            evidence.append(record)
            continue
        checks, passed = [], True
        for key in required:
            if key not in comparisons:
                passed = False
                continue
            other, overlaps, vectors = comparisons[key]
            delta = abs(other.result.beta-base.result.beta[i])/max(abs(base.result.beta[i]), base.result.metadata['k0'])
            j = int(np.argmax(overlaps[i] - np.minimum(delta, 1.)))
            overlap = float(overlaps[i, j])
            # Compare whole nearly degenerate spans when the eigensolver rotates them.
            left = np.flatnonzero(abs(base.eigenvalues-base.eigenvalues[i]) <= config.cluster_gap*max(1., abs(base.eigenvalues[i])))
            right = np.flatnonzero(abs(other.eigenvalues-other.eigenvalues[j]) <= config.cluster_gap*max(1., abs(other.eigenvalues[j])))
            if len(left) > 1 and len(left) == len(right):
                sub = align_subspaces(base.vectors[:, left], vectors[:, right])
                if sub['old_rank'] == len(left) == sub['new_rank']:
                    overlap = float(np.min(sub['singular_values'])**2)
            ok = bool(other.numerical_valid[j] and delta[j] <= config.verification_beta_tolerance
                      and overlap >= config.verification_overlap)
            if enclosed:
                ok &= bool(other.result.metadata.get('boundaries', {}).get('closed_conductor_boundary', False))
            else:
                ok &= other.evidence[j]['edge_fraction'] <= config.edge_fraction_max
            checks.append({'kind': key, 'overlap': overlap, 'beta_drift': float(delta[j]),
                           'candidate': j, 'passed': ok, 'fingerprint': fingerprint(other.result)})
            passed &= ok
        reasons = list(record['reasons'])
        if enclosed and not physical_wall:
            passed = False
            reasons.append('physical_enclosure_not_present')
        if not enclosed and record['edge_fraction'] > config.edge_fraction_max:
            passed = False
            reasons.append('artificial_edge_participation')
        complete = len(checks) == len(required)
        if not complete: reasons.append('verification_missing')
        if complete and not passed: reasons.append('confinement_verification_failed')
        state = 'bound' if passed and complete else ('radiation_or_box_suspect' if complete and not enclosed else 'unresolved')
        record.update(reasons=tuple(reasons), verification=tuple(checks))
        confinement.append(state)
        evidence.append(record)
    return replace(base, confinement=tuple(confinement), evidence=tuple(evidence))
