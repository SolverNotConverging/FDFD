"""Adaptive frequency continuation, verification solves and cutoff brackets."""
from dataclasses import replace
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from tqdm.auto import tqdm
from .contracts import PortSpec, TrackingConfig, VerificationSpec, TrackingSample, TrackedSweep
from .adapter import (require_no_pml, candidates_from_result, verify_candidates,
                      fingerprint, material_index_guess)
from .assignment import assign_modes, match_clusters, eigen_clusters
from .metrics import orthonormal_basis


class _BudgetExhausted(RuntimeError):
    pass


def track_modes(make_solver, frequencies, *, port=None, config=None, seed_modes=(0,),
                reference_frequency=None, scorer=None, progress=True):
    """Track modes from a factory ``make_solver(frequency_hz, VerificationSpec)``.

    The factory returns a newly configured and meshed 1D/2D waveguide solver.
    It must preserve physical geometry during refinement/padding. Requested
    frequencies are positive and unique. Bound exact-cutoff seeds retain their
    identity and are continued into frequencies with reconstructable fields.
    Pass seed_modes=None to track the complete solved set with branch discovery
    and a material-derived default search index. Explicit tuples select seeds.
    progress=False hides the frequency progress bar. Adaptive frequencies grow
    its total; verification stages are shown without double-counting a frequency.
    """
    with tqdm(desc='Mode tracking', unit='freq', disable=not progress,
              dynamic_ncols=True) as progress_bar:
        return _track_modes(make_solver, frequencies, port=port, config=config,
                            seed_modes=seed_modes, reference_frequency=reference_frequency,
                            scorer=scorer, progress_bar=progress_bar)


def _track_modes(make_solver, frequencies, *, port, config, seed_modes,
                 reference_frequency, scorer, progress_bar):
    port, config = port or PortSpec(), config or TrackingConfig()
    requested = np.asarray(frequencies, float)
    if requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all() or np.any(requested <= 0):
        raise ValueError('frequencies must be a nonempty positive finite sequence.')
    if len(np.unique(requested)) != len(requested):
        raise ValueError('frequencies must be unique.')
    requested = np.sort(requested)
    ref = float(requested[0] if reference_frequency is None else reference_frequency)
    if ref not in requested:
        raise ValueError('reference_frequency must be one of the requested frequencies.')
    automatic = seed_modes is None
    if not automatic and (not seed_modes or any(isinstance(i, bool) or int(i) != i or i < 0 for i in seed_modes) or len(set(seed_modes)) != len(seed_modes)):
        raise ValueError('seed_modes must contain unique nonnegative integer indices.')
    cache, checked, events, unresolved = {}, {}, [], []
    solves = 0
    baseline_grid = None
    scheduled_frequencies = set(requested)
    completed_frequencies = set()
    progress_bar.total = len(scheduled_frequencies)
    progress_bar.refresh()

    def raw(frequency, spec, count, guess):
        nonlocal solves, baseline_grid
        key = (float(frequency), spec, count, guess)
        if key in cache: return cache[key]
        if solves >= config.max_solves: raise _BudgetExhausted('eigensolve budget exhausted')
        stage = ('mesh x'+str(spec.mesh_factor) if spec.mesh_factor > 1 else
                 'padding '+format(spec.padding_fraction, '.0%') if spec.padding_fraction else
                 'primary')
        progress_bar.set_postfix(frequency=f'{frequency/1e9:.6g} GHz', stage=stage)
        solver = make_solver(float(frequency), spec)
        require_no_pml(solver)
        if not np.isclose(solver.frequency, frequency, rtol=1e-13, atol=0):
            raise ValueError('Scene factory returned the wrong frequency.')
        solver._ensure_grid()
        if spec == VerificationSpec():
            grid = (solver.mesh_data.bounds, solver.mesh_data.resolution)
            if baseline_grid is None: baseline_grid = grid
            if grid != baseline_grid: raise ValueError('Base sweeps require a fixed physical Yee grid.')
        solver._backend._cutoff_neff_tolerance = config.cutoff_neff
        solver._backend._tracking_wide_search = automatic
        if guess is None and automatic:
            guess = material_index_guess(solver)
        kwargs = dict(num_modes=count, neff_guess=guess, eigensolver_tolerance=config.eigensolver_tolerance)
        if len(solver.mesh_data.axes) == 1: kwargs['polarization'] = config.polarization
        solves += 1
        result = solver.solve(**kwargs)
        result.metadata['neff_guess'] = guess
        value = candidates_from_result(result, config)
        cache[key] = value
        return value

    def solve(frequency, count, guess):
        key = (frequency, count, guess)
        if key in checked: return checked[key]
        if frequency not in scheduled_frequencies:
            scheduled_frequencies.add(frequency)
            progress_bar.total = len(scheduled_frequencies)
            progress_bar.refresh()
        base = raw(frequency, VerificationSpec(), count, guess)
        variants = {}
        specs = [('mesh', VerificationSpec(mesh_factor=2))]
        if port.boundary == 'open':
            specs += [('padding_1', VerificationSpec(padding_fraction=.25)),
                      ('padding_2', VerificationSpec(padding_fraction=.5))]
        for label, spec in specs:
            try:
                variants[label] = raw(frequency, spec, count, guess)
            except (ArpackNoConvergence, _BudgetExhausted) as exc:
                events.append({'type': 'verification_unresolved', 'frequency': frequency,
                               'kind': label, 'reason': str(exc)})
                break
        value = verify_candidates(base, variants, port, config)
        checked[key] = value
        progress_bar.set_postfix(frequency=f'{frequency/1e9:.6g} GHz', stage='checked')
        if frequency not in completed_frequencies:
            completed_frequencies.add(frequency)
            progress_bar.update(1)
        return value

    initial = solve(ref, config.num_candidates, config.neff_guess)
    if not automatic and max(seed_modes) >= len(initial.result): raise ValueError('Seed index outside candidate set.')
    # Include complete clusters: selecting one arbitrary vector is not a stable seed.
    seeds = list(range(len(initial.result))) if automatic else list(seed_modes)
    for group in eigen_clusters(initial.eigenvalues, config.cluster_gap):
        if any(i in seeds for i in group):
            seeds += [i for i in group if i not in seeds]
    def cutoff_identity_ok(candidates, index):
        evidence = candidates.evidence[index]
        return (candidates.propagation[index] == 'cutoff_unresolved'
                and np.isfinite(evidence['residual'])
                and evidence['residual'] <= config.residual_tolerance)

    seed_usable = np.array([initial.numerical_valid[i] or cutoff_identity_ok(initial, i)
                            for i in seeds])
    if automatic:
        seeds = [i for i, usable in zip(seeds, seed_usable) if usable]
    elif not np.all(seed_usable):
        raise ValueError('Reference seeds must have finite fields or a reliable cutoff eigenpair identity.')
    phases = np.ones(len(seeds), complex)
    for t, i in enumerate(seeds):
        if not initial.numerical_valid[i]:
            continue
        v = initial.vectors[:, i]
        phases[t] = np.exp(-1j*np.angle(v[np.argmax(abs(v))]))
    seed_clusters = []
    for group in eigen_clusters(initial.eigenvalues[seeds], config.cluster_gap):
        if len(group) > 1 and np.all(initial.numerical_valid[np.array(seeds)[list(group)]]):
            indices = np.array(seeds)[list(group)]
            _, transform = orthonormal_basis(initial.vectors[:, indices])
            seed_clusters.append({'tracks': group, 'candidates': tuple(indices), 'transform': transform,
                                  'singular_values': np.ones(len(group)), 'kind': 'seed_subspace'})
    root = TrackingSample(initial, np.array(seeds, dtype=int), phases, np.ones(len(seeds)), tuple(seed_clusters))
    samples = {ref: root}

    def state(sample, fallback=None):
        ids = sample.candidate_indices
        vectors = np.full((sample.candidates.vectors.shape[0], len(ids)), np.nan+0j)
        valid = np.array([i >= 0 and sample.candidates.numerical_valid[i] for i in ids], dtype=bool)
        if np.any(valid):
            vectors[:, valid] = sample.candidates.vectors[:, ids[valid]]*sample.phases[valid]
        for cluster in sample.clusters:
            tracks = list(cluster['tracks'])
            candidates = list(cluster['candidates'])
            if np.all(sample.candidates.numerical_valid[candidates]):
                vectors[:, tracks] = sample.candidates.vectors[:, candidates] @ cluster['transform']
        if fallback is not None:
            old = state(fallback)
            missing = ~np.isfinite(vectors).all(axis=0)
            available = np.isfinite(old).all(axis=0)
            vectors[:, missing & available] = old[:, missing & available]
        return vectors

    def match(previous, left, right, blocked=()):
        ids = left.candidate_indices
        present = ids >= 0
        old_values = np.zeros(len(ids), complex)
        old_values[present] = left.candidates.eigenvalues[ids[present]]
        prediction = old_values.copy()
        if previous is not None:
            dt = left.frequency-previous.frequency
            if dt:
                common = present & (previous.candidate_indices >= 0)
                prediction[common] += (right.frequency-left.frequency)/dt*(
                    old_values[common]-previous.candidates.eigenvalues[previous.candidate_indices[common]])
        ov = state(left, previous)
        nv = right.vectors
        indices = np.full(len(ids), -1, int)
        phases = np.ones(len(ids), complex)
        overlaps = np.zeros(len(ids))
        records, occupied = [], set(blocked)
        old_finite = np.flatnonzero(present & np.isfinite(ov).all(axis=0))
        new_finite = np.array([i for i in np.flatnonzero(right.numerical_valid)
                               if i not in occupied], dtype=int)
        clusters = match_clusters(ov[:, old_finite], nv[:, new_finite],
                                  old_values[old_finite], right.eigenvalues[new_finite], config)
        for cluster in clusters:
            a = tuple(int(old_finite[i]) for i in cluster['old_members'])
            b = tuple(int(new_finite[i]) for i in cluster['new_members'])
            indices[list(a)] = b
            overlaps[list(a)] = cluster['overlap']
            occupied.update(b)
            records.append({'tracks': a, 'candidates': b, 'transform': cluster['transform'],
                            'singular_values': cluster['singular_values'], 'kind': 'tracked_subspace'})
        active = np.flatnonzero(present & (indices < 0) & np.isfinite(ov).all(axis=0))
        remaining = np.array([j for j in new_finite if j not in occupied], int)
        neural_status = 'disabled'
        if len(active) and len(remaining):
            allowed = np.broadcast_to(right.numerical_valid[remaining], (len(active), len(remaining))).copy()
            old_pol = left.candidates.result.metadata['polarizations']
            new_pol = right.result.metadata['polarizations']
            for i, track in enumerate(active):
                for j, candidate in enumerate(remaining):
                    allowed[i, j] &= old_pol[ids[track]] == new_pol[candidate]
            assigned, overlap, _, neural_status = assign_modes(
                ov[:, active], nv[:, remaining], old_values[active], right.eigenvalues[remaining], config,
                prediction=prediction[active], allowed=allowed, scorer=scorer,
                relative_step=abs(right.frequency-left.frequency)/left.frequency)
            for k, j in enumerate(assigned):
                if j < 0: continue
                track, candidate = active[k], remaining[j]
                indices[track] = candidate
                overlaps[track] = overlap[k, j]
                phases[track] = np.exp(-1j*np.angle(np.vdot(ov[:, track], nv[:, candidate])))
                occupied.add(int(candidate))
        # An exact-cutoff sample has no reliable reconstructed vector. Use
        # lambda=-neff^2 and polarization only for the step touching cutoff;
        # the adjacent valid vector supplies phase transport on the next step.
        active = np.flatnonzero(present & (indices < 0))
        remaining = np.array([j for j in range(len(right.result)) if j not in occupied], int)
        if len(active) and len(remaining):
            old_pol = left.candidates.result.metadata['polarizations']
            new_pol = right.result.metadata['polarizations']
            cost = np.full((len(active), len(remaining)), 1e6)
            for ai, track in enumerate(active):
                old_cutoff = cutoff_identity_ok(left.candidates, ids[track])
                for bj, candidate in enumerate(remaining):
                    new_cutoff = cutoff_identity_ok(right, candidate)
                    if not (old_cutoff or new_cutoff):
                        continue
                    if not (right.numerical_valid[candidate] or new_cutoff):
                        continue
                    if old_pol[ids[track]] != new_pol[candidate]:
                        continue
                    cost[ai, bj] = abs(right.eigenvalues[candidate]-prediction[track]) / max(
                        1., abs(prediction[track]))
            from scipy.optimize import linear_sum_assignment
            augmented = np.column_stack((cost, np.full((len(active), len(active)), config.unmatched_cost)))
            rows, columns = linear_sum_assignment(augmented)
            best = float(augmented[rows, columns].sum())
            for ai, bj in zip(rows, columns):
                if bj >= len(remaining) or cost[ai, bj] >= config.unmatched_cost:
                    continue
                alternative = augmented.copy()
                alternative[ai, bj] = 1e6
                ar, ac = linear_sum_assignment(alternative)
                if alternative[ar, ac].sum()-best < config.assignment_margin:
                    continue
                track, candidate = int(active[ai]), int(remaining[bj])
                indices[track] = candidate
                if right.numerical_valid[candidate]:
                    vector = nv[:, candidate]
                    if np.isfinite(ov[:, track]).all():
                        phases[track] = np.exp(-1j*np.angle(np.vdot(ov[:, track], vector)))
                    else:
                        phases[track] = np.exp(-1j*np.angle(vector[np.argmax(abs(vector))]))
                occupied.add(candidate)
                events.append({'type': 'cutoff_identity', 'frequency': right.frequency,
                               'track': track, 'candidate': candidate,
                               'cost': float(cost[ai, bj])})
        if neural_status != 'disabled':
            events.append({'type': 'neural_score', 'frequency': right.frequency, 'status': neural_status})
        return TrackingSample(right, indices, phases, overlaps, tuple(records))

    def crossing(left, right):
        good = (left.candidate_indices >= 0) & (right.candidate_indices >= 0)
        a = left.candidates.eigenvalues[left.candidate_indices[good]]
        b = right.candidates.eigenvalues[right.candidate_indices[good]]
        near_real = (abs(a.imag) < 1e-8*np.maximum(1., abs(a))) & (abs(b.imag) < 1e-8*np.maximum(1., abs(b)))
        return tuple(np.flatnonzero(good)[near_real & (a.real*b.real < 0)])

    def discover(sample):
        """Give every usable unassigned eigenpair an identity, including births."""
        occupied = set(sample.candidate_indices[sample.candidate_indices >= 0])
        born = [i for i in range(len(sample.candidates.result)) if i not in occupied
                and (sample.candidates.numerical_valid[i] or cutoff_identity_ok(sample.candidates, i))]
        if not born:
            return sample
        offset = len(sample.candidate_indices)
        # Every row has the same global track columns. -1 is an absent branch,
        # never an index into the last candidate.
        for old in (*samples.values(), sample):
            old.candidate_indices = np.r_[old.candidate_indices, np.full(len(born), -1, int)]
            old.phases = np.r_[old.phases, np.ones(len(born), complex)]
            old.overlaps = np.r_[old.overlaps, np.zeros(len(born))]
        sample.candidate_indices[offset:] = born
        records = list(sample.clusters)
        for local, candidate in enumerate(born):
            if sample.candidates.numerical_valid[candidate]:
                v = sample.candidates.vectors[:, candidate]
                sample.phases[offset+local] = np.exp(-1j*np.angle(v[np.argmax(abs(v))]))
            events.append({'type': 'branch_discovered', 'frequency': sample.frequency,
                           'track': offset+local, 'candidate': candidate})
        for group in eigen_clusters(sample.candidates.eigenvalues[born], config.cluster_gap):
            candidates = tuple(born[i] for i in group)
            if len(group) > 1 and np.all(sample.candidates.numerical_valid[list(candidates)]):
                _, transform = orthonormal_basis(sample.candidates.vectors[:, candidates])
                if transform.shape[1] == len(group):
                    records.append({'tracks': tuple(offset+i for i in group), 'candidates': candidates,
                                    'transform': transform, 'singular_values': np.ones(len(group)),
                                    'kind': 'discovered_subspace'})
        sample.clusters = tuple(records)
        return sample

    def connect(previous, left, frequency, depth):
        guess = config.neff_guess
        # A predictor-centred search helps avoid losing a branch at cutoff.
        if guess is None and not automatic:
            guess = complex(np.mean(left.candidates.result.neff[left.candidate_indices]))
        try:
            candidate = solve(frequency, config.num_candidates, guess)
            sample = match(previous, left, candidate)
            if not automatic and np.any(sample.candidate_indices < 0) and config.max_candidates > config.num_candidates:
                try:
                    sample = match(previous, left, solve(frequency, config.max_candidates, guess))
                    events.append({'type': 'candidate_expansion', 'frequency': frequency})
                except _BudgetExhausted:
                    pass
        except (ArpackNoConvergence, _BudgetExhausted) as exc:
            unresolved.append((min(left.frequency, frequency), max(left.frequency, frequency)))
            events.append({'type': 'solve_unresolved', 'frequency': frequency, 'reason': str(exc)})
            return left
        if automatic:
            # Reacquire a branch after a missing/spurious sample. Historical
            # anchors are matched jointly, with already assigned candidates
            # reserved, so a gap cannot give two tracks the same eigenpair.
            direction = np.sign(frequency-left.frequency)
            history = sorted((s for s in samples.values()
                              if direction*(s.frequency-left.frequency) < 0),
                             key=lambda s: abs(s.frequency-left.frequency))
            for anchor in history:
                missing = (sample.candidate_indices < 0) & (left.candidate_indices < 0)
                if not np.any(missing):
                    break
                indices = np.where(missing, anchor.candidate_indices, -1)
                sparse = replace(anchor, candidate_indices=indices, clusters=tuple(
                    c for c in anchor.clusters if all(missing[t] for t in c['tracks'])))
                recovered = match(None, sparse, candidate,
                                  blocked=sample.candidate_indices[sample.candidate_indices >= 0])
                found = missing & (recovered.candidate_indices >= 0)
                sample.candidate_indices[found] = recovered.candidate_indices[found]
                sample.phases[found] = recovered.phases[found]
                sample.overlaps[found] = recovered.overlaps[found]
                sample.clusters += recovered.clusters
        ambiguous = bool(np.any((left.candidate_indices >= 0) & (sample.candidate_indices < 0)))
        cutoff_tracks = () if ambiguous else crossing(left, sample)
        cutoff = bool(cutoff_tracks)
        width = abs(frequency-left.frequency)/max(frequency, left.frequency)
        if (ambiguous or cutoff) and depth < config.max_depth and width > config.min_relative_step and solves < config.max_solves:
            midpoint = (frequency+left.frequency)*.5
            events.append({'type': 'refine_cutoff' if cutoff else 'refine_assignment',
                           'interval': (min(left.frequency, frequency), max(left.frequency, frequency))})
            middle = connect(previous, left, midpoint, depth+1)
            if middle.frequency != left.frequency:
                return connect(left, middle, frequency, depth+1)
        if automatic:
            sample = discover(sample)
        samples[frequency] = sample
        if ambiguous or cutoff:
            interval = (min(left.frequency, frequency), max(left.frequency, frequency))
            unresolved.append(interval)
            events.append({'type': 'cutoff_bracket' if cutoff else 'assignment_unresolved',
                           'interval': interval, 'uncertainty_hz': interval[1]-interval[0],
                           'tracks': cutoff_tracks})
        old_groups = {tuple(x['tracks']) for x in left.clusters}
        new_groups = {tuple(x['tracks']) for x in sample.clusters}
        if old_groups != new_groups:
            events.append({'type': 'cluster_merge_split', 'frequency': frequency,
                           'old': tuple(sorted(old_groups)), 'new': tuple(sorted(new_groups))})
        return left if ambiguous and not automatic else sample

    for direction in (1, -1):
        previous, current = None, root
        points = requested[requested > ref] if direction == 1 else requested[requested < ref][::-1]
        for f in points:
            next_sample = connect(previous, current, float(f), 0)
            if next_sample is not current: previous, current = current, next_sample
    ordered = tuple(samples[f] for f in sorted(samples))
    # Independent reverse pair audit on the final adaptive sampling grid.
    for left, right in zip(ordered, ordered[1:]):
        common = (left.candidate_indices >= 0) & (right.candidate_indices >= 0)
        if not np.any(common): continue
        reverse = match(None, right, left.candidates)
        same = np.array_equal(reverse.candidate_indices[common], left.candidate_indices[common])
        clustered = bool(left.clusters or right.clusters)
        if clustered:
            same = set(reverse.candidate_indices[common]) == set(left.candidate_indices[common])
        if not same:
            interval = (left.frequency, right.frequency)
            unresolved.append(interval)
            events.append({'type': 'reverse_audit_unresolved', 'interval': interval})
    return TrackedSweep(port=port, config=config, requested_frequencies=tuple(float(f) for f in requested),
                        samples=ordered, events=tuple(events), unresolved_intervals=tuple(sorted(set(unresolved))),
                        metadata={'reference_frequency': ref, 'seed_modes': tuple(seeds), 'eigensolves': solves,
                         'automatic_tracking': automatic, 'num_tracks': len(root.candidate_indices),
                         'fingerprint': fingerprint(initial.result), 'normalization': 'combined_E_H_RMS_1_V_per_m',
                         'time_convention': 'exp(+i*omega*t)', 'spatial_convention': 'exp(-i*beta*z)'})
