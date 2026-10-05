from dataclasses import replace
import numpy as np
import pytest
from fdfd_common import materials, shapes
from fdfd_common.errors import ConfigurationError
from fdfd_waveguide_modes import ModeSolver1D, ModeSolver2D
from fdfd_mode_tracking import (PortSpec, TrackingConfig, VerificationSpec, track_modes,
                                load_sweep, export_subspace, ModeTracker1D, ModeTracker2D)
from fdfd_mode_tracking.adapter import candidates_from_result, verify_candidates
from fdfd_mode_tracking.metrics import measure, align_subspaces, orthonormal_basis
from fdfd_mode_tracking.assignment import assign_modes

C = 1/np.sqrt(8.854187817e-12*4e-7*np.pi)
WIDTH, HEIGHT = .02286, .01016
FC = C/(2*WIDTH)


def rectangle(frequency, spec=VerificationSpec(), *, loss=0.):
    nx, ny = 32, 16
    dx, dy = WIDTH/nx, HEIGHT/ny
    bounds = ((-dx, WIDTH+dx), (-dy, HEIGHT+dy))
    from fdfd_common import Material
    s = ModeSolver2D(frequency=frequency, x_range=bounds[0], y_range=bounds[1],
                     background_material=Material(name='fill', epsilon=1-1j*loss))
    wall = shapes.Difference(shape=shapes.Rectangle(bounds=bounds),
                              tool=shapes.Rectangle(bounds=((0., WIDTH), (0., HEIGHT))))
    s.add_geometry(shape=wall, material=materials.PEC, name='physical_wall')
    s.mesh(resolution=((nx+2)*spec.mesh_factor, (ny+2)*spec.mesh_factor), subpixels=1)
    return s


def slab(frequency, spec=VerificationSpec()):
    dx = WIDTH/80
    s = ModeSolver1D(frequency=frequency, x_range=(-dx, WIDTH+dx))
    s.add_layer(x_range=(-dx, 0.), material=materials.PEC)
    s.add_layer(x_range=(WIDTH, WIDTH+dx), material=materials.PEC)
    s.mesh(resolution=82*spec.mesh_factor, subpixels=1)
    return s


def config(**kw):
    return replace(TrackingConfig(num_candidates=2, max_candidates=3, max_depth=5,
                                  max_solves=100, verification_beta_tolerance=.006,
                                  verification_overlap=.995), **kw)


def test_measured_residual_order_and_archive(tmp_path):
    s = slab(.8*FC)
    r = s.solve(num_modes=3, polarization='both', neff_guess=-.7j)
    assert r.metadata['pml'] == ()
    assert r.metadata['boundaries']['closed_conductor_boundary']
    for j, (pol, index) in enumerate(r.metadata['source_indices']):
        assert r.solve_info['residuals'][j] == getattr(s._backend, 'residuals_'+pol)[index]
    r.save(tmp_path/'raw.h5')
    from fdfd_waveguide_modes import load_result
    loaded = load_result(tmp_path/'raw.h5')
    np.testing.assert_allclose(loaded.solve_info['residuals'], r.solve_info['residuals'])


@pytest.mark.parametrize('factory,pol', [(slab, 'TE'), (rectangle, 'both')])
def test_bound_evanescent_normalization_and_decay(factory, pol):
    cfg = config(polarization=pol)
    sweep = track_modes(factory, [.8*FC, .9*FC], port=PortSpec(boundary='enclosed'), config=cfg)
    modes = sweep.export()
    for mode, sample in zip(modes, sweep.samples):
        n = mode.beta/(2*np.pi*mode.frequency/C)
        exact = -1j*np.sqrt((FC/mode.frequency)**2-1)
        assert abs(n-exact) < .003
        assert mode.metadata['propagation'] == 'evanescent'
        assert mode.metadata['confinement'] == 'bound'
        assert abs(mode.complex_power.real) < 1e-12
        assert abs(np.exp(-1j*mode.beta*.01)) < 1
        _, norms = measure(sample.candidates.result, sample.candidates.fields)
        np.testing.assert_allclose(norms, 1., atol=1e-12)


def test_cutoff_branch_and_reverse_orientation(tmp_path):
    sweep = track_modes(rectangle, [.8*FC, 1.2*FC], port=PortSpec(boundary='enclosed'), config=config())
    assert sweep.metadata['reference_frequency'] == .8*FC
    assert any(e['type'] == 'cutoff_bracket' for e in sweep.events)
    assert sweep.unresolved_intervals
    modes = sweep.export()
    assert modes[0].beta.imag < 0 and modes[-1].beta.real > 0
    assert not modes[0].metadata['continuous_band_certified']
    negative = replace(sweep, port=PortSpec(boundary='enclosed', direction=-1)).export()
    for a, b in zip(modes, negative):
        assert a.beta == -b.beta
        np.testing.assert_allclose(a.fields['Ez'], -b.fields['Ez'])
        np.testing.assert_allclose(a.fields['Hx'], -b.fields['Hx'])
        np.testing.assert_allclose(a.fields['Ey'], b.fields['Ey'])
        assert a.complex_power == -b.complex_power
    sweep.save(tmp_path/'sweep.h5')
    loaded = load_sweep(tmp_path/'sweep.h5')
    np.testing.assert_allclose(loaded.export()[0].fields['Ey'], modes[0].fields['Ey'])
    assert loaded.unresolved_intervals == sweep.unresolved_intervals
    with pytest.raises(ValueError, match='No validated sample'):
        loaded.export(frequencies=[FC*1.12345])


def test_exact_discrete_cutoff_is_unresolved():
    dx = WIDTH/32
    f = C/(np.pi*dx)*np.sin(np.pi/64)
    solver = rectangle(f)
    solver._backend._cutoff_neff_tolerance = 1e-5
    result = solver.solve(num_modes=1, neff_guess=.01j)
    assert abs(result.neff[0]) < 1e-5
    assert not result.solve_info['reconstruction_valid'][0]
    assert np.isnan(result.fields['Ey']).all()
    candidates = candidates_from_result(result, config(cutoff_neff=1e-5))
    assert not candidates.numerical_valid[0]
    assert candidates.propagation[0] == 'cutoff_unresolved'

    sweep = track_modes(rectangle, [f, 1.1*f], port=PortSpec(boundary='enclosed'),
                        config=config(num_candidates=1, max_candidates=2,
                                      cutoff_neff=1e-5))
    cutoff_sample = next(sample for sample in sweep.samples if sample.frequency == f)
    candidate = int(cutoff_sample.candidate_indices[0])
    assert candidate >= 0
    assert cutoff_sample.candidates.confinement[candidate] == 'bound'
    assert not cutoff_sample.candidates.eligible[candidate]
    status = sweep.plot(component='E')._mode_tracking_viewer.status(
        list(sweep.samples).index(cutoff_sample), candidate)
    assert status['cutoff'] and not status['invalid']
    assert not status['injection_eligible']
    high_sample = next(sample for sample in sweep.samples if sample.frequency == 1.1*f)
    high_candidate = int(high_sample.candidate_indices[0])
    assert high_candidate >= 0 and high_sample.candidates.eligible[high_candidate]
    assert any(event['type'] == 'cutoff_identity' for event in sweep.events)
    with pytest.raises(ValueError, match='not injection eligible'):
        sweep.export(frequencies=[f])

    unbound_candidates = replace(
        cutoff_sample.candidates,
        confinement=tuple('radiation_or_box_suspect'
                          for _ in cutoff_sample.candidates.confinement),
    )
    unbound_sample = replace(cutoff_sample, candidates=unbound_candidates)
    unbound_sweep = replace(sweep, samples=(unbound_sample,), events=(), unresolved_intervals=())
    unbound_status = unbound_sweep.plot(component='E')._mode_tracking_viewer.status(0, candidate)
    assert unbound_status['cutoff'] and unbound_status['invalid']


def test_higher_order_cutoff_seed_joins_valid_high_frequency_branch():
    dx = WIDTH/32
    second_cutoff = C/(np.pi*dx)*np.sin(np.pi/32)
    sweep = track_modes(
        rectangle, [second_cutoff, 1.1*second_cutoff],
        port=PortSpec(boundary='enclosed'),
        config=config(num_candidates=2, max_candidates=4, cutoff_neff=1e-5),
        seed_modes=(0, 1),
    )
    low = next(sample for sample in sweep.samples if sample.frequency == second_cutoff)
    high = next(sample for sample in sweep.samples if sample.frequency == 1.1*second_cutoff)
    assert low.candidates.propagation[low.candidate_indices[0]] == 'propagating'
    assert low.candidates.propagation[low.candidate_indices[1]] == 'cutoff_unresolved'
    assert low.candidates.confinement[low.candidate_indices[1]] == 'bound'
    assert all(index >= 0 for index in high.candidate_indices)
    assert all(high.candidates.eligible[index] for index in high.candidate_indices)


def test_pml_and_missing_provenance_rejected():
    def factory(f, spec):
        solver = slab(f, spec)
        solver.add_pml(thickness=.001, sigma_max=0.)
        return solver
    with pytest.raises(ValueError, match='no-PML'):
        track_modes(factory, [.8*FC], config=config())
    r = slab(.8*FC).solve(num_modes=2)
    del r.metadata['pml']
    with pytest.raises(ValueError, match='provenance'):
        candidates_from_result(r, config())


def test_declared_enclosure_is_not_enough():
    solver = ModeSolver1D(frequency=10e9, x_range=.02)
    solver.mesh(resolution=40)
    base = candidates_from_result(solver.solve(num_modes=2), config())
    verified = verify_candidates(base, {}, PortSpec(boundary='enclosed'), config())
    assert not verified.eligible.any()
    assert 'physical_enclosure_not_present' in verified.evidence[0]['reasons']


def test_loss_does_not_reject_valid_bound_modes():
    sweep = track_modes(lambda f, spec: rectangle(f, spec, loss=.01), [1.3*FC],
                        port=PortSpec(boundary='enclosed'), config=config())
    assert sweep.samples[0].candidates.eligible[0]
    mode = sweep.export()[0]
    assert mode.beta.imag < 0
    assert mode.complex_power.real > 0


def test_assignment_crossing_phase_and_missing_candidate():
    cfg = config(assignment_margin=.01)
    old = np.eye(3, dtype=complex)
    new = old[:, [1, 0]]*np.exp(1j*np.array([.7, 2.]))
    indices, overlap, _ = assign_modes(old, new, np.array([1., 2., 3.]), np.array([.9, 2.1]), cfg)
    np.testing.assert_array_equal(indices, [1, 0, -1])
    np.testing.assert_allclose(overlap[:2].max(axis=1), 1.)
    forbidden = np.zeros((3, 2), bool)
    assert np.all(assign_modes(old, new, np.arange(3), np.arange(2), cfg, allowed=forbidden)[0] == -1)


def test_complex_subspace_rotation_and_rank_loss():
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(12, 2))+1j*rng.normal(size=(12, 2)))
    unitary, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
    new = q @ unitary
    aligned = align_subspaces(q, new)
    np.testing.assert_allclose(aligned['singular_values'], 1., atol=1e-12)
    np.testing.assert_allclose(new @ aligned['transform'], q, atol=1e-12)
    coeff = np.array([1., .2j])
    np.testing.assert_allclose(new @ (unitary.conj().T @ coeff), q @ coeff, atol=1e-12)
    assert orthonormal_basis(np.column_stack([q[:, 0], q[:, 0]]))[0].shape[1] == 1


def test_conventional_assignment_is_repeatable():
    args = (np.eye(2), np.eye(2), np.array([1., 2.]), np.array([1., 2.]), config())
    base = assign_modes(*args)
    repeated = assign_modes(*args)
    for actual, expected in zip(repeated, base):
        np.testing.assert_array_equal(actual, expected)


def test_tracking_public_api_has_no_learned_scoring():
    import inspect
    from dataclasses import fields
    assert 'neural_weight' not in {field.name for field in fields(TrackingConfig)}
    for function in (track_modes, ModeTracker1D.solve, ModeTracker2D.solve, assign_modes):
        assert 'scorer' not in inspect.signature(function).parameters
    with pytest.raises(TypeError):
        TrackingConfig(neural_weight=0.)
    with pytest.raises(TypeError):
        track_modes(slab, [FC], scorer=object())


def test_legacy_sweep_loads_without_retired_control(tmp_path):
    import h5py
    from fdfd_common.persistence import write_value
    sweep = track_modes(slab, [.8*FC], port=PortSpec(boundary='enclosed'),
                        config=config(polarization='TE'), progress=False)
    # Preserve historical provenance, but never execute an old scoring model.
    historical = {'type': 'neural_score', 'frequency': .8*FC, 'status': 'applied'}
    sweep.events += (historical,)
    path = tmp_path/'legacy.h5'
    sweep.save(path)
    with h5py.File(path, 'r+') as handle:
        assert handle.attrs['schema'] == '1.1'
        assert 'neural_weight' not in handle['sweep/config']
        handle.attrs['schema'] = '1.0'
        write_value(handle['sweep/config'], 'neural_weight', .05)
    loaded = load_sweep(path)
    assert not hasattr(loaded.config, 'neural_weight')
    assert historical in loaded.events
    for name, array in sweep.samples[0].candidates.fields.items():
        np.testing.assert_array_equal(loaded.samples[0].candidates.fields[name], array)
    np.testing.assert_array_equal(loaded.samples[0].candidate_indices, sweep.samples[0].candidate_indices)
    np.testing.assert_allclose(loaded.export()[0].fields['Ey'], sweep.export()[0].fields['Ey'])
    clean = tmp_path/'current.h5'
    loaded.save(clean)
    with h5py.File(clean, 'r') as handle:
        assert handle.attrs['schema'] == '1.1'
        assert 'neural_weight' not in handle['sweep/config']
    assert load_sweep(clean).config == loaded.config


def test_current_schema_does_not_ignore_unknown_controls(tmp_path):
    import h5py
    from fdfd_common.persistence import write_value
    from fdfd_common.errors import PersistenceError
    sweep = track_modes(slab, [.8*FC], port=PortSpec(boundary='enclosed'),
                        config=config(polarization='TE'), progress=False)
    path = tmp_path/'invalid.h5'
    sweep.save(path)
    with h5py.File(path, 'r+') as handle:
        write_value(handle['sweep/config'], 'unknown_control', 1.)
    with pytest.raises(PersistenceError):
        load_sweep(path)


def test_tm_cutoff_and_physical_maxwell_ratio():
    cfg = config(polarization='TM', neff_guess=-.7j)
    sweep = track_modes(slab, [.8*FC, 1.2*FC], port=PortSpec(boundary='enclosed'), config=cfg)
    modes = sweep.export()
    for mode in modes:
        n = mode.beta/(2*np.pi*mode.frequency/C)
        expected_squared = 1-(FC/mode.frequency)**2
        assert abs(n*n-expected_squared) < .001
        np.testing.assert_allclose(mode.fields['Ex'], n*mode.metadata['eta0']*mode.fields['Hy'], atol=1e-9)


def test_te_maxwell_ratio_and_no_raw_mutation():
    solver = slab(.8*FC)
    result = solver.solve(num_modes=2, polarization='TE', neff_guess=-.7j)
    before = {n: v.copy() for n, v in result.fields.items()}
    candidate = candidates_from_result(result, config())
    np.testing.assert_allclose(candidate.fields['Hx'],
        -result.neff*candidate.fields['Ey']/result.metadata['eta0'], atol=1e-12)
    for n in before: np.testing.assert_array_equal(result.fields[n], before[n])


def test_open_box_modes_fail_domain_verification():
    def box(f, spec):
        width = .02
        n = 100
        pad = round(n*spec.padding_fraction)
        s = ModeSolver1D(frequency=f, x_range=(-pad*width/n, width+pad*width/n))
        s.mesh(resolution=(n+2*pad)*spec.mesh_factor)
        return s
    sweep = track_modes(box, [10e9, 10.1e9], port=PortSpec(boundary='open'),
                        config=config(neff_guess=.8, polarization='TE'))
    assert not sweep.samples[0].candidates.eligible.any()
    assert all(c == 'radiation_or_box_suspect' for c in sweep.samples[0].candidates.confinement)
    with pytest.raises(ValueError, match='not injection eligible'): sweep.export()
    figure = sweep.plot(component='Ey')
    viewer = figure._mode_tracking_viewer
    assert all(viewer.status(0, i)['invalid'] for i in range(len(sweep.samples[0].candidates.result)))
    assert any(line.get_marker() == 'x' for axis in (viewer.phase_axis, viewer.decay_axis)
               for line in axis.lines)
    assert all(not np.isfinite(line.get_ydata()).any() for line in viewer.phase_axis.lines
               if line.get_label() == 'track 0')
    assert len(_dispersion_edges(viewer, style='--')) == 1


def test_square_degeneracy_requires_explicit_excitation():
    def square(f, spec):
        width, n = .02, 20
        dx = width/n
        bounds = ((-dx, width+dx),)*2
        s = ModeSolver2D(frequency=f, x_range=bounds[0], y_range=bounds[1])
        wall = shapes.Difference(shape=shapes.Rectangle(bounds=bounds),
                                  tool=shapes.Rectangle(bounds=((0., width),)*2))
        s.add_geometry(shape=wall, material=materials.PEC)
        s.mesh(resolution=((n+2)*spec.mesh_factor,)*2, subpixels=1)
        return s
    sweep = track_modes(square, [6e9, 6.1e9], port=PortSpec(boundary='enclosed'),
                        config=config(num_candidates=3, max_candidates=5, verification_overlap=.99))
    assert len(sweep.metadata['seed_modes']) == 2
    assert all(s.clusters for s in sweep.samples)
    with pytest.raises(ValueError, match='subspace'): sweep.export()
    terms = export_subspace(sweep, 6e9, 0, [1., .2j])
    assert len(terms) == 2
    assert all(t.metadata['representation'] == 'eigenmode_superposition_term' for t in terms)
    viewer = sweep.plot(component='E')._mode_tracking_viewer
    assert viewer.status(0, 0)['degenerate']


def test_budget_exhaustion_is_explicit():
    sweep = track_modes(slab, [.8*FC, .9*FC], port=PortSpec(boundary='enclosed'),
                        config=config(polarization='TE', max_solves=2))
    assert sweep.metadata['eigensolves'] == 2
    assert any(e['type'] == 'solve_unresolved' for e in sweep.events)
    with pytest.raises(ValueError, match='No validated sample'): sweep.export()


def test_avoided_crossing_follows_rotating_eigenbranch():
    cfg = config()
    previous = np.eye(2, dtype=complex)
    for angle in np.linspace(.05, np.pi/2, 25):
        basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], complex)
        permutation = [1, 0]
        indices, _, _ = assign_modes(previous, basis[:, permutation],
            np.array([-1., 1.]), np.array([1., -1.]), cfg)
        np.testing.assert_array_equal(indices, [1, 0])
        previous = basis


def test_open_bound_slab_passes_independent_verification():
    from fdfd_common import Material
    def factory(f, spec):
        spacing, cells = .0002, 200
        padding = round(cells*spec.padding_fraction)
        extent = cells*spacing/2+padding*spacing
        solver = ModeSolver1D(frequency=f, x_range=(-extent, extent))
        solver.add_layer(x_range=(-.003, .003), material=Material(name='core', epsilon=4.))
        solver.mesh(resolution=(cells+2*padding)*spec.mesh_factor, subpixels=1)
        return solver
    sweep = track_modes(factory, [20e9], config=config(polarization='TE', neff_guess=1.8))
    assert sweep.samples[0].candidates.eligible[0]
    assert sweep.export()[0].metadata['confinement'] == 'bound'


def test_material_first_sweep_api_and_gui_show_every_candidate(monkeypatch):
    dx = WIDTH/80
    tracker = ModeTracker1D(frequencies=[.8*FC, .9*FC], x_range=(-dx, WIDTH+dx),
                            port=PortSpec(boundary='enclosed', name='TE sweep'))
    left = tracker.add_layer(x_range=(-dx, 0.), material=materials.PEC, name='left')
    tracker.add_layer(x_range=(WIDTH, WIDTH+dx), material=materials.PEC, name='right')
    tracker.mesh(resolution=82, subpixels=1)
    sweep = tracker.solve(num_modes=3, polarization='TE',
                          tracking_config=config(polarization='TE'))
    assert tracker.result is sweep
    assert len(sweep.samples[0].candidates.result) == 3
    assert sweep.config.num_candidates == sweep.config.max_candidates == 3
    assert sweep.metadata['automatic_tracking']
    for sample in sweep.samples:
        assert len(sample.candidates.result) == 3
        assert sample.candidates.result.metadata['neff_guess'] == 1.
        assert set(sample.candidate_indices) == {0, 1, 2}
    from matplotlib import pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda **kwargs: None)
    figure = tracker.show(component='Ey', block=False)
    viewer = figure._mode_tracking_viewer
    assert sum(axis.get_visible() for axis in viewer.field_axes) == 3
    viewer.select_frequency(1)
    assert viewer.sample_index == 1
    assert 'all solved modes' in figure._suptitle.get_text()
    tracker.set_material(geometry=left, material=materials.PMC)
    assert tracker.result is None


def test_material_search_guess_includes_background_and_anisotropic_regions():
    from fdfd_common import Material
    from fdfd_mode_tracking.adapter import material_index_guess
    solver = ModeSolver1D(frequency=10e9, x_range=.02,
                          background_material=Material(epsilon=4., mu=2.))
    assert material_index_guess(solver) == pytest.approx(np.sqrt(8.))
    solver.add_layer(x_range=(.004, .016),
                     material=Material(epsilon=(2., 9., 3.), mu=(4., 1., 1.)))
    solver.add_layer(x_range=(0., .001), material=materials.PEC)
    assert material_index_guess(solver) == pytest.approx(6.)
    lossy = ModeSolver1D(frequency=10e9, x_range=.02,
                         background_material=Material(epsilon=4.-.2j))
    assert material_index_guess(lossy) == pytest.approx(np.sqrt(4.-.2j))


@pytest.mark.parametrize('reference', [0, 1])
def test_automatic_births_gaps_and_reacquisition(tmp_path, reference):
    # A controlled candidate window of actual TE eigenfields: TE3 enters at
    # the second sample; TE1 disappears and returns at the third sample.
    frequencies = np.array([3.2, 3.3, 3.4, 3.5])*FC
    windows = [(0, 1), (1, 2), (0, 2), (0, 2)]

    def factory(f, spec):
        solver = slab(f, spec)
        original = solver.solve

        def solve(**kwargs):
            result = original(**{**kwargs, 'num_modes': 3})
            order = np.argsort(-result.neff.real)
            window = windows[int(np.argmin(abs(frequencies-f)))]
            chosen = order[list(window)]
            metadata = dict(result.metadata)
            metadata['solve_info'] = {key: np.asarray(value)[chosen]
                                      if np.asarray(value).shape == (3,) else value
                                      for key, value in result.solve_info.items()}
            for key in ('polarizations', 'source_indices'):
                metadata[key] = tuple(metadata[key][i] for i in chosen)
            return replace(result, neff=result.neff[chosen], metadata=metadata,
                           fields={key: value[..., chosen] for key, value in result.fields.items()})

        solver.solve = solve
        return solver

    sweep = track_modes(factory, frequencies, seed_modes=None,
                        reference_frequency=frequencies[reference],
                        port=PortSpec(boundary='enclosed'),
                        config=config(polarization='TE', max_depth=0))
    assert all(len(sample.candidates.result) == 2 for sample in sweep.samples)
    assert all(len(sample.candidate_indices) == 3 for sample in sweep.samples)
    for sample in sweep.samples:
        assert set(sample.candidate_indices) == {-1, 0, 1}
    first, second, third, fourth = sweep.samples
    te1 = int(np.flatnonzero(first.candidate_indices == 0)[0])
    te2 = int(np.flatnonzero(first.candidate_indices == 1)[0])
    te3 = int(np.flatnonzero(second.candidate_indices == 1)[0])
    assert second.candidate_indices[te1] == -1
    assert third.candidate_indices[te1] == fourth.candidate_indices[te1] == 0
    assert second.candidate_indices[te2] == 0
    assert third.candidate_indices[te3] == fourth.candidate_indices[te3] == 1
    figure = sweep.plot(component='Ey')
    viewer = figure._mode_tracking_viewer
    for style in ('-', '--'):
        for lo, hi in _dispersion_edges(viewer, track=te1, style=style):
            assert not lo <= second.frequency/1e9 <= hi  # Never bridge absence.
    sweep.save(tmp_path/'births.h5')
    loaded = load_sweep(tmp_path/'births.h5')
    for old, new in zip(sweep.samples, loaded.samples):
        np.testing.assert_array_equal(old.candidate_indices, new.candidate_indices)


def test_automatic_tracking_can_start_with_no_valid_seed(monkeypatch):
    import fdfd_mode_tracking.sweep as implementation
    original = implementation.candidates_from_result

    def invalidate_first(result, cfg):
        candidates = original(result, cfg)
        if result.frequency == .8*FC:
            candidates.numerical_valid[:] = False
        return candidates

    monkeypatch.setattr(implementation, 'candidates_from_result', invalidate_first)
    sweep = track_modes(slab, [.8*FC, .9*FC], seed_modes=None,
                        port=PortSpec(boundary='enclosed'), config=config(polarization='TE'))
    assert np.all(sweep.samples[0].candidate_indices == -1)
    assert set(sweep.samples[-1].candidate_indices) == {0, 1}
    assert len(sweep.samples[0].candidate_indices) == 2


def test_automatic_square_spectrum_retains_fourfold_degeneracy():
    from fdfd_mode_tracking.assignment import eigen_clusters
    width, cells = .02, 20
    dx = width/cells
    bounds = ((-dx, width+dx),)*2
    tracker = ModeTracker2D(frequencies=[9e9, 9.1e9], x_range=bounds[0], y_range=bounds[1],
                            port=PortSpec(boundary='enclosed'))
    tracker.add_geometry(shape=shapes.Difference(
        shape=shapes.Rectangle(bounds=bounds),
        tool=shapes.Rectangle(bounds=((0., width),)*2)), material=materials.PEC)
    tracker.mesh(resolution=(cells+2, cells+2), subpixels=1)
    sweep = tracker.solve(num_modes=12, tracking_config=config(max_depth=1))
    # TE21/TM21 and their axis-swapped partners span a fourfold eigenspace.
    # A narrow ARPACK search can omit a member and return a farther mode, which
    # makes even whole-subspace matching fail as the sampled span rotates.
    assert len(sweep.samples) == 2
    assert sweep.metadata['num_tracks'] == 12
    assert not any(e['type'] == 'branch_discovered' for e in sweep.events)
    for sample in sweep.samples:
        groups = eigen_clusters(sample.candidates.eigenvalues, sweep.config.cluster_gap)
        assert [len(group) for group in groups] == [2, 2, 2, 4, 2]
        assert set(sample.candidate_indices) == set(range(12))
        assert len(sample.candidates.result) == 12
        assert [len(c['tracks']) for c in sample.clusters] == [2, 2, 2, 4, 2]


def test_material_first_2d_geometry_helpers_and_validation():
    tracker = ModeTracker2D(frequencies=[10e9], x_range=.02, y_range=.01)
    tracker.add_rectangle(x_range=(.005, .015), y_range=(.003, .007),
                          material=materials.air, name='rectangle')
    tracker.add_circle(center=(.01, .005), radius=.001, material=materials.air, name='circle')
    tracker.add_polygon(points=((.001, .001), (.002, .001), (.0015, .002)),
                        material=materials.air, name='polygon')
    mesh = tracker.mesh(max_element_size=.001, subpixels=2)
    assert mesh.resolution == (20, 10)
    with pytest.raises(ConfigurationError, match='positive integer'):
        tracker.solve(num_modes=0)


def test_cutoff_gui_marks_bracket_endpoints():
    sweep = track_modes(rectangle, [.8*FC, 1.2*FC], port=PortSpec(boundary='enclosed'), config=config())
    viewer = sweep.plot(component='Hz', quantity='real')._mode_tracking_viewer
    marked = [(si, ci) for si, sample in enumerate(sweep.samples)
              for ci in range(len(sample.candidates.result)) if viewer.status(si, ci)['cutoff']]
    assert marked
    assert any(line.get_marker() == '*' for axis in (viewer.phase_axis, viewer.decay_axis)
               for line in axis.lines)


@pytest.fixture
def progress_bars(monkeypatch):
    import fdfd_mode_tracking.sweep as implementation
    bars = []

    class RecordingBar:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.n = 0
            self.total = None
            self.stages = []
            self.closed = False
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def refresh(self):
            pass

        def set_postfix(self, **kwargs):
            self.stages.append(kwargs)

        def update(self, amount):
            self.n += amount
            assert self.n <= self.total

    monkeypatch.setattr(implementation, 'tqdm', RecordingBar)
    return bars


@pytest.mark.parametrize('boundary,per_frequency', [('enclosed', 2), ('open', 4)])
def test_progress_counts_frequencies_not_verification_solves(progress_bars, boundary, per_frequency):
    factory = slab
    if boundary == 'open':
        tracker = ModeTracker1D(frequencies=[.8*FC, .9*FC], x_range=(0., WIDTH))
        tracker.mesh(resolution=40)
        factory = tracker._factory  # Unlike slab(), this honors padding requests.
    sweep = track_modes(factory, [.8*FC, .9*FC], seed_modes=None,
                        port=PortSpec(boundary=boundary),
                        config=config(polarization='TE', max_depth=0))
    bar, = progress_bars
    assert bar.n == bar.total == 2
    assert bar.closed and not bar.options['disable']
    assert sweep.metadata['eigensolves'] == per_frequency*2
    stages = {status['stage'] for status in bar.stages}
    assert {'primary', 'mesh x2', 'checked'} <= stages
    if boundary == 'open':
        assert {'padding 25%', 'padding 50%'} <= stages


def test_progress_adaptive_total_and_cached_frequencies(progress_bars):
    frequencies_solved = []

    def factory(f, spec):
        if spec == VerificationSpec():
            frequencies_solved.append(f)
        return slab(f, spec)

    sweep = track_modes(factory, [.8*FC, 1.2*FC], seed_modes=None,
                        port=PortSpec(boundary='enclosed'),
                        config=config(polarization='TE', max_depth=2))
    bar, = progress_bars
    assert len(set(frequencies_solved)) > 2
    assert bar.n == bar.total == len(set(frequencies_solved))
    assert bar.closed
    assert any(event['type'] == 'refine_cutoff' for event in sweep.events)


@pytest.mark.parametrize('error', [RuntimeError('test failure'), KeyboardInterrupt()])
def test_progress_closes_on_failure_or_interrupt(progress_bars, error):
    def broken(f, spec):
        raise error

    with pytest.raises(type(error)):
        track_modes(broken, [.8*FC], config=config())
    bar, = progress_bars
    assert bar.closed and bar.n == 0


def test_progress_budget_does_not_pretend_completion(progress_bars):
    sweep = track_modes(slab, [.8*FC, .9*FC], seed_modes=None,
                        port=PortSpec(boundary='enclosed'),
                        config=config(polarization='TE', max_solves=2))
    bar, = progress_bars
    assert bar.closed and bar.n == 1 and bar.total == 2
    assert sweep.unresolved_intervals


def test_progress_can_be_silenced(capsys):
    track_modes(slab, [.8*FC], port=PortSpec(boundary='enclosed'),
                config=config(polarization='TE'), progress=False)
    captured = capsys.readouterr()
    assert captured.err == ''


@pytest.mark.parametrize('tracker_type', [ModeTracker1D, ModeTracker2D])
def test_material_first_api_forwards_progress(monkeypatch, tracker_type):
    import fdfd_mode_tracking.api as implementation
    options = {}
    marker = object()

    def fake_track(*args, **kwargs):
        options.update(kwargs)
        return marker

    monkeypatch.setattr(implementation, 'track_modes', fake_track)
    kwargs = {'y_range': (0., WIDTH)} if tracker_type is ModeTracker2D else {}
    tracker = tracker_type(frequencies=[FC], x_range=(0., WIDTH), **kwargs)
    assert tracker.solve(progress=False) is marker
    assert options['progress'] is False


def _dispersion_edges(viewer, *, track=0, style='-', axis=None):
    axis = viewer.phase_axis if axis is None else axis
    label = f'track {track}' + (' (ineligible)' if style == '--' else '')
    line = next(line for line in axis.lines if line.get_label() == label)
    assert line.get_linestyle() == style
    return [tuple(chunk[:2]) for chunk in np.asarray(line.get_xdata()).reshape(-1, 3)]


@pytest.fixture
def display_sweep():
    sweep = track_modes(slab, np.linspace(.8*FC, .9*FC, 5),
                        port=PortSpec(boundary='enclosed'), progress=False,
                        config=config(num_candidates=1, max_candidates=1, polarization='TE'))
    assert not sweep.unresolved_intervals
    assert all(sample.candidates.eligible[0] for sample in sweep.samples)
    yield sweep
    from matplotlib import pyplot as plt
    plt.close('all')


@pytest.mark.parametrize('confinement', ['radiation_or_box_suspect', 'unresolved'])
def test_ineligible_tracks_are_dashed_and_keep_export_guard(display_sweep, confinement, tmp_path):
    sweep = display_sweep
    original_ids = [sample.candidate_indices.copy() for sample in sweep.samples]
    for sample in sweep.samples[:2]:
        sample.candidates.confinement = (confinement,)
    frequencies = [sample.frequency/1e9 for sample in sweep.samples]
    viewer = sweep.plot(component='Ey')._mode_tracking_viewer
    for axis in (viewer.phase_axis, viewer.decay_axis):
        assert _dispersion_edges(viewer, style='--', axis=axis) == [
            tuple(frequencies[0:2]), tuple(frequencies[1:3])]
        assert _dispersion_edges(viewer, style='-', axis=axis) == [
            tuple(frequencies[2:4]), tuple(frequencies[3:5])]
        solid = next(line for line in axis.lines if line.get_label() == 'track 0')
        dashed = next(line for line in axis.lines if line.get_label() == 'track 0 (ineligible)')
        assert solid.get_color() == dashed.get_color()
        assert sum(line.get_marker() == 'x' for line in axis.lines) == 2
    assert 'NON-BOUND' in viewer.field_axes[0].get_title()
    assert any(text.get_text() == '×' for text in viewer.field_axes[0].texts)
    assert viewer.status(0, 0)['tracking_valid']
    assert not viewer.status(0, 0)['injection_eligible']
    for sample, ids in zip(sweep.samples, original_ids):
        np.testing.assert_array_equal(sample.candidate_indices, ids)
    with pytest.raises(ValueError, match='not injection eligible'):
        sweep.export(frequencies=[sweep.samples[0].frequency])
    assert sweep.export(frequencies=[sweep.samples[-1].frequency])
    sweep.save(tmp_path/'diagnostic.h5')
    restored = load_sweep(tmp_path/'diagnostic.h5').plot(component='Ey')._mode_tracking_viewer
    assert _dispersion_edges(restored, style='--') == _dispersion_edges(viewer, style='--')


@pytest.mark.parametrize('count', [0, 1, 12, 20, 32, 100])
def test_track_palette_does_not_repeat_or_change_prefix(count):
    from matplotlib.colors import is_color_like
    from fdfd_mode_tracking.visualization import _track_colors

    colors = _track_colors(count)
    assert len(colors) == len(set(colors)) == count
    assert all(is_color_like(color) for color in colors)
    assert colors == _track_colors(count+10)[:count]


def test_track_colors_follow_global_identity_and_survive_reload(display_sweep, tmp_path):
    sweep = display_sweep
    # One candidate at each frequency; a later branch has a high global ID.
    # Missing IDs deliberately exercise palettes beyond tab10 and tab20.
    tracks = [0, 0, 31, 31, 31]
    for sample, track in zip(sweep.samples, tracks):
        sample.candidate_indices = np.full(32, -1, dtype=int)
        sample.candidate_indices[track] = 0
        sample.phases = np.ones(32, dtype=complex)
        sample.overlaps = np.ones(32)
    sweep.samples[2].candidates.confinement = ('unresolved',)
    viewer = sweep.plot(component='Ey')._mode_tracking_viewer
    assert len(set(viewer.track_colors)) == 32
    for axis in (viewer.phase_axis, viewer.decay_axis):
        for track, color in enumerate(viewer.track_colors):
            for label in (f'track {track}', f'track {track} (ineligible)'):
                line = next(line for line in axis.lines if line.get_label() == label)
                assert line.get_color() == color
        markers = [line for line in axis.lines if line.get_picker() == 5]
        assert [line.get_color() for line in markers] == [viewer.track_colors[t] for t in tracks]
        warnings = [line for line in axis.lines if line.get_marker() == 'x']
        assert len(warnings) == 1 and warnings[0].get_color() == 'red'
    for si, track in enumerate(tracks):
        viewer.select_frequency(si)
        assert viewer.field_axes[0].lines[0].get_color() == viewer.track_colors[track]
    sweep.save(tmp_path/'colors.h5')
    restored = load_sweep(tmp_path/'colors.h5').plot(component='Ey')._mode_tracking_viewer
    assert restored.track_colors == viewer.track_colors


@pytest.mark.parametrize('gap_kind', ['numerical', 'absent', 'assignment', 'reverse', 'skipped_solve'])
def test_diagnostic_lines_do_not_bridge_unreliable_identity(display_sweep, gap_kind):
    sweep = display_sweep
    for sample in sweep.samples:
        sample.candidates.confinement = ('radiation_or_box_suspect',)
    frequencies = [sample.frequency/1e9 for sample in sweep.samples]
    expected = list(zip(frequencies[:-1], frequencies[1:]))
    if gap_kind in ('numerical', 'absent'):
        if gap_kind == 'numerical':
            sweep.samples[2].candidates.numerical_valid[0] = False
        else:
            sweep.samples[2].candidate_indices[0] = -1
        expected = [expected[0], expected[3]]
    else:
        interval = (sweep.samples[1].frequency, sweep.samples[2].frequency)
        sweep.unresolved_intervals = (interval,)
        if gap_kind != 'skipped_solve':
            event_type = 'assignment_unresolved' if gap_kind == 'assignment' else 'reverse_audit_unresolved'
            sweep.events += ({'type': event_type, 'interval': interval},)
        expected.pop(1)
    viewer = sweep.plot(component='Ey')._mode_tracking_viewer
    for axis in (viewer.phase_axis, viewer.decay_axis):
        assert _dispersion_edges(viewer, style='--', axis=axis) == expected
        assert _dispersion_edges(viewer, style='-', axis=axis) == []


def test_cutoff_bracket_is_not_an_identity_gap(display_sweep):
    sweep = display_sweep
    interval = (sweep.samples[1].frequency, sweep.samples[2].frequency)
    sweep.unresolved_intervals = (interval,)
    sweep.events += ({'type': 'cutoff_bracket', 'interval': interval, 'tracks': (0,)},)
    viewer = sweep.plot(component='Ey')._mode_tracking_viewer
    assert len(_dispersion_edges(viewer)) == 4
    # A reverse-audit failure still takes precedence if it shares that bracket.
    sweep.events += ({'type': 'reverse_audit_unresolved', 'interval': interval},)
    viewer = sweep.plot(component='Ey')._mode_tracking_viewer
    assert len(_dispersion_edges(viewer)) == 3


@pytest.mark.parametrize('confinement', ['bound', 'unresolved'])
def test_reliable_exact_cutoff_keeps_spectral_line_not_export(display_sweep, confinement):
    sample = display_sweep.samples[2]
    sample.candidates.propagation = ('cutoff_unresolved',)
    sample.candidates.confinement = (confinement,)
    sample.candidates.numerical_valid[0] = False
    sample.candidates.result = replace(sample.candidates.result, neff=np.array([0j]))
    viewer = display_sweep.plot(component='Ey')._mode_tracking_viewer
    assert viewer.status(2, 0)['tracking_valid']
    assert len(_dispersion_edges(viewer, style='--')) == 2
    assert len(_dispersion_edges(viewer)) == 2
    assert viewer.status(2, 0)['invalid'] == (confinement != 'bound')
    with pytest.raises(ValueError, match='not injection eligible'):
        display_sweep.export(frequencies=[sample.frequency])
