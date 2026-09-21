"""Interactive dispersion and all-candidate field viewer."""
import numpy as np
from .assignment import eigen_clusters


class ModeTrackingViewer:
    """Matplotlib controller retained on the returned Figure for interaction."""
    def __init__(self, sweep, *, component='E', quantity='magnitude'):
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.widgets import RadioButtons, Slider

        if component not in ('E', 'H', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
            raise ValueError('component must be E, H, or a Cartesian field component.')
        if quantity not in ('real', 'imag', 'magnitude', 'phase'):
            raise ValueError('quantity must be real, imag, magnitude, or phase.')
        self.sweep = sweep
        self.component = component
        self.quantity = quantity
        self.sample_index = 0
        self.frequencies = np.array([sample.frequency for sample in sweep.samples])
        max_modes = max(len(sample.candidates.result) for sample in sweep.samples)
        rows, columns = (1, max_modes) if max_modes <= 3 else (2, int(np.ceil(max_modes/2)))
        self.figure = plt.figure(figsize=(max(11, 3.1*columns), 5.2+2.5*rows))
        outer = self.figure.add_gridspec(2, 1, height_ratios=(1.25, 1.5*rows),
                                         left=.08, right=.98, top=.94, bottom=.18,
                                         hspace=.28)
        dispersion = outer[0].subgridspec(1, 2, wspace=.28)
        self.phase_axis = self.figure.add_subplot(dispersion[0, 0])
        self.decay_axis = self.figure.add_subplot(dispersion[0, 1], sharex=self.phase_axis)
        field_grid = outer[1].subgridspec(rows, columns, wspace=.28, hspace=.38)
        self.field_axes = [self.figure.add_subplot(field_grid[i//columns, i%columns])
                           for i in range(rows*columns)]
        self._cutoff = self._cutoff_lookup()
        self._draw_dispersion()
        self.selection_lines = [axis.axvline(self.frequencies[0]/1e9, color='black', lw=1, alpha=.55)
                                for axis in (self.phase_axis, self.decay_axis)]
        self.component_axis = self.figure.add_axes((.02, .02, .18, .12))
        self.quantity_axis = self.figure.add_axes((.21, .02, .18, .12))
        self.frequency_axis = self.figure.add_axes((.47, .065, .45, .035))
        self.component_control = RadioButtons(self.component_axis,
            ('E', 'H', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'),
            active=('E', 'H', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz').index(component))
        self.quantity_control = RadioButtons(self.quantity_axis,
            ('magnitude', 'real', 'imag', 'phase'),
            active=('magnitude', 'real', 'imag', 'phase').index(quantity))
        self.frequency_control = Slider(self.frequency_axis, 'solved frequency', 0,
                                        max(1, len(self.frequencies)-1), valinit=0, valstep=1)
        self.frequency_control.valtext.set_text(f'{self.frequencies[0]/1e9:.6g} GHz')
        if len(self.frequencies) == 1:
            self.frequency_control.set_active(False)
        self.component_control.on_clicked(self._set_component)
        self.quantity_control.on_clicked(self._set_quantity)
        self.frequency_control.on_changed(self._set_frequency)
        self.click_id = self.figure.canvas.mpl_connect('button_press_event', self._click)
        self.legend = self.phase_axis.legend(handles=[
            Line2D([], [], ls='-', color='tab:blue', label='injection eligible'),
            Line2D([], [], ls='--', color='tab:blue', label='tracked, not injection eligible'),
            Line2D([], [], marker='o', ls='', color='tab:blue', label='ordinary'),
            Line2D([], [], marker='D', ls='', color='tab:purple', label='degenerate'),
            Line2D([], [], marker='*', ls='', color='tab:orange', label='cutoff bracket'),
            Line2D([], [], marker='x', ls='', color='red', label='non-bound / spurious'),
        ], loc='best', fontsize='small')
        self._draw_fields()
        self.figure._mode_tracking_viewer = self

    def _cutoff_lookup(self):
        result = set()
        samples = {sample.frequency: sample for sample in self.sweep.samples}
        for event in self.sweep.events:
            if event.get('type') != 'cutoff_bracket':
                continue
            for frequency in event['interval']:
                sample = samples.get(frequency)
                if sample is None:
                    continue
                for track in event.get('tracks', ()):
                    if track < len(sample.candidate_indices) and sample.candidate_indices[track] >= 0:
                        result.add((frequency, int(sample.candidate_indices[track])))
        for sample in self.sweep.samples:
            for candidate, state in enumerate(sample.candidates.propagation):
                if state == 'cutoff_unresolved':
                    result.add((sample.frequency, candidate))
        return result

    def status(self, sample_index, candidate):
        sample = self.sweep.samples[sample_index]
        groups = eigen_clusters(sample.candidates.eigenvalues, self.sweep.config.cluster_gap)
        degenerate = any(candidate in group and len(group) > 1 for group in groups)
        cutoff = (sample.frequency, candidate) in self._cutoff
        residual = sample.candidates.evidence[candidate]['residual']
        cutoff_identity = (sample.candidates.propagation[candidate] == 'cutoff_unresolved'
                           and np.isfinite(residual)
                           and residual <= self.sweep.config.residual_tolerance)
        cutoff_bound = cutoff_identity and sample.candidates.confinement[candidate] == 'bound'
        invalid = not bool(sample.candidates.eligible[candidate]) and not cutoff_bound
        invalid_kind = None
        if invalid:
            invalid_kind = ('SPURIOUS/INVALID' if not sample.candidates.numerical_valid[candidate]
                            else 'NON-BOUND')
        return {'invalid': invalid, 'degenerate': degenerate, 'cutoff': cutoff,
                'invalid_kind': invalid_kind,
                'tracking_valid': bool(sample.candidates.numerical_valid[candidate] or cutoff_identity),
                'injection_eligible': bool(sample.candidates.eligible[candidate]),
                'propagation': sample.candidates.propagation[candidate],
                'confinement': sample.candidates.confinement[candidate]}

    def _marker(self, status):
        return '*' if status['cutoff'] else ('D' if status['degenerate'] else 'o')

    def _draw_dispersion(self):
        tracked_colors = [f'C{i%10}' for i in range(max(len(s.candidate_indices) for s in self.sweep.samples))]
        # Cutoff uncertainty concerns reconstruction, not necessarily identity.
        # Other unresolved intervals must not be bridged, including failed
        # primary solves that left no sample in the archive.
        cutoff_intervals = {tuple(e['interval']) for e in self.sweep.events
                            if e['type'] == 'cutoff_bracket'}
        identity_intervals = {tuple(interval) for interval in self.sweep.unresolved_intervals
                              if tuple(interval) not in cutoff_intervals}
        identity_intervals.update(tuple(e['interval']) for e in self.sweep.events
                                  if e['type'] in ('assignment_unresolved', 'reverse_audit_unresolved'))
        # One style per adjacent edge avoids gaps at a change of eligibility:
        # both eligible -> solid; either ineligible -> dashed. Never change IDs
        # or certification just to draw a connecting line.
        for track, color in enumerate(tracked_colors):
            values = np.full((len(self.sweep.samples), 2), np.nan)
            eligible = np.zeros(len(self.sweep.samples), dtype=bool)
            for si, sample in enumerate(self.sweep.samples):
                if track >= len(sample.candidate_indices) or sample.candidate_indices[track] < 0:
                    continue
                candidate = sample.candidate_indices[track]
                status = self.status(si, candidate)
                if not status['tracking_valid']:
                    continue
                neff = sample.candidates.result.neff[candidate]
                values[si] = (neff.real, -neff.imag)
                eligible[si] = status['injection_eligible']
            edges = {'-': ([], [], []), '--': ([], [], [])}
            for si in range(len(self.frequencies)-1):
                if not np.isfinite(values[si:si+2]).all():
                    continue
                lo, hi = self.frequencies[si:si+2]
                if any(max(lo, a) < min(hi, b) for a, b in identity_intervals):
                    continue
                style = '-' if eligible[si] and eligible[si+1] else '--'
                x, phase, decay = edges[style]
                x.extend((lo/1e9, hi/1e9, np.nan))
                phase.extend((*values[si:si+2, 0], np.nan))
                decay.extend((*values[si:si+2, 1], np.nan))
            for style, (x, phase, decay) in edges.items():
                label = f'track {track}' + (' (ineligible)' if style == '--' else '')
                self.phase_axis.plot(x, phase, style, color=color, lw=1.3, alpha=.75, label=label)
                self.decay_axis.plot(x, decay, style, color=color, lw=1.3, alpha=.75, label=label)
        # Mark every candidate returned at every primary/adaptive frequency.
        for si, sample in enumerate(self.sweep.samples):
            tracked = {int(candidate): track for track, candidate in enumerate(sample.candidate_indices)
                       if candidate >= 0}
            for candidate, neff in enumerate(sample.candidates.result.neff):
                status = self.status(si, candidate)
                color = tracked_colors[tracked[candidate]] if candidate in tracked else '.45'
                marker = self._marker(status)
                for axis, value in ((self.phase_axis, neff.real), (self.decay_axis, -neff.imag)):
                    axis.plot(sample.frequency/1e9, value, marker=marker, ls='', color=color,
                              markersize=8 if marker == '*' else 6, picker=5)
                    if status['invalid']:
                        axis.plot(sample.frequency/1e9, value, marker='x', ls='', color='red',
                                  mew=1.8, markersize=8)
        for lo, hi in self.sweep.unresolved_intervals:
            for axis in (self.phase_axis, self.decay_axis):
                axis.axvspan(lo/1e9, hi/1e9, color='tab:orange', alpha=.08)
        self.phase_axis.set(ylabel=r'phase index  Re($n_{eff}$)', title='Tracked mode dispersion')
        self.decay_axis.set(ylabel=r'decay/loss index  $-Im(n_{eff})$', title='Attenuation and evanescence')
        for axis in (self.phase_axis, self.decay_axis):
            axis.set_xlabel('frequency (GHz)')
            axis.grid(True, alpha=.25)

    def _field_values(self, sample, candidate):
        fields = sample.candidates.fields
        if self.component in fields:
            value = fields[self.component][..., candidate]
            coordinates = sample.candidates.result.field_coordinates[self.component]
            label = self.component
            shown_quantity = self.quantity
        else:
            names = tuple(name for name in fields if name.startswith(self.component))
            # Display a stagger-safe RMS aggregate at cell centres.
            from .metrics import cell_fields
            centred = cell_fields(sample.candidates.result, {name: fields[name] for name in names})
            value = np.sqrt(sum(abs(centred[name][..., candidate])**2 for name in names))
            coordinates = sample.candidates.result.mesh_data.coordinates
            label = '|'+self.component+'|'
            shown_quantity = 'magnitude'
        operation = {'real': np.real, 'imag': np.imag, 'magnitude': np.abs, 'phase': np.angle}[shown_quantity]
        return operation(value), coordinates, label, shown_quantity

    def _draw_fields(self):
        sample = self.sweep.samples[self.sample_index]
        tracked = {int(candidate): track for track, candidate in enumerate(sample.candidate_indices)
                   if candidate >= 0}
        for axis in self.field_axes:
            axis.clear()
            axis.set_visible(False)
        for candidate in range(len(sample.candidates.result)):
            axis = self.field_axes[candidate]
            axis.set_visible(True)
            values, coordinates, label, shown_quantity = self._field_values(sample, candidate)
            if values.ndim == 1:
                axis.plot(coordinates[0]*1e3, values, color=f'C{tracked.get(candidate, candidate)%10}')
                axis.set_xlabel(sample.candidates.result.mesh_data.axes[0]+' (mm)')
            else:
                axis.pcolormesh(coordinates[0]*1e3, coordinates[1]*1e3, values.T,
                                shading='auto', cmap='viridis')
                axis.set(xlabel=sample.candidates.result.mesh_data.axes[0]+' (mm)',
                         ylabel=sample.candidates.result.mesh_data.axes[1]+' (mm)', aspect='equal')
            status = self.status(self.sample_index, candidate)
            flags = []
            if status['degenerate']: flags.append('DEGENERATE')
            if status['cutoff']: flags.append('CUTOFF')
            if status['invalid']: flags.append(status['invalid_kind']+' ×')
            track = f' · track {tracked[candidate]}' if candidate in tracked else ''
            suffix = ' · '+' / '.join(flags) if flags else ''
            axis.set_title(f'mode {candidate}{track}\n{label} {shown_quantity}{suffix}', fontsize=9,
                           color='firebrick' if status['invalid'] else 'black')
            if status['invalid']:
                axis.text(.5, .5, '×', transform=axis.transAxes, ha='center', va='center',
                          fontsize=46, color='red', alpha=.55, weight='bold')
        frequency = sample.frequency/1e9
        self.figure.suptitle(f'{self.sweep.port.name}: all solved modes at {frequency:.6g} GHz', fontsize=12)
        for line in self.selection_lines:
            line.set_xdata([frequency, frequency])
        self.figure.canvas.draw_idle()

    def select_frequency(self, index):
        index = int(np.clip(round(index), 0, len(self.frequencies)-1))
        self.sample_index = index
        if int(round(self.frequency_control.val)) != index:
            self.frequency_control.set_val(index)
        else:
            self._draw_fields()

    def _set_frequency(self, value):
        self.sample_index = int(np.clip(round(value), 0, len(self.frequencies)-1))
        self.frequency_control.valtext.set_text(f'{self.frequencies[self.sample_index]/1e9:.6g} GHz')
        self._draw_fields()

    def _set_component(self, value):
        self.component = value
        self._draw_fields()

    def _set_quantity(self, value):
        self.quantity = value
        self._draw_fields()

    def _click(self, event):
        if event.inaxes not in (self.phase_axis, self.decay_axis) or event.xdata is None:
            return
        self.select_frequency(int(np.argmin(abs(self.frequencies/1e9-event.xdata))))


def plot_sweep(sweep, *, component='E', quantity='magnitude'):
    return ModeTrackingViewer(sweep, component=component, quantity=quantity).figure


def show_sweep(sweep, *, component='E', quantity='magnitude', block=True):
    from matplotlib import pyplot as plt
    if not isinstance(block, bool):
        raise ValueError('block must be a boolean.')
    figure = plot_sweep(sweep, component=component, quantity=quantity)
    plt.show(block=block)
    return figure
