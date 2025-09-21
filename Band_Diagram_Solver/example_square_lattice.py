from Band_Diagram import BandDiagramSolver2D

solver = BandDiagramSolver2D(a=1.0, Nx=40, background_er=10.2)
solver.add_circular_inclusion(radius=0.4, er=1.0)

points, labels = solver.default_rectangular_lattice_path()
beta_path, tick_positions = solver.generate_bloch_path(points, total_points=200)
solver.set_tick_labels(labels, tick_positions)

result = solver.compute_band_structure(beta_path, num_bands=5)

solver.plot_band_diagram(result, wnmax=0.6)
