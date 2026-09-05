from fdfd_periodic_modes import PeriodicModeSolver3D

# Later / elsewhere — reload and plot immediately
solver = PeriodicModeSolver3D.load_results("modes_full.npz")
solver.visualize_with_gui()
