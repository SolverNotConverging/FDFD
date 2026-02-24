from Periodic_Solver_3D import PeriodicModeSolver3D  # adjust to your module name

# Later / elsewhere — reload and plot immediately
solver = PeriodicModeSolver3D.load_results("modes_full.npz")
solver.visualize_with_gui()
