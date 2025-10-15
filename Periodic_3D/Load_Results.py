from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver  # adjust to your module name

# Later / elsewhere — reload and plot immediately
solver = Periodic_3D_Mode_Solver.load_results("modes_full.npz")
solver.visualize_with_gui()
