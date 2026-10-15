"""Grid convergence against the validated FEM grounded-slab periodic cell.

Stored FEM references use first-order triangles: 50e-6 m maximum edge at
20e9 Hz for the antenna, and 100e-6 m for the other comparisons. All use
max_refinements=0, TM, PEC walls and x+ PML. No FEM installation is
required to run the FDFD comparison; see docs/solvers/periodic.rst.

    python benchmarks/periodic/leaky_wave_fem_reference.py --check
    python benchmarks/periodic/leaky_wave_fem_reference.py --case slab --frequencies 20e9 --check
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

import fdfd
from fdfd import materials

FEM_REFERENCE = np.array(
    [
        -0.01454590826862274 - 0.09323259689950363j,
        0.01454590826862245 + 0.09323259689950367j,
        0.5792212652919826 - 0.5252337359649764j,
        -0.5792212652919808 + 0.5252337359649766j,
    ]
)
FEM_REFERENCES = {
    "antenna": {
        18e9: np.array(
            [
                -0.41798552756884316 - 0.02520160351518428j,
                0.4179855275688467 + 0.025201603515184426j,
                0.9160543684780482 - 0.07654551260303112j,
                -0.9160543684780493 + 0.07654551260303151j,
            ]
        ),
        20e9: FEM_REFERENCE,
        22e9: np.array(
            [
                -0.4316608264408845 + 0.012947476542689419j,
                0.43166082644088466 - 0.012947476542689105j,
                -0.740619221038214 - 0.024576247156232158j,
                0.7406192210382201 + 0.024576247156232633j,
            ]
        ),
    },
    "slab": {
        20e9: np.array(
            [
                -0.06546748694914711 - 7.102375504829061e-6j,
                0.06546748694914724 + 7.1023755047032695e-6j,
                -0.5576661675321186 + 0.5883053964858241j,
                0.5576661675321186 - 0.5883053964858249j,
            ]
        ),
    },
}
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem


def compare(resolutions, frequencies=(20e9,), case="antenna"):
    rows = []
    jobs = [(frequency, nx, nz) for frequency in frequencies for nx, nz in resolutions]
    for frequency, nx, nz in tqdm(jobs, desc="FEM comparison", unit="solve"):
        reference = FEM_REFERENCES[case][frequency]
        solver = fdfd.PeriodicModeSolver2D(
            frequency=frequency, x_range=(0.0, 10e-3), z_range=(0.0, 8e-3), boundary=materials.PEC
        )
        solver.add_rectangle(
            x_range=(0.0, 1.27e-3),
            z_range=(0.0, 8e-3),
            material=fdfd.Material(name="antenna substrate", epsilon=10.2),
        )
        if case == "antenna":
            solver.add_rectangle(x_range=(1.27e-3, 1.32e-3), z_range=(1e-3, 2e-3), material=materials.PEC)
        solver.add_pml(thickness=2.5e-3, sides="x+", order=3, strength=5.0)
        solver.mesh(resolution=(nx, nz))
        result = solver.solve(num_modes=4, polarization="TM", guess=0.0, tol=1e-9, ncv=36)
        reference_indices, calculated_indices = linear_sum_assignment(
            np.abs(reference[:, None] - result.neff[None, :])
        )
        for reference_index, calculated_index in zip(reference_indices, calculated_indices, strict=True):
            expected, value = reference[reference_index], result.neff[calculated_index]
            rows.append(
                dict(
                    case=case,
                    frequency_hz=frequency,
                    nx=nx,
                    nz=nz,
                    dx_m=10e-3 / nx,
                    dz_m=8e-3 / nz,
                    mode=reference_index + 1,
                    neff_real=value.real,
                    neff_imag=value.imag,
                    fem_neff_real=expected.real,
                    fem_neff_imag=expected.imag,
                    relative_complex_error=abs(value - expected) / abs(expected),
                    backward_error=result.info.residuals[calculated_index],
                )
            )
        worst = max(r["relative_complex_error"] for r in rows[-4:])
        tqdm.write(f"{case} at {frequency / 1e9:g} GHz, {nx} x {nz}: maximum complex-neff error {worst:.2%}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", default=["1000x320"])
    parser.add_argument("--frequencies", nargs="+", type=float, default=[18e9, 20e9, 22e9])
    parser.add_argument("--case", choices=tuple(FEM_REFERENCES), default="antenna")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Require <3%% complex-neff error on the finest grid at every frequency.",
    )
    args = parser.parse_args()
    resolutions = [tuple(map(int, grid.lower().split("x"))) for grid in args.grids]
    if any(len(pair) != 2 or min(pair) < 2 for pair in resolutions):
        parser.error("Use grid sizes such as 1000x320.")
    if any(f not in FEM_REFERENCES[args.case] for f in args.frequencies):
        parser.error("No stored FEM reference for this case/frequency; slab supports 20e9.")
    rows = compare(resolutions, args.frequencies, args.case)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    for frequency in args.frequencies:
        for mode in range(1, 5):
            selected = [row for row in rows if row["mode"] == mode and row["frequency_hz"] == frequency]
            axis.loglog(
                [r["dx_m"] * 1e6 for r in selected],
                [r["relative_complex_error"] for r in selected],
                "o-",
                label=f"{frequency / 1e9:g} GHz, mode {mode}",
            )
    axis.set(
        xlabel="x cell width (um)",
        ylabel="Relative complex-neff error against FEM",
        title=f"Periodic {args.case}: FDFD vs FEM",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "convergence.png", dpi=160)
    if args.check:
        finest = max(resolutions, key=lambda pair: pair[0] * pair[1])
        selected = [r for r in rows if (r["nx"], r["nz"]) == finest]
        if any(r["relative_complex_error"] > 0.03 or r["backward_error"] > 1e-6 for r in selected):
            raise SystemExit("FDFD did not meet the FEM comparison tolerance; refine the grid.")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
