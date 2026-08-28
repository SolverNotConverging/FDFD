"""Launch the interactive FEM transmission-line calculator."""

from __future__ import annotations

from FEM_Mode_Solver import launch_transmission_line_calculator


def main() -> None:
    launch_transmission_line_calculator(show=True)


if __name__ == "__main__":
    main()
