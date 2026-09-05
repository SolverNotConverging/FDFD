"""Keep numerical and static plotting tests independent of a desktop session."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
