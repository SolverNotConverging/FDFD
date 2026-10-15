"""Sphinx configuration. Build with: sphinx-build -W docs docs/_build/html"""

from importlib.metadata import version as _version

project = "fdfd"
author = "fdfd contributors"
release = _version("fdfd")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autosummary_generate = False

html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": "Finite-difference frequency-domain electromagnetics",
    "fixed_sidebar": True,
}
