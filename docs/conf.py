"""Sphinx configuration for the astroARIADNE documentation."""
import datetime
import os
import sys
from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Import the package from source so autodoc works without a full install
# (heavy/compiled deps are mocked below).
sys.path.insert(0, os.path.abspath('..'))

project = 'astroARIADNE'
author = 'Jose Vines'
copyright = f'{datetime.date.today().year}, {author}'

try:
    release = _pkg_version('astroARIADNE')
except PackageNotFoundError:
    release = '1.5.0'
version = '.'.join(release.split('.')[:2])

extensions = [
    'myst_parser',                 # Markdown (MyST) support
    'sphinx.ext.autodoc',          # API docs from docstrings
    'sphinx.ext.napoleon',         # NumPy/Google-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Markdown + reStructuredText.
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'

myst_enable_extensions = ['colon_fence', 'deflist', 'attrs_inline']
myst_heading_anchors = 3

# Don't choke the RTD build on the heavy / compiled scientific dependencies;
# autodoc only needs to import the modules to read their docstrings.
# numba is NOT mocked: isochrone.py registers a numba warning category at
# import time (which must be a real class), and numba ships binary wheels so
# RTD installs it cheaply.
autodoc_mock_imports = [
    'isochrones', 'dynesty', 'pyphot', 'dustmaps', 'astroquery',
    'extinction', 'regions', 'PyAstronomy', 'pymultinest', 'healpy',
    'pandas', 'astropy', 'matplotlib', 'termcolor', 'tqdm',
    'corner', 'tabulate', 'requests', 'arviz',
]
autodoc_default_options = {'members': True, 'undoc-members': True,
                           'show-inheritance': True}
autoclass_content = 'both'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

html_theme = 'sphinx_rtd_theme'
html_title = f'astroARIADNE {release}'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# README/CHANGELOG live at the repo root; MyST `{include}` pulls them in, so
# there's a single source of truth and no duplication.
suppress_warnings = ['myst.header', 'myst.xref_missing']
