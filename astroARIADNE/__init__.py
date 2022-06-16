"""
ARIADNE is a module to easily fit SED models using nested sampling algorithms.

It allows to fit single models (Phoenix v2, BT-Settl, BT-Cond, BT-NextGen
Castelli & Kurucz 2004 and Kurucz 1993) or multiple models in a single run.
If multiple models are fit for, then ARIADNE automatically averages the
parameters posteriors as in the Bayesian Model Average framework. This
averages over the models and thus the averaged posteriors account for model
specific uncertainties.
"""
from pkg_resources import get_distribution

from .fitter import Fitter
from .star import Star
from .plotter import SEDPlotter
from .config import __bibtex__

__all__ = [
    'Fitter',
    'Star',
    'SEDPlotter'
]

__author__ = 'Jose Ignacio Vines'
__email__ = 'jose.vines@ug.uchile.cl'
__license__ = 'MIT'
__description__ = 'Bayesian Model Averaging SED fitter'
__version__ = get_distribution('astroARIADNE').version
