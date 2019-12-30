"""Some configs for the code to run."""
import os
import inspect

from pkg_resources import resource_filename

gridsdir = resource_filename('astroARIADNE', 'Datafiles/model_grids')
priorsdir = resource_filename('astroARIADNE', 'Datafiles/prior')
filesdir = resource_filename('astroARIADNE', 'Datafiles/')

try:
    modelsdir = os.environ['ARIADNE_MODELS']
except KeyError:
    modelsdir = None
