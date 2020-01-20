"""Some configs for the code to run."""
import inspect
import os

from pkg_resources import resource_filename

__ROOT__ = '/'.join(os.path.abspath(inspect.getfile(inspect.currentframe())
                                    ).split('/')[:-1])
gridsdir = resource_filename('astroARIADNE', 'Datafiles/model_grids')
priorsdir = resource_filename('astroARIADNE', 'Datafiles/prior')
filesdir = resource_filename('astroARIADNE', 'Datafiles')

try:
    modelsdir = os.environ['ARIADNE_MODELS']
except KeyError:
    modelsdir = None
