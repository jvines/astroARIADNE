"""Some configs for the code to run."""
import inspect
import os
import numpy as np

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

# pyphot filter names
filter_names = np.array([
        '2MASS_H', '2MASS_J', '2MASS_Ks',
        'GROUND_COUSINS_I', 'GROUND_COUSINS_R',
        'GROUND_JOHNSON_U', 'GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
        'TYCHO_B_MvB', 'TYCHO_V_MvB',
        'STROMGREN_b', 'STROMGREN_u', 'STROMGREN_v', 'STROMGREN_y',
        'GaiaDR2v2_G', 'GaiaDR2v2_RP', 'GaiaDR2v2_BP',
        'PS1_g', 'PS1_i', 'PS1_r', 'PS1_w', 'PS1_y', 'PS1_z',
        'SDSS_g', 'SDSS_i', 'SDSS_r', 'SDSS_u', 'SDSS_z',
        'SkyMapper_u', 'SkyMapper_v', 'SkyMapper_g', 'SkyMapper_r',
        'SkyMapper_i', 'SkyMapper_z',
        'WISE_RSR_W1', 'WISE_RSR_W2',
        'GALEX_FUV', 'GALEX_NUV',
        'SPITZER_IRAC_36', 'SPITZER_IRAC_45',
        'NGTS_I', 'TESS', 'KEPLER_Kp'
    ])

# termcolor colors
colors = [
        'red', 'green', 'blue', 'yellow',
        'grey', 'magenta', 'cyan', 'white'
    ]

# Isochrone mask array
iso_mask = np.array([1, 1, 1,
                     0, 0,
                     1, 1, 1,
                     0, 0,
                     0, 0, 0, 0,
                     1, 1, 1,
                     0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0,
                     1, 1,
                     0, 0,
                     0, 0,
                     0, 1, 0])

# Isochrone bands array
iso_bands = [
            'H', 'J', 'K',
            'U', 'V', 'B',
            'G', 'RP', 'BP',
            'W1', 'W2',
            'TESS'
        ]
