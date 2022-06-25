"""Star.py contains the Star class which contains the data regarding a star."""

__all__ = ['Star']

import random

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from dustmaps.planck import PlanckQuery, PlanckGNILCQuery
from dustmaps.lenz2017 import Lenz2017Query
from dustmaps.bayestar import BayestarQuery
from termcolor import colored

from .config import gridsdir, filter_names, colors, iso_mask, iso_bands
from .isochrone import estimate
from .librarian import Librarian
from .error import StarWarning
from .phot_utils import *
from .utils import display_star_fin, display_star_init


def extract_from_lib(lib):
    """Extract relevant parameters from lib.

    Returns
    -------
    [plx, plx_e, dist, dist_e, rad, rad_e, temp, temp_e, lum, lum_e]
    """
    if lib is None:
        return [-1] * 10
    return [
        lib.plx, lib.plx_e,
        lib.dist, lib.dist_e,
        lib.rad, lib.rad_e,
        lib.temp, lib.temp_e,
        lib.lum, lib.lum_e
    ]


class Star:
    """Object that holds stellar magnitudes and other relevant information.

    Parameters
    ----------
    starname: str
        The name of the object. If ra and dec aren't provided nor is a
        list of magnitudes with associated uncertainties provided, the search
        for stellar magnitudes will be done using the object's name instead.
    ra: float
        RA coordinate of the object in degrees.
    dec: float
        DEC coordinate of the object in degrees.
    g_id: int, optional
        The Gaia DR2 identifier.
    plx: float, optional
        The parallax of the star in case no internet connection is available
        or if no parallax can be found on Gaia DR2.
    plx_e: float, optional
        The error on the parallax.
    rad: float, optional
        The radius of the star in case no internet connection is available
        or if no radius can be found on Gaia DR2.
    rad_e: float, optional
        The error on the stellar radius.
    temp: float, optional
        The effective temperature of the star in case no internet connection
        is available or if no effective temperature can be found on Gaia DR2.
    temp_e: float, optional
        The error on the effective temperature.
    lum: float, optional
        The stellar luminosity in case no internet connection
        is available or if no luminosity can be found on Gaia DR2.
    lum_e: float, optional
        The error on the stellar luminosity.
    dist: float, optional
        The distance in parsec.
    dist_e: float, optional
        The error on the distance.
    mag_dict: dictionary, optional
        A dictionary with the filter names as keys (names must correspond to
        those in the filter_names attribute) and with a tuple containing the
        magnitude and error for that filter as the value. Provide in case no
        internet connection is available.
    offline: bool
        If False it overrides the coordinate search entirely.
    verbose: bool, optional
        Set to False to suppress printed outputs.
    ignore: list, optional
        A list with the catalogs to ignore for whatever reason.

    Attributes
    ----------
    full_grid: ndarray
        The full grid of fluxes.
    teff: ndarray
        The effective temperature axis of the flux grid.
    logg: ndarray
        The gravity axis of the flux grid
    z: ndarray, float
        If fixed_z is False, then z is the metallicity axis of the flux grid.
        Otherwise z has the same value as fixed_z
    starname: str
        The name of the object.
    ra: float
        RA coordinate of the object.
    dec: float
        DEC coordinate of the object.
    wave: ndarray
        An array containing the wavelengths associated to the different
        filters retrieved.
    flux: ndarray
        An array containing the fluxes of the different retrieved magnitudes.

    """
    filter_names = filter_names

    colors = colors

    dustmaps = {
        'SFD': SFDQuery,
        'Lenz': Lenz2017Query,
        'Planck13': PlanckQuery,
        'Planck16': PlanckGNILCQuery,
        'Bayestar': BayestarQuery,
    }

    def __init__(self, starname, ra, dec, g_id=None,
                 plx=None, plx_e=None, rad=None, rad_e=None,
                 temp=None, temp_e=None, lum=None, lum_e=None,
                 dist=None, dist_e=None, Av=None, Av_e=None,
                 offline=False, mag_dict=None, verbose=True, ignore=None,
                 dustmap='SFD'):
        """See class docstring."""
        # MISC
        self.verbose = verbose
        self.offline = offline

        # Star stuff
        self.starname = starname
        self.ra_dec_to_deg(ra, dec)

        c = random.choice(self.colors)

        display_star_init(self, c)

        if verbose:
            if plx is not None:
                StarWarning('Parallax', 0).warn()
            if rad is not None:
                StarWarning('Radius', 0).warn()
            if temp is not None:
                StarWarning('Temperature', 0).warn()
            if lum is not None:
                StarWarning('Luminosity', 0).warn()
            if mag_dict is not None:
                StarWarning('Magnitudes', 0).warn()

        self.get_plx = True if plx is None else False
        self.get_dist = True if dist is None and plx is None else False
        self.get_rad = True if rad is None else False
        self.get_temp = True if temp is None else False
        self.get_lum = True if lum is None else False
        self.get_mags = True if mag_dict is None else False
        self.get_logg = False  # This is set to True after self.estimate_logg

        self.g_id = g_id

        # Lookup archival magnitudes, radius, temperature, luminosity
        # and parallax
        lookup = self.get_rad + self.get_temp + self.get_plx \
            + self.get_mags + self.get_dist

        if lookup:
            if not offline:
                if verbose:
                    print(
                        colored('\t\t*** LOOKING UP ARCHIVAL INFORMATION ***',
                                c)
                    )
                lib = Librarian(starname, self.ra, self.dec, g_id=self.g_id,
                                mags=self.get_mags, ignore=ignore)
                self.g_id = lib.g_id
                self.tic = lib.tic
                self.kic = lib.kic
            else:
                print(
                    colored('\t\t*** ARCHIVAL LOOKUP OVERRIDDEN ***', c)
                )
                if self.get_mags:
                    StarWarning('', 1).__raise__()
                lib = None
                self.tic = False
                self.kic = False

            # [plx, plx_e, dist, dist_e, rad, rad_e, temp, temp_e, lum, lum_e]
            libouts = extract_from_lib(lib)

            if self.get_plx:
                self.plx = libouts[0]
                self.plx_e = libouts[1]
            else:
                self.plx = plx
                self.plx_e = plx_e

            if self.get_dist:
                self.dist = libouts[2]
                self.dist_e = libouts[3]
            elif dist is not None:
                self.dist = dist
                self.dist_e = dist_e
            else:
                self.calculate_distance()

            if self.get_rad:
                self.rad = libouts[4]
                self.rad_e = libouts[5]
            else:
                self.rad = rad
                self.rad_e = rad_e

            if self.get_temp:
                self.temp = libouts[6]
                self.temp_e = libouts[7]
            else:
                self.temp = temp
                self.temp_e = temp_e

            if self.get_lum:
                self.lum = libouts[8]
                self.lum_e = libouts[9]
            else:
                self.lum = lum
                self.lum_e = lum_e

        if self.get_mags:
            self.used_filters = lib.used_filters
            self.mags = lib.mags
            self.mag_errs = lib.mag_errs
        else:
            filters = []
            self.used_filters = np.zeros(self.filter_names.shape[0])
            self.mags = np.zeros(self.filter_names.shape[0])
            self.mag_errs = np.zeros(self.filter_names.shape[0])
            for k in mag_dict.keys():
                filt_idx = np.where(k == self.filter_names)[0]
                self.used_filters[filt_idx] = 1
                self.mags[filt_idx] = mag_dict[k][0]
                self.mag_errs[filt_idx] = mag_dict[k][1]

                filters.append(k)
        self.filter_mask = np.where(self.used_filters == 1)[0]

        # IRX filters
        self.irx_filter_mask = np.array([])
        self.irx_used_filters = np.zeros(self.filter_names.shape[0])

        # Get max Av
        if Av is None:
            self.Av_e = None
            dmap = self.dustmaps[dustmap]()
            coords = SkyCoord(self.ra, self.dec, distance=self.dist,
                              unit=(u.deg, u.deg, u.pc), frame='icrs')
            if dustmap in ['SFD', 'Lenz']:
                ebv = dmap(coords)
                self.Av = ebv * 2.742
            elif dustmap == 'Bayestar':
                ebvs = dmap(coords, mode='percentile', pct=[15, 50, 84])
                if np.any(np.isnan(ebvs)):
                    StarWarning(None, 2).warn()
                    ebv = self.dustmaps['SFD']()(coords)
                    self.Av = ebv * 2.742
                else:
                    mags = ebvs * 2.742 * 0.884
                    self.Av = mags[1]
                    self.Av_e = max([mags[1] - mags[0], mags[2] - mags[1]])
            elif dustmap in ['Planck13', 'Planck16']:
                ebv = dmap(coords)
                self.Av = ebv * 3.1
        else:
            self.Av = Av
        # Get the wavelength and fluxes of the retrieved magnitudes.
        wave, flux, flux_er, bandpass = extract_info(
            self.mags, self.mag_errs, self.filter_names)

        self.wave = np.zeros(self.filter_names.shape[0])
        self.flux = np.zeros(self.filter_names.shape[0])
        self.flux_er = np.zeros(self.filter_names.shape[0])
        self.bandpass = np.zeros(self.filter_names.shape[0])

        for k in wave.keys():
            filt_idx = np.where(k == self.filter_names)[0]
            self.wave[filt_idx] = wave[k]
            self.bandpass[filt_idx] = bandpass[k]
            self.flux[filt_idx] = flux[k]
            self.flux_er[filt_idx] = flux_er[k]

        rel_er = self.flux_er[self.filter_mask] / self.flux[self.filter_mask]
        mx_rel_er = rel_er.max() + 0.1
        upper = self.flux_er[self.filter_mask] == 0
        flx = self.flux[self.filter_mask][upper]
        for i, f in zip(self.filter_mask[upper], flx):
            self.flux_er[i] = mx_rel_er * f

        # self.calculate_distance()
        c = random.choice(self.colors)
        display_star_fin(self, c)
        c = random.choice(self.colors)
        self.print_mags(c)

    def __repr__(self):
        """Repr overload."""
        return self.starname

    def ra_dec_to_deg(self, ra, dec):
        """Transform ra, dec from selected unit to degrees."""
        if isinstance(ra, float) and isinstance(dec, float):
            self.ra = ra
            self.dec = dec
            return
        c = SkyCoord(ra, dec, frame='icrs')

        self.ra = c.ra.deg
        self.dec = c.dec.deg
        pass

    def load_grid(self, model):
        """Load the model grid for interpolation."""
        # Grid stuff
        if model.lower() == 'phoenix':
            gridname = gridsdir + '/model_grid_Phoenixv2.dat'
        if model.lower() == 'btsettl':
            gridname = gridsdir + '/model_grid_BT_Settl.dat'
        if model.lower() == 'btnextgen':
            gridname = gridsdir + '/model_grid_BT_NextGen.dat'
        if model.lower() == 'btcond':
            gridname = gridsdir + '/model_grid_BT_Cond.dat'
        if model.lower() == 'ck04':
            gridname = gridsdir + '/model_grid_CK04.dat'
        if model.lower() == 'kurucz':
            gridname = gridsdir + '/model_grid_Kurucz.dat'
        if model.lower() == 'coelho':
            gridname = gridsdir + '/model_grid_Coelho.dat'

        self.full_grid = np.loadtxt(gridname)
        self.teff = self.full_grid[:, 0]
        self.logg = self.full_grid[:, 1]
        self.z = self.full_grid[:, 2]
        if self.verbose:
            print('Grid ' + model + ' loaded.')

    def calculate_distance(self):
        """Calculate distance using parallax in solar radii."""
        if self.plx == -1:
            self.dist = -1
            self.dist_e = -1
            return
        dist = 1 / (0.001 * self.plx)
        dist_e = dist * self.plx_e / self.plx
        self.dist = dist
        self.dist_e = dist_e

    def print_mags(self, c=None):
        """Pretty print of magnitudes and errors."""
        master, headers = self.__prepare_mags()
        if c is not None:
            print(
                colored('\t\t{:^20s}\t{:^9s}\t{:^11s}'.format(*headers), c)
            )
            print(colored(
                '\t\t--------------------\t---------\t-----------', c)
            )
            for i in range(master.shape[0]):
                printer = '\t\t{:^20s}\t{: ^9.4f}\t{: ^11.4f}'
                print(colored(printer.format(*master[i]), c))
        else:
            print('\t\t{:^20s}\t{:^9s}\t{:^11s}'.format(*headers))
            print('\t\t--------------------\t---------\t-----------')
            for i in range(master.shape[0]):
                printer = '\t\t{:^20s}\t{: ^9.4f}\t{: ^11.4f}'
                print(printer.format(*master[i]))
        print('')

    def save_mags(self, out):
        """Save the used magnitudes in a file."""
        master, headers = self.__prepare_mags()
        fmt = '%s %2.4f %2.4f'
        np.savetxt(out + 'mags.dat', master, header=' '.join(headers),
                   delimiter=' ', fmt=fmt)

    def __prepare_mags(self):
        """Prepare mags for either printing or saving in a file."""
        mags = self.mags[np.append(self.filter_mask,
                                   self.irx_filter_mask).astype(int)]
        ers = self.mag_errs[np.append(self.filter_mask,
                                      self.irx_filter_mask).astype(int)]
        filt = self.filter_names[np.append(self.filter_mask,
                                           self.irx_filter_mask).astype(int)]
        master = np.zeros(
            mags.size,
            dtype=[
                ('var1', 'U16'),
                ('var2', float),
                ('var3', float)
            ])
        master['var1'] = filt
        master['var2'] = mags
        master['var3'] = ers
        headers = ['Filter', 'Magnitude', 'Uncertainty']
        return master, headers

    def estimate_logg(self, out='.'):
        """Estimate logg values from MIST isochrones."""
        self.get_logg = True
        c = random.choice(self.colors)
        params = dict()  # params for isochrones.
        if self.temp is not None and self.temp_e != 0:
            params['Teff'] = (self.temp, self.temp_e)
        if self.lum is not None and self.lum != 0:
            params['LogL'] = (np.log10(self.lum),
                              np.log10(self.lum_e))
        if self.get_rad and self.rad is not None and self.rad != 0:
            params['radius'] = (self.rad, self.rad_e)
        params['parallax'] = (self.plx, self.plx_e)
        mags = self.mags[iso_mask == 1]
        mags_e = self.mag_errs[iso_mask == 1]
        used_bands = []
        for m, e, b in zip(mags, mags_e, iso_bands):
            if m != 0:
                params[b] = (m, e)
                used_bands.append(b)

        if self.verbose:
            print(
                colored(
                    '\t\t*** ESTIMATING LOGG USING MIST ISOCHRONES ***', c
                )
            )
        logg_est = estimate(used_bands, params, logg=True, out_folder=out)
        if logg_est is not None:
            self.logg = logg_est[0]
            self.logg_e = logg_est[1]
            print(colored('\t\t\tEstimated log g : ', c), end='')
            print(
                colored(
                    '{:.3f} +/- {:.3f}'.format(self.logg, self.logg_e), c)
            )

    def add_mag(self, mag, err, filt):
        """Add an individual photometry point to the SED."""
        mask = self.filter_names == filt
        self.mags[mask] = mag
        self.mag_errs[mask] = err
        if filt not in self.filter_names[-5:]:
            self.used_filters[mask] = 1
            self.filter_mask = np.where(self.used_filters == 1)[0]
        else:
            self.irx_used_filters[mask] = 1
        self.irx_filter_mask = np.where(self.irx_used_filters == 1)[0]

        self.__reload_fluxes()
        print(colored(f'\t\tAdded {filt} {mag} +/- {err}!!', 'yellow'))
        pass

    def remove_mag(self, filt):
        """Remove an individual photometry point."""
        mask = self.filter_names == filt
        self.mags[mask] = 0
        self.mag_errs[mask] = 0
        self.used_filters[mask] = 0
        self.filter_mask = np.where(self.used_filters == 1)[0]

        self.__reload_fluxes()
        pass

    def __reload_fluxes(self):
        # Get the wavelength and fluxes of the retrieved magnitudes.
        wave, flux, flux_er, bandpass = extract_info(
            self.mags, self.mag_errs, self.filter_names)

        self.wave = np.zeros(self.filter_names.shape[0])
        self.flux = np.zeros(self.filter_names.shape[0])
        self.flux_er = np.zeros(self.filter_names.shape[0])
        self.bandpass = np.zeros(self.filter_names.shape[0])

        for k in wave.keys():
            filt_idx = np.where(k == self.filter_names)[0]
            self.wave[filt_idx] = wave[k]
            self.bandpass[filt_idx] = bandpass[k]
            self.flux[filt_idx] = flux[k]
            self.flux_er[filt_idx] = flux_er[k]

        rel_er = self.flux_er[self.filter_mask] / self.flux[self.filter_mask]
        mx_rel_er = rel_er.max() + 0.1
        upper = self.flux_er[self.filter_mask] == 0
        flx = self.flux[self.filter_mask][upper]
        for i, f in zip(self.filter_mask[upper], flx):
            self.flux_er[i] = mx_rel_er * f
