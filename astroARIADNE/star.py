"""Star.py contains the Star class which contains the data regarding a star."""

import pickle
import random
from contextlib import closing

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from scipy.interpolate import RegularGridInterpolator
from termcolor import colored

from .config import gridsdir
from .isochrone import estimate
from .librarian import Librarian
from .phot_utils import *
from .utils import display_star_fin, display_star_init


class Star:
    """Object that holds stellar magnitudes and other relevant information.

    Parameters
    ----------
    starname : str
        The name of the object. If ra and dec aren't provided nor is a
        list of magnitudes with associated uncertainties provided, the search
        for stellar magnitudes will be done using the object's name instead.

    ra : float
        RA coordinate of the object in degrees.

    dec : float
        DEC coordinate of the object in degrees.

    get_plx : bool, optional
        Set to True in order to query Gaia DR2 for the stellar parallax.

    plx : float, optional
        The parallax of the star in case no internet connection is available
        or if no parallax can be found on Gaia DR2.

    plx_e : float, optional
        The error on the parallax.

    get_rad : bool, optional
        Set to True in order to query Gaia DR2 for the stellar radius, if
        available.

    rad : float, optional
        The radius of the star in case no internet connection is available
        or if no radius can be found on Gaia DR2.

    rad_e : float, optional
        The error on the stellar radius.

    get_temp : bool, optional
        Set to True in order to query Gaia DR2 for the effective temperature,
        if available.

    temp : float, optional
        The effective temperature of the star in case no internet connection
        is available or if no effective temperature can be found on Gaia DR2.

    temp_e : float, optional
        The error on the effective temperature.

    get_lum : bool, optional
        Set to True in order to query Gaia DR2 for the stellar luminosity,
        if available.

    lum : float, optional
        The stellar luminosity in case no internet connection
        is available or if no luminosity can be found on Gaia DR2.

    lum_e : float, optional
        The error on the stellar luminosity.

    mag_dict : dictionary, optional
        A dictionary with the filter names as keys (names must correspond to
        those in the filter_names attribute) and with a tuple containing the
        magnitude and error for that filter as the value. Provide in case no
        internet connection is available.

    coordinate_search : bool
        If True uses coordinates to search for the object in the catalogs.
        Else it uses the object's name.

    verbose : bool, optional
        Set to False to suppress printed outputs.

    ignore : list, optional
        A list with the catalogs to ignore for whatever reason.

    Attributes
    ----------
    catalogs : dict
        A dictionary with the Vizier catalogs of different surveys
        used to retrieve stellar magnitudes.

    full_grid : ndarray
        The full grid of fluxes.

    teff : ndarray
        The effective temperature axis of the flux grid.

    logg : ndarray
        The gravity axis of the flux grid

    z : ndarray, float
        If fixed_z is False, then z is the metallicity axis of the flux grid.
        Otherwise z has the same value as fixed_z

    starname : str
        The name of the object.

    ra : float
        RA coordinate of the object.

    dec : float
        DEC coordinate of the object.

    filters : ndarray
        An array containing the filters or bands for which there is
        archival photometry

    magnitudes : ndarray
        An array containing the archival magnitudes for the object.

    errors : ndarray
        An array containing the uncertainties in the magnitudes.

    wave : ndarray
        An array containing the wavelengths associated to the different
        filters retrieved.

    flux : ndarray
        An array containing the fluxes of the different retrieved magnitudes.

    grid : ndarray
        An array containing a grid with teff, logg and z if it's not fixed
        to be used for interpolation later.

    """

    # pyphot filter names

    filter_names = sp.array([
        '2MASS_H', '2MASS_J', '2MASS_Ks',
        'GROUND_COUSINS_I', 'GROUND_COUSINS_R',
        'GROUND_JOHNSON_U', 'GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
        'TYCHO_B_MvB', 'TYCHO_V_MvB',
        'STROMGREN_b', 'STROMGREN_u', 'STROMGREN_v', 'STROMGREN_y',
        'GaiaDR2v2_G', 'GaiaDR2v2_RP', 'GaiaDR2v2_BP',
        'PS1_g', 'PS1_i', 'PS1_r', 'PS1_w', 'PS1_y',  'PS1_z',
        'SDSS_g', 'SDSS_i', 'SDSS_r', 'SDSS_u', 'SDSS_z',
        'WISE_RSR_W1', 'WISE_RSR_W2',
        'GALEX_FUV', 'GALEX_NUV',
        'SPITZER_IRAC_36', 'SPITZER_IRAC_45',
        'NGTS_I', 'TESS', 'KEPLER_Kp'
    ])

    colors = [
        'red', 'green', 'blue', 'yellow',
        'grey', 'magenta', 'cyan', 'white'
    ]

    def __init__(self, starname, ra, dec, g_id=None,
                 plx=None, plx_e=None,
                 rad=None, rad_e=None,
                 temp=None, temp_e=None,
                 lum=None, lum_e=None,
                 logg=None, logg_e=None,
                 dist=None, dist_e=None,
                 Av=None,
                 mag_dict=None, verbose=True, ignore=[]):
        """See class docstring."""
        # MISC
        self.verbose = verbose

        # Star stuff
        self.starname = starname
        self.ra_dec_to_deg(ra, dec)

        c = random.choice(self.colors)

        display_star_init(self, c)

        if verbose:
            if plx is not None:
                print('Parallax input detected.', end=' ')
                print('Overriding coordinate search.')
            if rad is not None:
                print('Radius input detected.', end=' ')
                print('Overriding coordinate search.')
            if temp is not None:
                print('Temperature input detected.', end=' ')
                print('Overriding coordinate search.')
            if lum is not None:
                print('Luminosity input detected.', end=' ')
                print('Overriding coordinate search.')
            if mag_dict is not None:
                print('Input magnitudes detected.', end=' ')
                print('Overriding coordinate search.')

        self.get_rad = True if rad is None else False
        self.get_dist = True if dist is None else False
        self.get_plx = True if plx is None else False
        self.get_temp = True if temp is None else False
        self.get_lum = True if lum is None else False
        self.get_mags = True if mag_dict is None else False
        self.get_logg = False

        self.g_id = g_id

        # Lookup archival magnitudes, radius, temperature, luminosity
        # and parallax
        lookup = self.get_rad + self.get_temp + self.get_plx \
            + self.get_mags + self.get_dist
        if lookup:
            if verbose:
                print(
                    colored('\t\t*** LOOKING UP ARCHIVAL INFORMATION ***', c)
                )
            lib = Librarian(starname, self.ra, self.dec, g_id=self.g_id,
                            mags=self.get_mags, ignore=ignore)
            self.g_id = lib.g_id
            self.tic = lib.tic
            self.kic = lib.kic
            if self.get_plx:
                self.plx = lib.plx
                self.plx_e = lib.plx_e
            else:
                self.plx = plx
                self.plx_e = plx_e

            if self.get_dist:
                self.dist = lib.dist
                self.dist_e = lib.dist_e
            else:
                self.dist = dist
                self.dist_e = dist_e

            if self.get_rad:
                self.rad = lib.rad
                self.rad_e = lib.rad_e
            else:
                self.rad = rad
                self.rad_e = rad_e

            if self.get_temp:
                self.temp = lib.temp
                self.temp_e = lib.temp_e
            else:
                self.temp = temp
                self.temp_e = temp_e

            if self.get_lum:
                self.lum = lib.lum
                self.lum_e = lib.lum_e
            else:
                self.lum = lum
                self.lum_e = lum_e

            if self.get_mags:
                self.used_filters = lib.used_filters
                self.mags = lib.mags
                self.mag_errs = lib.mag_errs

        if not self.get_mags:
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

        # Get max Av
        if Av is None:
            sfd = SFDQuery()
            coords = SkyCoord(self.ra, self.dec,
                              unit=(u.deg, u.deg), frame='icrs')
            ebv = sfd(coords)
            self.Av = ebv * 2.742
        else:
            self.Av = Av
        # Get the wavelength and fluxes of the retrieved magnitudes.
        wave, flux, flux_er, bandpass = extract_info(
            self.mags[self.filter_mask], self.mag_errs[self.filter_mask],
            self.filter_names[self.filter_mask])

        self.wave = np.zeros(self.filter_names.shape[0])
        self.flux = np.zeros(self.filter_names.shape[0])
        self.flux_er = np.zeros(self.filter_names.shape[0])
        self.bandpass = np.zeros(self.filter_names.shape[0])

        for k in wave.keys():
            filt_idx = np.where(k == self.filter_names)[0]
            self.wave[filt_idx] = wave[k]
            self.flux[filt_idx] = flux[k]
            self.flux_er[filt_idx] = flux_er[k]
            self.bandpass[filt_idx] = bandpass[k]

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

    def interpolate(self, out_name):
        """Create interpolation grids for later evaluation."""
        raise DeprecationWarning()
        if self.verbose:
            print('Interpolating grids for filters:')
        interpolators = np.zeros(self.filter_names.shape[0], dtype=object)
        ut = np.unique(self.full_grid[:, 0])
        ug = np.unique(self.full_grid[:, 1])
        uz = np.unique(self.full_grid[:, 2])
        for ii, f in enumerate(self.filter_names):
            cube = np.zeros((ut.shape[0], ug.shape[0], uz.shape[0]))
            if self.verbose:
                print(f)
            for i, t in enumerate(ut):
                t_idx = self.full_grid[:, 0] == t
                for j, g in enumerate(ug):
                    g_idx = self.full_grid[:, 1] == g
                    for k, z in enumerate(uz):
                        z_idx = self.full_grid[:, 2] == z
                        flx = self.full_grid[:, 3 + ii][t_idx * g_idx * z_idx]
                        insert = flx[0] if len(flx) == 1 else 0
                        cube[i, j, k] = insert
            filt_idx = np.where(f == self.filter_names)[0]
            interpolators[filt_idx] = RegularGridInterpolator(
                (ut, ug, uz), cube, bounds_error=False)
        with closing(open(out_name + '.pkl', 'wb')) as jar:
            pickle.dump(interpolators, jar)

    def get_interpolated_flux(self, temp, logg, z, filt):
        """Interpolate the grid of fluxes in a given teff, logg and z.

        Parameters
        ----------
        temp : float
            The effective temperature.

        logg : float
            The superficial gravity.

        z : float
            The metallicity.

        filt : str
            The desired filter.

        Returns
        -------
        flux : float
            The interpolated flux at temp, logg, z for filter filt.

        """
        values = (temp, logg, z) if not self.fixed_z else (temp, logg)
        flux = self.interpolators[filt](values)
        return flux

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
                colored('\t\t{:^16s}\t{:^9s}\t{:^11s}'.format(*headers), c)
            )
            print(colored(
                '\t\t----------------\t---------\t-----------', c)
            )
            for i in range(master.shape[0]):
                printer = '\t\t{:^16s}\t{: ^9.4f}\t{: ^11.4f}'
                print(colored(printer.format(*master[i]), c))
        else:
            print('\t\t{:^16s}\t{:^9s}\t{:^11s}'.format(*headers))
            print('\t\t----------------\t---------\t-----------')
            for i in range(master.shape[0]):
                printer = '\t\t\t{:^16s}\t{: ^9.4f}\t{: ^11.4f}'
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
        mags = self.mags[self.filter_mask]
        ers = self.mag_errs[self.filter_mask]
        filt = self.filter_names[self.filter_mask]
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

    def estimate_logg(self):
        """Estimate logg values from MIST isochrones."""
        self.get_logg = True
        c = random.choice(self.colors)
        params = dict()  # params for isochrones.
        if self.temp is not None and self.temp_e != 0:
            params['Teff'] = (self.temp, self.temp_e)
        if self.lum is not None and self.lum != 0:
            params['LogL'] = (np.log10(self.lum),
                              np.log10(self.lum_e))
        # if self.get_rad and self.rad is not None and self.rad != 0:
        #     params['radius'] = (self.rad, self.rad_e)
        params['parallax'] = (self.plx, self.plx_e)
        mask = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                         0, 1, 0])
        mags = self.mags[mask == 1]
        mags_e = self.mag_errs[mask == 1]
        bands = ['H', 'J', 'K', 'V', 'B', 'G', 'RP', 'BP', 'W1', 'W2', 'TESS']
        used_bands = []
        for m, e, b in zip(mags, mags_e, bands):
            if m != 0:
                params[b] = (m, e)
                used_bands.append(b)

        if self.verbose:
            print(
                colored(
                    '\t\t*** ESTIMATING LOGG USING MIST ISOCHRONES ***', c
                )
            )
        logg_est = estimate(used_bands, params, logg=True)
        if logg_est is not None:
            self.logg = logg_est[0]
            self.logg_e = logg_est[1]
            print(colored('\t\t\tEstimated log g : ', c), end='')
            print(
                colored(
                    '{:.3f} +/- {:.3f}'.format(self.logg, self.logg_e), c)
            )

    def add_mag(self, mag, err, filter):
        """Add an individual photometry point to the SED."""
        mask = self.filter_names == filter
        self.mags[mask] = mag
        self.mag_errs[mask] = err
        self.used_filters[mask] = 1
        self.filter_mask = np.where(self.used_filters == 1)[0]

        self.__reload_fluxes()
        pass

    def __reload_fluxes(self):
        # Get the wavelength and fluxes of the retrieved magnitudes.
        wave, flux, flux_er, bandpass = extract_info(
            self.mags[self.filter_mask], self.mag_errs[self.filter_mask],
            self.filter_names[self.filter_mask])

        self.wave = np.zeros(self.filter_names.shape[0])
        self.flux = np.zeros(self.filter_names.shape[0])
        self.flux_er = np.zeros(self.filter_names.shape[0])
        self.bandpass = np.zeros(self.filter_names.shape[0])

        for k in wave.keys():
            filt_idx = np.where(k == self.filter_names)[0]
            self.wave[filt_idx] = wave[k]
            self.flux[filt_idx] = flux[k]
            self.flux_er[filt_idx] = flux_er[k]
            self.bandpass[filt_idx] = bandpass[k]
