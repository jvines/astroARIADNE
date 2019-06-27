"""Star.py contains the Star class which contains the data regarding a star."""
from __future__ import division, print_function

import astropy.units as u
import scipy as sp
from astropy.coordinates import Angle, SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from scipy.interpolate import griddata
from tqdm import tqdm

from phot_utils import *


class Star:
    """Object that holds stellar magnitudes and other relevant information.

    Parameters
    ----------
    starname : str
        The name of the object. If ra and dec aren't provided nor is a
        list of magnitudes with associated uncertainties prvided, the search
        for stellar magnitudes will be done using the object's name instead.

    ra : float
        RA coordinate of the object in degrees.

    dec : float
        DEC coordinate of the object in degrees.

    coord_search : bool
        If True uses coordinates to search for the object in the catalogs.
        Else it uses the object's name.

    fixed_z : bool, float, optional
        Bool with False if the fit won't have a fixed metallicity. Else
        fixed_z must be a float with the desired metallicity value. This value
        must not be [Fe/H].

    get_plx : bool, optional
        Set to True in order to query Gaia DR2 (or Hipparcos if for some reason
        the Gaia parallax is unavailable) for the stellar parallax.

    get_rad : bool, optional
        Set to True in order to query Gaia DR2 for the stellar radius, if
        available.

    verbose : bool, optional
        Set to False to supress printed outputs.

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

    # pyphot filter names: currently unused are U R I PS1_w

    filter_names = sp.array([
        '2MASS_H', '2MASS_J', '2MASS_Ks',
        'GROUND_JOHNSON_U', 'GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
        'GROUND_COUSINS_R', 'GROUND_COUSINS_I',
        'GaiaDR2v2_G', 'GaiaDR2v2_RP', 'GaiaDR2v2_BP',
        'PS1_g', 'PS1_i', 'PS1_r', 'PS1_w', 'PS1_y',  'PS1_z',
        'SDSS_g', 'SDSS_i', 'SDSS_r', 'SDSS_u', 'SDSS_z',
        'WISE_RSR_W1', 'WISE_RSR_W2'
    ])

    # Catalogs magnitude names
    __apass_mags = ['Vmag', 'Bmag', 'g_mag', 'r_mag', 'i_mag']
    __apass_errs = ['e_Vmag', 'e_Bmag', 'e_g_mag', 'e_r_mag', 'e_i_mag']
    __apass_filters = ['GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
                       'SDSS_g', 'SDSS_r', 'SDSS_i']
    __wise_mags = ['W1mag', 'W2mag']
    __wise_errs = ['e_W1mag', 'e_W2mag']
    __wise_filters = ['WISE_RSR_W1', 'WISE_RSR_W2']
    __ps1_mags = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    __ps1_errs = ['e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_ymag']
    __ps1_filters = ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']
    __twomass_mags = ['Jmag', 'Hmag', 'Kmag']
    __twomass_errs = ['e_Jmag', 'e_Hmag', 'e_Kmag']
    __twomass_filters = ['2MASS_J', '2MASS_H', '2MASS_Ks']
    __gaia_mags = ['Gmag', 'BPmag', 'RPmag']
    __gaia_errs = ['e_Gmag', 'e_BPmag', 'e_RPmag']
    __gaia_filters = ['GaiaDR2v2_G',  'GaiaDR2v2_BP', 'GaiaDR2v2_RP']
    __sdss_mags = ['umag', 'zmag']
    __sdss_errs = ['e_umag', 'e_zmag']
    __sdss_filters = ['SDSS_u', 'SDSS_z']

    # APASS DR9, WISE, PAN-STARRS DR1, GAIA DR2, 2MASS, SDSS DR9
    catalogs = {
        'apass': [
            'II/336/apass9',
            zip(__apass_mags, __apass_errs, __apass_filters)
        ],
        'Wise': [
            'II/311/wise',
            zip(__wise_mags, __wise_errs, __wise_filters)
        ],
        'Pan-STARRS':
        [
            'II/349/ps1',
            zip(__ps1_mags, __ps1_errs, __ps1_filters)
        ],
        'Gaia':
        [
            'I/345/gaia2',
            zip(__gaia_mags, __gaia_errs, __gaia_filters)
        ],
        '2MASS': [
            'II/246/out',
            zip(__twomass_mags, __twomass_errs, __twomass_filters)
        ],
        'SDSS': ['V/147/sdss12', zip(__sdss_mags, __sdss_errs, __sdss_filters)]
    }

    def __init__(self, starname, ra, dec, coord_search=False,
                 fixed_z=False, get_plx=False, plx=None, plx_e=None,
                 get_rad=False, rad=None, rad_e=None, mag_dict=None,
                 verbose=True):
        # MISC
        self.verbose = verbose
        self.get_rad = get_rad
        self.get_plx = get_plx

        # Grid stuff
        self.full_grid = sp.loadtxt('test_grid.dat')
        self.teff = self.full_grid[:, 0]
        self.logg = self.full_grid[:, 1]
        self.z = self.full_grid[:, 2] if not fixed_z else fixed_z
        self.__coord_search = coord_search
        self.fixed_z = fixed_z
        self.model_grid = dict()

        for i, f in enumerate(self.filter_names):
            self.model_grid[f] = self.full_grid[:, 3 + i]

        # Star stuff
        self.starname = starname
        self.ra = ra
        self.dec = dec

        if not mag_dict:
            self.get_magnitudes()
        else:
            mags = []
            errors = []
            filters = []
            for k in mag_dict.keys():
                filters.append(k)
                mags.append(mag_dict[k][0])
                errors.append(mag_dict[k][1])
            self.filters = sp.array(filters)
            self.magnitudes = sp.array(mags)
            self.errors = sp.array(errors)

        # Get the wavelength and fluxes of the retrieved magnitudes.
        wave, flux, flux_er, bandpass = extract_info(
            self.magnitudes, self.errors, self.filters)

        self.wave = wave
        self.flux = flux
        self.flux_er = flux_er
        self.bandpass = bandpass

        # Create the grid to interpolate later.
        if not fixed_z:
            self.grid = sp.vstack((self.teff, self.logg, self.z)).T
        else:
            self.grid = sp.vstack((self.teff, self.logg)).T

        if get_plx and not (plx and plx_e):
            self.get_parallax()
        else:
            self.plx = plx
            self.plx_e = plx_e
        self.calculate_distance()
        if get_rad:
            self.get_radius()
        elif rad and rad_e:
            self.rad = rad
            self.rad_e = rad_e

    def get_magnitudes(self):
        """Retrieve the magnitudes of the star.

        Looks into APASS, WISE, Pan-STARRS, Gaia, 2MASS and SDSS surveys
        looking for different magnitudes for the star, along with the
        associated uncertainties.
        """
        if self.verbose:
            print('Looking online for archival magnitudes for star', end=' ')
            print(self.starname)

        cats = self.get_catalogs()

        filters = []
        magnitudes = []
        errors = []

        for c in tqdm(self.catalogs.keys()):
            # load magnitude names, filter names and error names of
            # current catalog
            current = self.catalogs[c][1]
            try:
                # load current catalog
                current_cat = cats[self.catalogs[c][0]]
                for m, e, f in current:
                    if sp.ma.is_masked(current_cat[m][0]):
                        continue
                    filters.append(f)
                    magnitudes.append(current_cat[m][0])
                    errors.append(current_cat[e][0])
            except Exception as e:
                print('Star is not available in catalog', end=' ')
                print(c)

        self.filters = sp.array(filters)
        self.magnitudes = sp.array(magnitudes)
        self.errors = sp.array(errors)

    def get_parallax(self):
        """Retrieve the parallax of the star.

        Retrieve the parallax of the star from Gaia
        (or Hipparcos if Gaia is absent)
        """
        catalog = Gaia.query_object_async(
            SkyCoord(
                ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'
            ), radius=Angle(.001, "deg")
        )

        if self.verbose:
            print('Searching for parallax in Gaia...')

        try:
            plx = catalog['parallax'][0]
            plx_e = catalog['parallax_error'][0]
            self.plx = plx
            self.plx_e = plx_e

            if self.verbose:
                print('Parallax found!\nParallax value', end=': ')
                print(self.plx, end=' +- ')
                print(self.plx_e, end=' mas\n')
        except Exception as e:
            print('No Gaia parallax found for this star.', end=' ')
            print('Try inputting manually.')

    def get_radius(self):
        """Retrieve the stellar radius from Gaia if available."""
        catalog = Gaia.query_object_async(
            SkyCoord(
                ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'
            ), radius=Angle(.001, "deg")
        )

        if self.verbose:
            print('Searching for radius in Gaia...')

        try:
            rad = catalog['radius_val'][0]
            rad_upper = catalog['radius_percentile_upper'][0]
            rad_lower = catalog['radius_percentile_lower'][0]
            e_up = rad_upper - rad
            e_lo = rad - rad_lower
            rad_e = (e_up + e_lo) / 2
            self.rad = rad
            self.rad_e = rad_e

            if self.verbose:
                print('Radius found!\nRadius value', end=': ')
                print(self.rad, end=' +- ')
                print(self.rad_e, end=' R_sun\n')
        except Exception as e:
            print('No radius value found.', end=' ')
            print('Try inputting manually.')

    def get_catalogs(self):
        """Retrieve available catalogs for a star from Vizier."""
        if self.__coord_search:
            cats = Vizier.query_region(
                SkyCoord(
                    ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'
                ), radius=Angle(.001, "deg")
            )
        else:
            cats = Vizier.query_object(self.starname)

        return cats

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
        # filter_index = sp.where(self.filter_names == filt)[0]
        if not self.fixed_z:
            flux = griddata(self.grid, self.model_grid[filt],
                            (temp, logg, z), method='linear')
        else:
            flux = griddata(self.grid, self.model_grid[filt],
                            (temp, logg), method='linear')

        return flux

    def calculate_distance(self):
        """Calculate distance using parallax in solar radii."""
        dist = 1 / (1000 * self.plx)
        dist_e = dist * self.plx_e / self.plx
        self.dist = dist * u.pc.to(u.solRad)
        self.dist_e = dist_e * u.pc.to(u.solRad)
