"""Helper class to look up broadband photometry and stellar parameters."""
import astropy.units as u
import scipy as sp
from astropy.coordinates import Angle, SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from tqdm import tqdm


class Librarian:
    """Docstring."""

    # pyphot filter names: currently unused are U R I PS1_w

    filter_names = sp.array([
        '2MASS_H', '2MASS_J', '2MASS_Ks',
        'GROUND_JOHNSON_U', 'GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
        'GaiaDR2v2_G', 'GaiaDR2v2_RP', 'GaiaDR2v2_BP',
        'PS1_g', 'PS1_i', 'PS1_r', 'PS1_w', 'PS1_y',  'PS1_z',
        'SDSS_g', 'SDSS_i', 'SDSS_r', 'SDSS_u', 'SDSS_z',
        'WISE_RSR_W1', 'WISE_RSR_W2', 'GALEX_FUV', 'GALEX_NUV'
    ])

    # Catalogs magnitude names
    # NOTE: SDSS_z is breaking the fit for some reason.
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
    __sdss_mags = ['umag']
    # __sdss_mags = ['umag', 'zmag']
    __sdss_errs = ['e_umag']
    # __sdss_errs = ['e_umag', 'e_zmag']
    __sdss_filters = ['SDSS_u']
    # __sdss_filters = ['SDSS_u', 'SDSS_z']
    __galex_mags = ['FUV', 'NUV']
    __galex_errs = ['e_FUV', 'e_NUV']
    __galex_filters = ['GALEX_FUV', 'GALEX_NUV']

    # APASS DR9, WISE, PAN-STARRS DR1, GAIA DR2, 2MASS, SDSS DR9
    catalogs = {
        'apass': [
            'II/336/apass9',
            zip(__apass_mags, __apass_errs, __apass_filters)
        ],
        'Wise': [
            'II/328/allwise',
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
        'SDSS': [
            'V/147/sdss12', zip(__sdss_mags, __sdss_errs, __sdss_filters)
        ],
        'GALEX': [
            'II/312/ais', zip(__galex_mags, __galex_errs, __galex_filters)
        ]
    }

    def __init__(self, starname, ra, dec, get_plx, get_rad, get_temp, get_lum,
                 verbose=True):
        self.starname = starname
        self.ra = ra
        self.dec = dec
        self.verbose = verbose

        self.used_filters = sp.zeros(self.filter_names.shape[0])
        self.mags = sp.zeros(self.filter_names.shape[0])
        self.mag_errs = sp.zeros(self.filter_names.shape[0])

        self.get_stellar_params(get_plx, get_rad, get_temp, get_lum)
        pass

    def get_catalogs(self, coordinate_lookup):
        """Retrieve available catalogs for a star from Vizier."""
        if coordinate_lookup:
            cats = Vizier.query_region(
                SkyCoord(
                    ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'
                ), radius=Angle(.001, "deg")
            )
        else:
            cats = Vizier.query_object(self.starname)

        return cats

    def get_magnitudes(self, coordinate_lookup):
        """Retrieve the magnitudes of the star.

        Looks into APASS, WISE, Pan-STARRS, Gaia, 2MASS and SDSS surveys
        looking for different magnitudes for the star, along with the
        associated uncertainties.
        """
        if self.verbose:
            print('Looking online for archival magnitudes for star', end=' ')
            print(self.starname)

        cats = self.get_catalogs(coordinate_lookup)

        for c in tqdm(self.catalogs.keys()):
            # load magnitude names, filter names and error names of
            # current catalog
            current = self.catalogs[c][1]
            try:
                # load current catalog
                current_cat = cats[self.catalogs[c][0]]
                for m, e, f in current:
                    if sp.ma.is_masked(current_cat[m][0]):
                        if self.verbose:
                            print('No magnitude found for filter', end=' ')
                            print(f, end='. Skipping\n')
                        continue
                    if sp.ma.is_masked(current_cat[e][0]):
                        if self.verbose:
                            print('No error for filter', end=' ')
                            print(f, end='. Skipping\n')
                        continue
                    if current_cat[e][0] == 0:
                        if self.verbose:
                            print('Retrieved error for filter', end=' ')
                            print(f, end=' is 0. Skipping\n')
                        continue
                    filt_idx = sp.where(f == self.filter_names)[0]
                    self.used_filters[filt_idx] = 1
                    self.mags[filt_idx] = current_cat[m][0]
                    self.mag_errs[filt_idx] = current_cat[e][0]
            except Exception as e:
                if self.verbose:
                    print('Star is not available in catalog', end=' ')
                    print(c)
        pass

    def get_stellar_params(self, get_plx, get_rad, get_temp, get_lum):
        """Retrieve stellar parameters from Gaia if available.

        The retrieved stellar parameters are parallax, radius and effective
        temperature.
        The reported errors are 2 * the highest 1 sigma error found

        Parameters
        ----------
        get_plx : bool
            True to retrieve parallax.
        get_rad : bool
            True to retrieve radius.
        get_temp : bool
            True to retrieve effective temperature.
        get_lum : bool
            True to retrieve luminosity.

        """
        # Query Gaia
        catalog = Gaia.query_object_async(
            SkyCoord(
                ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'
            ), radius=Angle(.001, "deg")
        )

        if get_plx:
            self.get_parallax(catalog)
        if get_rad:
            self.get_radius(catalog)
        if get_temp:
            self.get_temperature(catalog)
        if get_lum:
            self.get_luminosity(catalog)
        pass

    def get_parallax(self, catalog):
        """Retrieve the parallax of the star."""
        if self.verbose:
            print('Searching for parallax in Gaia...')

        try:
            plx = catalog['parallax'][0]
            plx_e = catalog['parallax_error'][0]
            self.plx = plx + 0.082  # offset stassusn torres 18
            self.plx_e = plx_e

            if self.verbose:
                print('Parallax found!\nParallax value', end=': ')
                print(self.plx, end=' +- ')
                print(self.plx_e, end=' mas\n')
        except Exception as e:
            raise Exception('No Gaia parallax found for this star.')
        pass

    def get_radius(self, catalog):
        """Retrieve the stellar radius from Gaia if available."""
        if self.verbose:
            print('Searching for radius in Gaia...')

        try:
            rad = catalog['radius_val'][0]
            if sp.ma.is_masked(rad):
                self.rad = None
                self.rad_e = None
                return
            rad_upper = catalog['radius_percentile_upper'][0]
            rad_lower = catalog['radius_percentile_lower'][0]
            e_up = rad_upper - rad
            e_lo = rad - rad_lower
            rad_e = max([e_up, e_lo])
            self.rad = rad
            self.rad_e = rad_e

            if self.verbose:
                print('Radius found!\nRadius value', end=': ')
                print(self.rad, end=' +- ')
                print(self.rad_e, end=' R_sun\n')
        except Exception as e:
            raise Exception('No radius value found.')
        pass

    def get_temperature(self, catalog):
        """Retrieve effective temperature from Gaia if available."""
        if self.verbose:
            print('Searching for effective temperature in Gaia...')

        try:
            temp = catalog['teff_val'][0]
            temp_upper = catalog['teff_percentile_upper'][0]
            temp_lower = catalog['teff_percentile_lower'][0]
            e_up = temp_upper - temp
            e_lo = temp - temp_lower
            temp_e = max([e_up, e_lo])
            self.temp = temp
            self.temp_e = temp_e
            if self.verbose:
                print('Teff found!\nTeff value', end=': ')
                print(self.temp, end=' +- ')
                print(self.temp_e, end=' K\n')
        except Exception as e:
            raise Exception('No effective temperature value found.')
        pass

    def get_luminosity(self, catalog):
        """Retrieve the lumnosity from Gaia if available."""
        if self.verbose:
            print('Searching for luminosity in Gaia...')

        try:
            lum = catalog['lum_val'][0]
            if sp.ma.is_masked(lum):
                self.lum = None
                self.lum_e = None
                return
            lum_upper = catalog['lum_percentile_upper'][0]
            lum_lower = catalog['lum_percentile_lower'][0]
            e_up = lum_upper - lum
            e_lo = lum - lum_lower
            lum_e = max([e_up, e_lo])
            self.lum = lum
            self.lum_e = lum_e
            if self.verbose:
                print('Luminosity found!\nLuminosity value', end=': ')
                print(self.lum, end=' +- ')
                print(self.lum_e, end=' L_Sol\n')
        except Exception as e:
            raise Exception('No luminosity value found.')
        pass
