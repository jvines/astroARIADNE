# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*elif/ /^\s*def/
"""Helper class to look up broadband photometry and stellar parameters."""

__all__ = ['Librarian']

import logging
import os
import sys
import warnings

logger = logging.getLogger(__name__)

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from regions import CircleSkyRegion

from .error import CatalogWarning
from .config import filter_names

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

Vizier.ROW_LIMIT = -1
Vizier.columns = ['all']
Catalogs.ROW_LIMIT = -1
Catalogs.columns = ['all']


class Librarian:
    """Class that handles querying for photometry and astrometry data."""

    # pyphot filter names
    filter_names = filter_names

    # Catalogs magnitude names
    __apass_mags = ['vmag', 'bmag', 'g_mag', 'r_mag', 'i_mag']
    __apass_errs = ['e_vmag', 'e_bmag', 'e_g_mag', 'e_r_mag', 'e_i_mag']
    __apass_filters = ['GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
                       'SDSS_g', 'SDSS_r', 'SDSS_i']
    __ascc_mags = ['Vmag', 'Bmag']  # , 'Jmag', 'Hmag', 'Kmag']
    __ascc_errs = ['e_Vmag', 'e_Bmag']  # , 'e_Jmag', 'e_Hmag', 'e_Kmag']
    __ascc_filters = ['GROUND_JOHNSON_V', 'GROUND_JOHNSON_B']
    # '2MASS_J', '2MASS_H', '2MASS_Ks']
    __wise_mags = ['W1mag', 'W2mag']
    __wise_errs = ['e_W1mag', 'e_W2mag']
    __wise_filters = ['WISE_RSR_W1', 'WISE_RSR_W2']
    __ps1_mags = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    __ps1_errs = ['e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_ymag']
    __ps1_filters = ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']
    __tmass_mags = ['Jmag', 'Hmag', 'Kmag']
    __tmass_errs = ['e_Jmag', 'e_Hmag', 'e_Kmag']
    __tmass_filters = ['2MASS_J', '2MASS_H', '2MASS_Ks']
    __gaia_mags = ['Gmag', 'BPmag', 'RPmag']
    __gaia_errs = ['e_Gmag', 'e_BPmag', 'e_RPmag']
    __gaia_filters = ['GaiaDR2v2_G', 'GaiaDR2v2_BP', 'GaiaDR2v2_RP']
    # __sdss_mags = ['gmag', 'rmag', 'imag']
    __sdss_mags = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
    # __sdss_errs = ['e_gmag', 'e_rmag', 'e_imag']
    __sdss_errs = ['e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag']
    # __sdss_filters = ['SDSS_g', 'SDSS_r', 'SDSS_i']
    __sdss_filters = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']
    __galex_mags = ['FUV', 'NUV']
    __galex_errs = ['e_FUV', 'e_NUV']
    __galex_filters = ['GALEX_FUV', 'GALEX_NUV']
    __irac_mags = ['_3.6mag', '_4.5mag']
    __irac_errs = ['e_3.6mag', 'e_4.5mag']
    __irac_filters = ['SPITZER_IRAC_36', 'SPITZER_IRAC_45']
    __tycho_mags = ['BTmag', 'VTmag']
    __tycho_errs = ['e_BTmag', 'e_VTmag']
    __tycho_filters = ['TYCHO_B_MvB', 'TYCHO_V_MvB']
    __tess_mags = ['Tmag']
    __tess_errs = ['e_Tmag']
    __tess_filters = ['TESS']
    __skymapper_mags = ['u_psf', 'v_psf', 'g_psf', 'r_psf', 'i_psf', 'z_psf']
    __skymapper_errs = ['e_u_psf', 'e_v_psf', 'e_g_psf',
                        'e_r_psf', 'e_i_psf', 'e_z_psf']
    __skymapper_filters = ['SkyMapper_u', 'SkyMapper_v', 'SkyMapper_g',
                           'SkyMapper_r', 'SkyMapper_i', 'SkyMapper_z']

    # APASS DR9, WISE, PAN-STARRS DR1, GAIA DR2, 2MASS, SDSS DR9
    catalogs = {
        'APASS': [
            None, list(zip(__apass_mags, __apass_errs,
                          __apass_filters))
        ],
        'Wise': [
            'II/328/allwise', list(zip(__wise_mags, __wise_errs,
                                       __wise_filters))
        ],
        'Pan-STARRS': [
            'II/349/ps1', list(zip(__ps1_mags, __ps1_errs, __ps1_filters))
        ],
        'Gaia': [
            'I/355/gaiadr3', list(zip(__gaia_mags, __gaia_errs, __gaia_filters))
        ],
        '2MASS': [
            'II/246/out', list(zip(__tmass_mags, __tmass_errs, __tmass_filters))
        ],
        'SDSS': [
            'V/147/sdss12', list(zip(__sdss_mags, __sdss_errs, __sdss_filters))
        ],
        'GALEX': [
            'II/312/ais', list(zip(__galex_mags, __galex_errs, __galex_filters))
        ],
        'ASCC': [
            'I/280B/ascc', list(zip(__ascc_mags, __ascc_errs, __ascc_filters))
        ],
        'TYCHO2': [
            'I/259/tyc2', list(zip(__tycho_mags, __tycho_errs, __tycho_filters))
        ],
        'GLIMPSE': [
            'II/293/glimpse', list(zip(__irac_mags, __irac_errs,
                                       __irac_filters))
        ],
        'TESS': [
            'TIC', list(zip(__tess_mags, __tess_errs, __tess_filters))
        ],
        'SkyMapper': [
            None, list(zip(__skymapper_mags, __skymapper_errs,
                          __skymapper_filters))
        ],
        'STROMGREN_PAUNZ': [
            'J/A+A/580/A23/catalog', -1
        ],
        'STROMGREN_HAUCK': [
            'II/215/catalog', -1
        ],
        'MERMILLIOD': [
            'II/168/ubvmeans', -1
        ],
    }

    def __init__(self, starname, ra, dec, radius=None, g_id=None,
                 mags=True, ignore=None):
        self.starname = starname
        self.ra = ra
        self.dec = dec
        self.ignore = ignore if ignore is not None else []
        self.tic = None
        self.kic = None
        self.ids = []

        self.used_filters = np.zeros(self.filter_names.shape[0])
        self.mags = np.zeros(self.filter_names.shape[0])
        self.mag_errs = np.zeros(self.filter_names.shape[0])

        # self.create_logfile()

        if radius is None:
            self.radius = 3 * u.arcmin
        else:
            self.radius = radius
        if g_id is None:
            print('No Gaia ID provided. Searching for nearest source.')
            self.g_id = self._get_gaia_id(self.ra, self.dec, self.radius)
            print('Gaia ID found: {0}'.format(self.g_id))
        else:
            self.g_id = g_id

        self.gaia_params()
        if mags:
            self.gaia_query()
            self.spectroscopic_params = self.query_spectroscopic_params()
            # Backward compat: rave_params points to spectroscopic_params
            # only when the source is RAVE, otherwise None
            if (self.spectroscopic_params is not None
                    and self.spectroscopic_params.get('source') == 'RAVE_DR6'):
                self.rave_params = self.spectroscopic_params
            else:
                self.rave_params = None
            self.get_magnitudes()
            idx = self.used_filters >= 1
            self.used_filters[idx] = 1
        else:
            self.rave_params = None
            self.spectroscopic_params = None

        pass


    def gaia_params(self):
        """Query Gaia DR3 for stellar parameters via Vizier (no Gaia TAP required)."""
        import numpy.ma as ma
        from astropy.table import Table

        v = Vizier(columns=['all'])

        # Query main source table for astrometry and GSP-Phot Teff
        main_cats = v.query_constraints(
            catalog='I/355/gaiadr3', Source=str(self.g_id)
        )
        if not main_cats or len(main_cats[0]) == 0:
            raise ValueError(f"Star {self.g_id} not found in Gaia DR3")
        main = main_cats[0][0]

        # Query astrophysical parameters table for FLAME results
        ap_cats = v.query_constraints(
            catalog='I/355/paramp', Source=str(self.g_id)
        )
        ap = ap_cats[0][0] if ap_cats and len(ap_cats[0]) > 0 else None

        def _col(row, col):
            """Return (float_value, is_masked) for a column from a Vizier row."""
            if row is None:
                return float('nan'), True
            try:
                val = row[col]
            except (KeyError, IndexError):
                return float('nan'), True
            if ma.is_masked(val):
                return float('nan'), True
            return float(val), False

        # Map (tap_column_name, vizier_row, vizier_column_name)
        col_map = [
            ('parallax',           main, 'Plx'),
            ('parallax_error',     main, 'e_Plx'),
            ('teff_gspphot',       main, 'Teff'),
            ('teff_gspphot_lower', main, 'b_Teff'),
            ('teff_gspphot_upper', main, 'B_Teff'),
            ('radius_flame',       ap,   'Rad-Flame'),
            ('radius_flame_lower', ap,   'b_Rad-Flame'),
            ('radius_flame_upper', ap,   'B_Rad-Flame'),
            ('lum_flame',          ap,   'Lum-Flame'),
            ('lum_flame_lower',    ap,   'b_Lum-Flame'),
            ('lum_flame_upper',    ap,   'B_Lum-Flame'),
            ('mass_flame',         ap,   'Mass-Flame'),
            ('mass_flame_lower',   ap,   'b_Mass-Flame'),
            ('mass_flame_upper',   ap,   'B_Mass-Flame'),
            ('age_flame',          ap,   'Age-Flame'),
            ('age_flame_lower',    ap,   'b_Age-Flame'),
            ('age_flame_upper',    ap,   'B_Age-Flame'),
        ]

        data = {}
        for tap_name, row, viz_col in col_map:
            val, is_masked = _col(row, viz_col)
            data[tap_name] = ma.array([val], mask=[is_masked])

        res = Table(data)

        # Extract stellar parameters using existing helper methods
        self.plx, self.plx_e = self._get_parallax(res)
        self.temp, self.temp_e = self._get_teff(res)
        self.rad, self.rad_e = self._get_radius(res)
        self.lum, self.lum_e = self._get_lum(res)
        self.mass, self.mass_e = self._get_mass(res)
        self.age, self.age_e = self._get_age(res)
        self.dist, self.dist_e = self._get_distance(self.ra, self.dec,
                                                    self.radius, self.g_id)
        pass

    def query_rave_params(self):
        """Query RAVE DR6 for star-specific stellar parameters.

        Returns dict with teff, logg, feh and their errors, or None if not found.
        """
        # Check if RAVE ID was found in crossmatch
        if 'RAVE' not in self.ids or not self.ids['RAVE'] or self.ids['RAVE'] == 'skipped':
            return None

        rave_id = self.ids['RAVE']

        try:
            # Query RAVE DR6 catalog via Vizier using the ID
            #  MADERA (MAtisse and DEgas used in RAve) stellar parameters
            cat = Vizier.query_constraints(
                catalog='III/283/madera',
                ObsID=rave_id
            )

            if len(cat) == 0:
                return None

            rave_cat = cat[0]
            if len(rave_cat) == 0:
                return None

            row = rave_cat[0]

            if row['Qual'] == 1:
                return None

            rave_data = {
                'teff': float(row['TeffmC']),
                'teff_err': float(row['e_Teffm']),
                'logg': float(row['loggmC']),
                'logg_err': float(row['e_loggm']),
                'feh': float(row['[m/H]mC']),
                'feh_err': float(row['e_[m/H]m'])
            }

            print("RAVE DR6 parameters: ")
            print(f"Teff={rave_data['teff']:.0f}±{rave_data['teff_err']:.0f}K")
            print(f"logg={rave_data['logg']:.2f}±{rave_data['logg_err']:.2f}")
            print(f"[Fe/H]={rave_data['feh']:.2f}±{rave_data['feh_err']:.2f}")

            return rave_data

        except Exception as e:
            logger.warning('RAVE DR6 query failed: %s', e)

        return None

    def query_apogee_params(self):
        """Query APOGEE DR17 for stellar parameters.

        Returns dict with teff, logg, feh and their errors, or None if not found.
        """
        import numpy.ma as ma

        # Prefer 2MASS ID from crossmatch; fall back to Gaia EDR3 source_id
        tmass_id = None
        if isinstance(self.ids, dict) and '2MASS' in self.ids:
            tid = self.ids['2MASS']
            if tid and tid != 'skipped':
                tmass_id = str(tid)

        try:
            v = Vizier(columns=['**'], row_limit=5)
            if tmass_id is not None:
                cat = v.query_constraints(catalog='III/286/allstar',
                                          APOGEE=tmass_id)
            else:
                # Try Gaia EDR3 source_id
                cat = v.query_constraints(catalog='III/286/allstar',
                                          GaiaEDR3=str(self.g_id))

            if not cat or len(cat) == 0 or len(cat[0]) == 0:
                return None

            row = cat[0][0]

            # Quality: skip if ASPCAPFLAG has STAR_BAD (bit 23)
            aflag = row['AFlag']
            if not ma.is_masked(aflag) and int(aflag) & (1 << 23):
                print('APOGEE DR17: STAR_BAD flag set, skipping.')
                return None

            teff = row['Teff']
            e_teff = row['e_Teff']
            logg_val = row['logg']
            e_logg = row['e_logg']
            mh = row['[M/H]']
            e_mh = row['e_[M/H]']

            # Check for masked values
            for val in [teff, e_teff, logg_val, e_logg, mh, e_mh]:
                if ma.is_masked(val):
                    return None

            data = {
                'teff': float(teff),
                'teff_err': float(e_teff),
                'logg': float(logg_val),
                'logg_err': float(e_logg),
                'feh': float(mh),
                'feh_err': float(e_mh),
            }

            print('APOGEE DR17 parameters:')
            print(f"Teff={data['teff']:.0f}+/-{data['teff_err']:.0f}K")
            print(f"logg={data['logg']:.2f}+/-{data['logg_err']:.2f}")
            print(f"[M/H]={data['feh']:.2f}+/-{data['feh_err']:.2f}")

            return data

        except Exception as e:
            logger.warning('APOGEE DR17 query failed: %s', e)

        return None

    def query_galah_params(self):
        """Query GALAH DR3 for stellar parameters.

        Returns dict with teff, logg, feh and their errors, or None if not found.
        """
        import numpy.ma as ma

        try:
            v = Vizier(columns=['**'], row_limit=5)
            cat = v.query_constraints(
                catalog='J/MNRAS/506/150/catalog',
                GaiaEDR3=str(self.g_id)
            )

            if not cat or len(cat) == 0 or len(cat[0]) == 0:
                return None

            row = cat[0][0]

            # Quality: flag_sp must be 0
            flag_sp = row['Flagsp']
            if not ma.is_masked(flag_sp) and int(flag_sp) != 0:
                print('GALAH DR3: flag_sp != 0, skipping.')
                return None

            teff = row['Teff']
            e_teff = row['e_Teff']
            logg_val = row['logg']
            e_logg = row['e_logg']
            feh = row['[Fe/H]']
            e_feh = row['e_[Fe/H]']

            for val in [teff, e_teff, logg_val, e_logg, feh, e_feh]:
                if ma.is_masked(val):
                    return None

            data = {
                'teff': float(teff),
                'teff_err': float(e_teff),
                'logg': float(logg_val),
                'logg_err': float(e_logg),
                'feh': float(feh),
                'feh_err': float(e_feh),
            }

            print('GALAH DR3 parameters:')
            print(f"Teff={data['teff']:.0f}+/-{data['teff_err']:.0f}K")
            print(f"logg={data['logg']:.2f}+/-{data['logg_err']:.2f}")
            print(f"[Fe/H]={data['feh']:.2f}+/-{data['feh_err']:.2f}")

            return data

        except Exception as e:
            logger.warning('GALAH DR3 query failed: %s', e)

        return None

    def query_lamost_params(self):
        """Query LAMOST DR5 stellar parameter catalog (AFGK stars).

        Returns dict with teff, logg, feh and their errors, or None if not found.
        """
        import numpy.ma as ma

        try:
            coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg,
                             frame='icrs')
            v = Vizier(columns=['**'], row_limit=10)
            # V/164/stellar5 = LAMOST DR5 AFGK stellar parameters
            cat = v.query_region(coord, radius=5 * u.arcsec,
                                 catalog='V/164/stellar5')

            if not cat or len(cat) == 0 or len(cat[0]) == 0:
                return None

            tab = cat[0]
            tab.sort('_r')

            # Filter by SNR in g-band > 30
            for row in tab:
                snrg = row['snrg']
                if ma.is_masked(snrg) or float(snrg) < 30:
                    continue

                teff = row['Teff']
                e_teff = row['e_Teff']
                logg_val = row['logg']
                e_logg = row['e_logg']
                feh = row['[Fe/H]']
                e_feh = row['e_[Fe/H]']

                has_masked = False
                for val in [teff, e_teff, logg_val, e_logg, feh, e_feh]:
                    if ma.is_masked(val):
                        has_masked = True
                        break
                if has_masked:
                    continue

                data = {
                    'teff': float(teff),
                    'teff_err': float(e_teff),
                    'logg': float(logg_val),
                    'logg_err': float(e_logg),
                    'feh': float(feh),
                    'feh_err': float(e_feh),
                }

                print('LAMOST DR5 parameters:')
                print(f"Teff={data['teff']:.0f}+/-{data['teff_err']:.0f}K")
                print(f"logg={data['logg']:.2f}+/-{data['logg_err']:.2f}")
                print(f"[Fe/H]={data['feh']:.2f}+/-{data['feh_err']:.2f}")

                return data

        except Exception as e:
            logger.warning('LAMOST DR5 query failed: %s', e)

        return None

    def query_pastel_params(self):
        """Query PASTEL catalog for stellar parameters.

        PASTEL compiles literature Teff/logg/[Fe/H] determinations.
        Returns dict with teff, logg, feh and their errors, or None if not found.
        Errors may be absent in PASTEL; in that case conservative defaults
        are assigned.
        """
        import numpy.ma as ma

        try:
            coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg,
                             frame='icrs')
            v = Vizier(columns=['**'], row_limit=50)
            cat = v.query_region(coord, radius=5 * u.arcsec,
                                 catalog='B/pastel/pastel')

            if not cat or len(cat) == 0 or len(cat[0]) == 0:
                return None

            tab = cat[0]

            # PASTEL has multiple entries per star (one per literature source).
            # Keep only rows with Teff present, then take the median.
            teffs, loggs, fehs = [], [], []
            e_teffs, e_loggs, e_fehs = [], [], []

            for row in tab:
                t = row['Teff']
                if ma.is_masked(t):
                    continue
                teffs.append(float(t))
                et = row['e_Teff']
                e_teffs.append(float(et) if not ma.is_masked(et) else np.nan)

                lg = row['logg']
                if not ma.is_masked(lg):
                    loggs.append(float(lg))
                    elg = row['e_logg']
                    e_loggs.append(
                        float(elg) if not ma.is_masked(elg) else np.nan)

                fe = row['[Fe/H]']
                if not ma.is_masked(fe):
                    fehs.append(float(fe))
                    efe = row['e_[Fe/H]']
                    e_fehs.append(
                        float(efe) if not ma.is_masked(efe) else np.nan)

            if len(teffs) == 0:
                return None

            teff_med = float(np.median(teffs))
            # Use reported error if available, else scatter, else 100 K default
            e_teff_vals = [v for v in e_teffs if not np.isnan(v)]
            teff_err = (float(np.median(e_teff_vals)) if e_teff_vals
                        else (float(np.std(teffs)) if len(teffs) > 1
                              else 100.0))

            logg_med = float(np.median(loggs)) if loggs else np.nan
            e_logg_vals = [v for v in e_loggs if not np.isnan(v)]
            logg_err = (float(np.median(e_logg_vals)) if e_logg_vals
                        else (float(np.std(loggs)) if len(loggs) > 1
                              else 0.2))

            feh_med = float(np.median(fehs)) if fehs else np.nan
            e_feh_vals = [v for v in e_fehs if not np.isnan(v)]
            feh_err = (float(np.median(e_feh_vals)) if e_feh_vals
                       else (float(np.std(fehs)) if len(fehs) > 1
                             else 0.1))

            if np.isnan(logg_med) or np.isnan(feh_med):
                return None

            data = {
                'teff': teff_med,
                'teff_err': max(teff_err, 50.0),  # floor at 50 K
                'logg': logg_med,
                'logg_err': max(logg_err, 0.05),  # floor at 0.05 dex
                'feh': feh_med,
                'feh_err': max(feh_err, 0.05),  # floor at 0.05 dex
            }

            print(f'PASTEL parameters (median of {len(teffs)} entries):')
            print(f"Teff={data['teff']:.0f}+/-{data['teff_err']:.0f}K")
            print(f"logg={data['logg']:.2f}+/-{data['logg_err']:.2f}")
            print(f"[Fe/H]={data['feh']:.2f}+/-{data['feh_err']:.2f}")

            return data

        except Exception as e:
            logger.warning('PASTEL query failed: %s', e)

        return None

    def query_spectroscopic_params(self):
        """Query multiple spectroscopic catalogs in priority order.

        Priority: APOGEE > GALAH > RAVE > LAMOST > PASTEL.
        Returns dict with teff, logg, feh + errors + source, or None.
        """
        for method, source in [
            (self.query_apogee_params, 'APOGEE_DR17'),
            (self.query_galah_params, 'GALAH_DR3'),
            (self.query_rave_params, 'RAVE_DR6'),
            (self.query_lamost_params, 'LAMOST_DR5'),
            (self.query_pastel_params, 'PASTEL'),
        ]:
            try:
                result = method()
                if result is not None:
                    result['source'] = source
                    return result
            except Exception as e:
                logger.warning('%s query failed: %s', source, e)
        return None

    def _query_gaia_catalogs(self):
        """Query Gaia DR3 for catalog crossmatches."""
        # Table mappings DR2→DR3
        catalog_map = {
            'tycho2tdsc_merge': 'TYCHO2',
            'panstarrs1': 'Pan-STARRS',
            'sdssdr13': 'SDSS',
            'allwise': 'Wise',
            'tmass_psc_xsc': '2MASS',
            'apassdr9': 'APASS',
            'ravedr6': 'RAVE',
            'skymapperdr2': 'SkyMapper'
        }

        IDS = {
            'TYCHO2': '',
            'APASS': '',
            '2MASS': '',
            'Pan-STARRS': '',
            'SDSS': '',
            'Wise': '',
            'Gaia': self.g_id,
            'SkyMapper': '',
            'RAVE': '',
        }

        gaia_tap_down = False
        xmatch_needed = []

        for table_name, catalog_name in catalog_map.items():
            if catalog_name in self.ignore:
                IDS[catalog_name] = 'skipped'
                CatalogWarning(catalog_name, 7).warn()
                continue

            if gaia_tap_down:
                xmatch_needed.append(catalog_name)
                continue

            query = f"""
                SELECT xmatch.original_ext_source_id
                FROM gaiadr3.{table_name}_best_neighbour AS xmatch
                WHERE xmatch.source_id = {self.g_id}
            """

            try:
                j = Gaia.launch_job_async(query)
                r = j.get_results()
                if len(r):
                    IDS[catalog_name] = r[0][0]
                else:
                    IDS[catalog_name] = 'skipped'
                    print(f'Star not found in catalog {catalog_name}', end='.\n')
            except Exception as e:
                err_str = str(e).lower()
                is_service_error = any(s in err_str for s in [
                    'service unavailable', 'bad gateway', '503', '502', '500',
                    'connection', 'timeout', 'unavailable',
                ])
                if is_service_error:
                    gaia_tap_down = True
                    xmatch_needed.append(catalog_name)
                IDS[catalog_name] = 'skipped'
                logger.warning('Error querying %s: %s', catalog_name, e)

        if xmatch_needed:
            logger.warning('Gaia TAP down, trying VizieR XMatch for: %s', xmatch_needed)
            self._xmatch_fallback(IDS, xmatch_needed)

        IDS['GALEX'] = ''
        IDS['TESS'] = ''
        IDS['MERMILLIOD'] = ''
        IDS['STROMGREN_PAUNZ'] = ''
        IDS['STROMGREN_HAUCK'] = ''
        return IDS

    def gaia_query(self):
        """Query Gaia DR3 (with DR2 fallback) for catalog IDs."""
        print("Querying Gaia DR3 for catalog crossmatches...")
        self.ids = self._query_gaia_catalogs()
        return

    def _xmatch_fallback(self, IDS, catalogs_needed):
        """Use VizieR XMatch to recover catalog IDs when Gaia TAP is down.

        Pattern A catalogs: extract the catalog ID so the normal get_magnitudes()
        path can fetch photometry from the Vizier cone search results.
        Pattern B (APASS): pull magnitudes directly from the XMatch row because
        _get_apass_from_gaia() would hit Gaia TAP again on its own.
        """
        coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=self.radius)

        # Catalogs not available on XMatch server -- use VizieR cone search instead
        CONE_ONLY = {
            'RAVE':      ('III/283/madera', 'ObsID'),
            'SkyMapper': ('II/379',         ['ObjectId', 'object_id']),
        }

        # (vizier_catalog, id_columns_to_try)  -- None means special handling
        xmatch_config = {
            '2MASS':      ('vizier:II/246/out',      ['_2MASS', '2MASS', '_2M']),
            'Wise':       ('vizier:II/328/allwise',  ['AllWISE']),
            'Pan-STARRS': ('vizier:II/349/ps1',      ['objID']),
            'SDSS':       ('vizier:V/147/sdss12',    ['objID']),
            'TYCHO2':     ('vizier:I/259/tyc2',      None),   # composite TYC1-TYC2-TYC3
            'APASS':      ('vizier:II/336/apass9',   None),   # Pattern B: extract mags
        }

        for catalog_name in catalogs_needed:
            # Cone-search-only catalogs (not available via XMatch)
            if catalog_name in CONE_ONLY:
                viz_cat, id_col = CONE_ONLY[catalog_name]
                try:
                    result = Vizier.query_region(
                        SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame='icrs'),
                        radius=self.radius,
                        catalog=viz_cat,
                    )
                    if not result or len(result[0]) == 0:
                        print('VizieR cone: no match for {}'.format(catalog_name))
                        IDS[catalog_name] = 'skipped'
                        continue
                    result[0].sort('_r')
                    row = result[0][0]
                    cols = [id_col] if isinstance(id_col, str) else id_col
                    found_id = next((row[c] for c in cols if c in result[0].colnames), None)
                    if found_id is None:
                        logger.warning('VizieR cone: no ID column found for %s', catalog_name)
                        IDS[catalog_name] = 'skipped'
                    else:
                        IDS[catalog_name] = found_id
                        print('VizieR cone: {} ID = {}'.format(catalog_name, found_id))
                except Exception as e:
                    logger.warning('VizieR cone fallback failed for %s: %s', catalog_name, e)
                    IDS[catalog_name] = 'skipped'
                continue

            if catalog_name not in xmatch_config:
                continue
            vizier_cat, id_cols = xmatch_config[catalog_name]

            try:
                xm = XMatch.query(
                    cat1='vizier:I/355/gaiadr3',
                    cat2=vizier_cat,
                    max_distance=self.radius,
                    area=region,
                )
                if xm is None or len(xm) == 0:
                    print('VizieR XMatch fallback: no match for {}'.format(catalog_name))
                    IDS[catalog_name] = 'skipped'
                    continue

                xm.sort('angDist')

                # Prefer the row that matches our exact Gaia source ID
                row = xm[0]
                if 'Source' in xm.colnames:
                    mask = xm['Source'] == self.g_id
                    if mask.sum() > 0:
                        row = xm[mask][0]

                # Pattern B: APASS -- extract magnitudes directly
                if catalog_name == 'APASS':
                    IDS['APASS'] = 'xmatch_done'
                    added = False
                    for mag_col, err_col, filt in self.catalogs['APASS'][1]:
                        filt_idx = np.where(filt == self.filter_names)[0]
                        if self.used_filters[filt_idx] == 1:
                            CatalogWarning(filt, 6).warn()
                            continue
                        try:
                            mag = row[mag_col]
                            err = row[err_col]
                        except (KeyError, IndexError):
                            continue
                        if not self._qc_mags(mag, err, mag_col):
                            continue
                        self._add_mags(mag, err, filt)
                        added = True
                    if added:
                        print('VizieR XMatch fallback: APASS mags added')
                    continue

                # Pattern A: TYCHO2 -- compose hyphen-separated ID
                if catalog_name == 'TYCHO2':
                    try:
                        tyc_id = ('{}-{}-{}'.format(int(row['TYC1']),
                                                    int(row['TYC2']),
                                                    int(row['TYC3'])))
                        IDS['TYCHO2'] = tyc_id
                        print('VizieR XMatch fallback: TYCHO2 ID = {}'.format(tyc_id))
                    except (KeyError, ValueError):
                        IDS['TYCHO2'] = 'skipped'
                    continue

                # Pattern A: generic -- take first matching ID column
                found_id = None
                for col in id_cols:
                    if col in xm.colnames:
                        found_id = row[col]
                        break

                if found_id is None:
                    logger.warning('VizieR XMatch fallback: no ID column for %s', catalog_name)
                    IDS[catalog_name] = 'skipped'
                else:
                    IDS[catalog_name] = found_id
                    print('VizieR XMatch fallback: {} ID = {}'.format(catalog_name, found_id))

            except Exception as e:
                logger.warning('VizieR XMatch fallback failed for %s: %s', catalog_name, e)
                IDS[catalog_name] = 'skipped'

    def get_magnitudes(self):
        """Retrieve the magnitudes of the star.

        Looks into APASS, WISE, Pan-STARRS, Gaia, 2MASS and SDSS surveys
        looking for different magnitudes for the star, along with the
        associated uncertainties.
        """
        print('Looking online for archival magnitudes for star', end=' ')
        print(self.starname)
        catalogs = [c[1][0] for c in self.catalogs.items() if c[1][0] is not None]
        cats = self.get_catalogs(self.ra, self.dec, self.radius, catalogs)
        skips = ['ASCC', 'GLIMPSE']

        for c in self.catalogs.keys():
            if c in skips:
                continue
            if c in self.ignore:
                CatalogWarning(c, 7).warn()
                continue

            # Skip if ID was not found (XMatch fallback already attempted in _query_gaia_catalogs)
            if self.ids[c] == 'skipped':
                continue

            # Load catalog from Vizier results (except for special cases handled below)
            if c not in ['TESS', 'APASS', 'SkyMapper']:
                try:
                    current_cat = cats[self.catalogs[c][0]]
                    current_cat.sort('_r')
                except TypeError:
                    CatalogWarning(c, 5).warn()
                    continue
            elif c == 'TESS':
                self._retrieve_from_tess()
                continue
            elif c == 'APASS':
                self._get_apass_from_gaia()
                continue
            elif c == 'SkyMapper':
                self._get_skymapper_from_tap()
                continue

            # Handle specific catalog types
            if c == 'Wise':
                self._get_wise(current_cat)
                continue
            elif c == 'TYCHO2':
                self._get_ascc_tycho2_stromgren(cats, False, 'TYCHO2')
                self._get_ascc_tycho2_stromgren(cats, False, 'ASCC')
                continue
            elif c == 'SDSS':
                self._get_sdss(current_cat)
                continue
            elif c == 'Pan-STARRS':
                self._get_ps1(current_cat)
                continue
            elif c == 'Gaia':
                self._get_gaia(current_cat)
                continue
            elif c == '2MASS':
                self._get_2mass_glimpse(cats, '2MASS')
                self._get_2mass_glimpse(cats, 'GLIMPSE')
                continue
            elif c == 'GALEX':
                current_cat = self._gaia_galex_xmatch(cats, self.ra, self.dec,
                                                      self.radius)
                if len(current_cat) == 0:
                    CatalogWarning(c, 5).warn()
                    continue
                self._retrieve_from_galex(current_cat, c)
                continue
            elif c == 'MERMILLIOD':
                current_cat = self._gaia_mermilliod_xmatch(self.ra, self.dec,
                                                           self.radius)
                if len(current_cat) == 0:
                    CatalogWarning(c, 5).warn()
                    continue
                self._retrieve_from_mermilliod(current_cat)
                continue
            elif c == 'STROMGREN_PAUNZ':
                current_cat = self._gaia_paunzen_xmatch(self.ra, self.dec,
                                                        self.radius)
                if len(current_cat) == 0:
                    CatalogWarning(c, 5).warn()
                    continue
                self._retrieve_from_stromgren(current_cat, 'STROMGREN_PAUNZEN')
                continue
            elif c == 'STROMGREN_HAUCK':
                current_cat = self._gaia_hauck_xmatch(self.ra, self.dec,
                                                      self.radius)
                if len(current_cat) == 0:
                    CatalogWarning(c, 5).warn()
                    continue
                self._retrieve_from_stromgren(current_cat, 'STROMGREN_HAUCK')
                continue
        pass

    def _retrieve_from_tess(self):
        print('Checking catalog TICv8')
        tic = self.get_TIC(self.ra, self.dec, self.radius)
        tic.sort('dstArcSec')
        mask = tic['GAIA'] == str(self.g_id)
        cat = tic[mask]
        if len(cat) > 0:
            is_star = cat['objType'][0] == 'STAR'
            if not is_star:
                CatalogWarning('TESS', 8).warn()
                return
            self.tic = int(cat['ID'][0])
            kic = cat['KIC'][0]
            self.kic = int(kic) if not np.ma.is_masked(kic) else None
            m, e, f = self.catalogs['TESS'][1][0]
            filt_idx = np.where(f == self.filter_names)[0]
            if self.used_filters[filt_idx] == 1:
                CatalogWarning(f, 6).warn()
                return
            mag = cat[m][0]
            err = cat[e][0]
            if not self._qc_mags(mag, err, m):
                return

            self._add_mags(mag, err, f)
        else:
            CatalogWarning('TIC', 5).warn()

    def _retrieve_from_cat(self, cat, name):
        if len(cat):
            for m, e, f in self.catalogs[name][1]:
                filt_idx = np.where(f == self.filter_names)[0]

                if self.used_filters[filt_idx] == 1:
                    CatalogWarning(f, 6).warn()
                    continue
                mag = cat[m][0]
                err = cat[e][0]
                if not self._qc_mags(mag, err, m):
                    continue

                self._add_mags(mag, err, f)
        else:
            CatalogWarning(name, 5).warn()

    def _retrieve_from_mermilliod(self, cat):
        print('Checking catalog Mermilliod')
        if self.g_id is None:
            CatalogWarning('MERMILLIOD', 5).warn()
            return
        mask = cat['Source'] == self.g_id
        matched = cat[mask]
        if len(matched) == 0:
            CatalogWarning('MERMILLIOD', 1).warn()
            return
        cat = matched[0]
        v = cat['Vmag']
        v_e = cat['e_Vmag']
        bv = cat['B-V']
        bv_e = cat['e_B-V']
        ub = cat['U-B']
        ub_e = cat['e_U-B']
        if not self._qc_mags(v, v_e, 'vmag'):
            return
        filts = ['GROUND_JOHNSON_V']
        mags = [v]
        err = [v_e]
        if self._qc_mags(bv, bv_e, 'B-V'):
            b = bv + v
            b_e = np.sqrt(v_e ** 2 + bv_e ** 2)
            filts.append('GROUND_JOHNSON_B')
            mags.append(b)
            err.append(b_e)
        if self._qc_mags(ub, ub_e, 'U-B'):
            u = ub + b
            u_e = np.sqrt(b_e ** 2 + ub_e ** 2)
            filts.append('GROUND_JOHNSON_U')
            mags.append(u)
            err.append(u_e)
        for m, e, f in zip(mags, err, filts):
            filt_idx = np.where(f == self.filter_names)[0]

            if self.used_filters[filt_idx] == 1:
                CatalogWarning(f, 6).warn()
                continue
            self._add_mags(m, e, f)

    def _retrieve_from_stromgren(self, cat, n):
        print('Checking catalog ' + n)
        if self.g_id is None:
            CatalogWarning(n, 5).warn()
            return
        mask = cat['Source'] == self.g_id
        matched = cat[mask]
        if len(matched) == 0:
            CatalogWarning(n, 1).warn()
            return
        cat = matched[0]
        y = cat['Vmag']
        y_e = cat['e_Vmag']
        if not self._qc_mags(y, y_e, 'ymag'):
            return
        if np.isnan(y_e):
            y_e = 0
        by = cat['b-y']
        by_e = cat['e_b-y']
        m1 = cat['m1']
        m1_e = cat['e_m1']
        c1 = cat['c1']
        c1_e = cat['e_c1']
        b = by + y
        v = m1 + 2 * by + y
        u = c1 + 2 * m1 + 3 * by + y
        b_e = np.sqrt(by_e ** 2 + y_e ** 2)
        v_e = np.sqrt(m1_e ** 2 + 4 * by_e ** 2 + y_e ** 2)
        u_e = np.sqrt(c1_e ** 2 + 4 * m1_e ** 2 + 9 * by_e ** 2 + y_e ** 2)
        mags = [u, v, b, y]
        err = [u_e, v_e, b_e, y_e]
        filts = ['STROMGREN_u', 'STROMGREN_v', 'STROMGREN_b', 'STROMGREN_y']
        for m, e, f in zip(mags, err, filts):
            filt_idx = np.where(f == self.filter_names)[0]
            if self.used_filters[filt_idx] == 1:
                CatalogWarning(f, 6).warn()
                continue
            self._add_mags(m, e, f)
        pass

    def _retrieve_from_galex(self, cat, name):
        print('Checking catalog GALEX')
        mask = cat['Source'] == self.g_id
        if mask.sum() == 0:
            CatalogWarning('GALEX', 0).warn()
            return
        cat = cat[mask][0]
        Fexf = cat['Fexf']
        Nexf = cat['Nexf']
        Fafl = cat['Fafl']
        Nafl = cat['Nafl']
        for m, e, f in self.catalogs[name][1]:
            if f == 'GALEX_FUV' and (Fexf > 0 or Fafl > 0):
                CatalogWarning(f, 8).warn()
                continue
            if f == 'GALEX_NUV' and (Nexf > 0 or Nafl > 0):
                CatalogWarning(f, 8).warn()
                continue
            filt_idx = np.where(f == self.filter_names)[0]

            if self.used_filters[filt_idx] == 1:
                CatalogWarning(f, 6).warn()
                continue
            mag = cat[m]
            err = cat[e]
            if not self._qc_mags(mag, err, m):
                continue

            self._add_mags(mag, err, f)

    def _retrieve_from_2mass(self, cat, name):
        qflg = cat['Qflg']
        cflg = cat['Cflg']
        for m, e, f in self.catalogs[name][1]:
            filt_idx = np.where(f == self.filter_names)[0]

            if f == '2MASS_J':
                if qflg[0][0] not in 'ABCD' or cflg[0][0] != '0':
                    CatalogWarning(f, 8).warn()
                    continue
            if f == '2MASS_H':
                if qflg[0][1] not in 'ABCD' or cflg[0][1] != '0':
                    CatalogWarning(f, 8).warn()
                    continue
            if f == '2MASS_Ks':
                if qflg[0][2] not in 'ABCD' or cflg[0][2] != '0':
                    CatalogWarning(f, 8).warn()
                    continue

            if self.used_filters[filt_idx] == 1:
                CatalogWarning(f, 6).warn()
                continue
            mag = cat[m][0]
            err = cat[e][0]
            if not self._qc_mags(mag, err, m):
                continue

            self._add_mags(mag, err, f)

    def _retrieve_from_wise(self, cat, name):
        qph = cat['qph']
        for m, e, f in self.catalogs[name][1]:
            filt_idx = np.where(f == self.filter_names)[0]

            if f == 'WISE_RSR_W1':
                if qph[0][0] not in 'ABC':
                    CatalogWarning(f, 8).warn()
                    continue
            if f == 'WISE_RSR_W2':
                if qph[0][1] not in 'ABC':
                    CatalogWarning(f, 8).warn()
                    continue

            if self.used_filters[filt_idx] == 1:
                CatalogWarning(f, 6).warn()
                continue
            mag = cat[m][0]
            err = cat[e][0]
            if not self._qc_mags(mag, err, m):
                continue

            self._add_mags(mag, err, f)

    def _add_mags(self, mag, er, filt):
        filt_idx = np.where(filt == self.filter_names)[0]
        if er == 0 or np.ma.is_masked(er):
            self.used_filters[filt_idx] = 2
        else:
            self.used_filters[filt_idx] = 1
        self.mags[filt_idx] = mag
        self.mag_errs[filt_idx] = er

    def _get_ascc_tycho2_stromgren(self, cats, near, name):
        print('Checking catalog ' + name)
        try:
            cat = cats[self.catalogs[name][0]]
            cat.sort('_r')
        except TypeError:
            CatalogWarning(name, 5).warn()
            return
        if not near:
            try:
                tyc1, tyc2, tyc3 = self.ids['TYCHO2'].split('-')
            except TypeError:
                tyc1, tyc2, tyc3 = self.ids['TYCHO2'].split('b-')
            mask = cat['TYC1'] == int(tyc1)
            mask *= cat['TYC2'] == int(tyc2)
            mask *= cat['TYC3'] == int(tyc3)
        else:
            mask = [0]
        if 'STROMGREN' not in name:
            self._retrieve_from_cat(cat[mask], name)
        else:
            self._retrieve_from_stromgren(cat[mask])

    def _get_apass_from_gaia(self):
        """Query APASS photometry directly from Gaia external.apassdr9 table.

        This method queries Gaia's external catalog tables directly via TAP,
        avoiding the broken Vizier recno matching.
        """
        print('Checking catalog APASS')

        # Skip if APASS ID not found in crossmatch
        if self.ids.get('APASS') in ('skipped', 'xmatch_done') or not self.ids.get('APASS'):
            CatalogWarning('APASS', 5).warn()
            return

        # Query external.apassdr9 table via Gaia TAP with JOIN
        query = f"""
            SELECT
                apass.recno,
                apass.vmag,
                apass.e_vmag,
                apass.u_e_vmag,
                apass.bmag,
                apass.e_bmag,
                apass.u_e_bmag,
                apass.g_mag,
                apass.e_g_mag,
                apass.u_e_g_mag,
                apass.r_mag,
                apass.e_r_mag,
                apass.u_e_r_mag,
                apass.i_mag,
                apass.e_i_mag,
                apass.u_e_i_mag
            FROM
                external.apassdr9 AS apass
            INNER JOIN
                gaiadr3.apassdr9_best_neighbour AS xmatch
            ON
                apass.recno = xmatch.original_ext_source_id
            WHERE
                xmatch.source_id = {self.g_id}
        """

        try:
            job = Gaia.launch_job_async(query)
            result = job.get_results()

            if len(result) == 0:
                CatalogWarning('APASS', 5).warn()
                return

            # Extract first row (should be only one match)
            row = result[0]

            # Process each magnitude/error pair
            for mag_col, err_col, filt in self.catalogs['APASS'][1]:
                filt_idx = np.where(filt == self.filter_names)[0]

                # Check if magnitude already retrieved
                if self.used_filters[filt_idx] == 1:
                    CatalogWarning(filt, 6).warn()
                    continue

                # Get magnitude and error from result
                mag = row[mag_col]
                err = row[err_col]

                # Quality check the magnitude
                if not self._qc_mags(mag, err, mag_col):
                    continue

                # Add magnitude to arrays
                self._add_mags(mag, err, filt)

        except Exception as e:
            logger.warning('Error querying APASS from Gaia: %s', e)
            CatalogWarning('APASS', 5).warn()

    def _get_wise(self, cat):
        print('Checking catalog All-WISE')
        mask = cat['AllWISE'] == self.ids['Wise']
        is_star = cat[mask]['ex'] == 0
        if is_star:
            self._retrieve_from_wise(cat[mask], 'Wise')
        else:
            CatalogWarning('WISE', 8).warn()

    def _get_2mass_glimpse(self, cats, name):
        print('Checking catalog ' + name)
        try:
            cat = cats[self.catalogs[name][0]]
            cat.sort('_r')
        except TypeError:
            CatalogWarning(name, 5).warn()
            return

        # Try multiple possible column names for robustness against Vizier schema changes
        # Changed 2024-12: Vizier renamed '_2MASS' to '2MASS' (removed underscore)
        possible_columns = ['2MASS', '_2MASS', '_2M', 'Designation']
        column_to_use = None

        for col in possible_columns:
            if col in cat.colnames:
                column_to_use = col
                break

        if column_to_use is None:
            warning_msg = f"No 2MASS identifier column found in Vizier results. "
            warning_msg += f"Available columns: {cat.colnames}"
            CatalogWarning(name, 6, warning_msg).warn()
            return

        if name == '2MASS':
            mask = cat[column_to_use] == self.ids['2MASS']
            self._retrieve_from_2mass(cat[mask], '2MASS')
        else:
            mask = cat[column_to_use] == self.ids['2MASS']
            self._retrieve_from_cat(cat[mask], 'GLIMPSE')

    def _get_sdss(self, cat):
        print('Checking catalog SDSS DR12')
        mask = cat['objID'] == int(self.ids['SDSS'])
        is_star = cat[mask]['class'] == 6
        is_good_quality = cat[mask]['Q'] == 3 or cat[mask]['Q'] == 2
        if is_star and is_good_quality:
            self._retrieve_from_cat(cat[mask], 'SDSS')
        else:
            CatalogWarning('SDSS', 8).warn()

    def _get_ps1(self, cat):
        print('Checking catalog Pan-STARRS1')
        mask = cat['objID'] == self.ids['Pan-STARRS']
        is_star = not (cat[mask]['Qual'] & 1 and cat[mask]['Qual'] & 2)
        is_good_quality = (cat[mask]['Qual'] & 4 or cat[mask]['Qual'] & 16)
        is_good_quality = is_good_quality and not cat[mask]['Qual'] & 128
        if is_star and is_good_quality:
            self._retrieve_from_cat(cat[mask], 'Pan-STARRS')
        else:
            CatalogWarning('Pan-STARRS', 8).warn()

    def _get_gaia(self, cat):
        print('Checking catalog Gaia DR3')
        mask = cat['DR3Name'] == f'Gaia DR3 {self.ids["Gaia"]}'
        self._retrieve_from_cat(cat[mask], 'Gaia')

    def _get_skymapper_from_tap(self):
        """Query SkyMapper DR2 photometry directly from SkyMapper TAP service.

        SkyMapper DR2 is not available in Vizier, so we query their TAP service directly.
        """
        print('Checking catalog SkyMapper DR2')

        # Skip if SkyMapper ID not found in crossmatch
        if self.ids.get('SkyMapper') == 'skipped' or not self.ids.get('SkyMapper'):
            CatalogWarning('SkyMapper', 5).warn()
            return

        # Query SkyMapper DR2 via TAP service
        from astroquery.utils.tap.core import TapPlus

        tap_url = 'https://api.skymapper.nci.org.au/public/tap/'
        skymapper_tap = TapPlus(url=tap_url)

        query = f"""
            SELECT
                sm.object_id,
                sm.raj2000,
                sm.dej2000,
                sm.u_psf,
                sm.e_u_psf,
                sm.v_psf,
                sm.e_v_psf,
                sm.g_psf,
                sm.e_g_psf,
                sm.r_psf,
                sm.e_r_psf,
                sm.i_psf,
                sm.e_i_psf,
                sm.z_psf,
                sm.e_z_psf,
                sm.flags
            FROM
                dr2.master AS sm
            WHERE
                sm.object_id = {self.ids['SkyMapper']}
        """

        try:
            job = skymapper_tap.launch_job_async(query)
            result = job.get_results()

            if len(result) == 0:
                CatalogWarning('SkyMapper', 5).warn()
                return

            # Extract first row
            row = result[0]

            # Check quality flags (0 = good)
            if row['flags'] != 0:
                CatalogWarning('SkyMapper', 8).warn()
                return

            # Process each magnitude/error pair
            for mag_col, err_col, filt in self.catalogs['SkyMapper'][1]:
                filt_idx = np.where(filt == self.filter_names)[0]

                # Check if magnitude already retrieved
                if self.used_filters[filt_idx] == 1:
                    CatalogWarning(filt, 6).warn()
                    continue

                # Get magnitude and error from result
                mag = row[mag_col]
                err = row[err_col]

                # Quality check the magnitude
                if not self._qc_mags(mag, err, mag_col):
                    continue

                # Add magnitude to arrays
                self._add_mags(mag, err, filt)

        except Exception as e:
            logger.warning('Error querying SkyMapper from TAP: %s', e)
            CatalogWarning('SkyMapper', 5).warn()

    # Removed: Old _get_skymapper() method - replaced by _get_skymapper_from_tap()
    # which queries SkyMapper DR2 directly via TAP service

    @staticmethod
    def _get_distance(ra, dec, radius, g_id):
        """Retrieve Bailer-Jones EDR3 distance."""
        print('Querying Bailer-Jones EDR3 distance for source {}...'.format(g_id))
        try:
            res = Vizier.query_constraints(
                catalog='I/352/gedr3dis', Source=str(g_id))
            if not res or len(res[0]) == 0:
                logger.warning('Bailer-Jones: source %s not found in catalog', g_id)
                return -1, -1
            row = res[0][0]
            dist = row['rgeo']
            lo = dist - row['b_rgeo']
            hi = row['B_rgeo'] - dist
            print('Bailer-Jones distance: {:.1f} pc'.format(dist))
            return dist, max(lo, hi)
        except Exception as e:
            logger.warning('Bailer-Jones query failed: %s', e)
            return -1, -1

    @staticmethod
    def _get_parallax(res):
        """Extract parallax from DR3 results."""
        if np.ma.is_masked(res['parallax'][0]):
            CatalogWarning('masked', 0).warn()
            return -1, -1
        plx = res['parallax'][0]
        if plx <= 0:
            CatalogWarning('{:.6f} mas'.format(float(plx)), 0).warn()
            return -1, -1
        plx_e = res['parallax_error'][0]
        # Parallax correction 37.0 ± 20 µas from Lindegren+21
        return plx + 0.037, np.sqrt(plx_e ** 2 + 0.02 ** 2)

    @staticmethod
    def _get_radius(res):
        """Extract radius from DR3 FLAME results."""
        rad = res['radius_flame'][0]
        if np.ma.is_masked(rad):
            CatalogWarning('radius', 1).warn()
            return 0, 0
        lo = res['radius_flame_lower'][0]
        up = res['radius_flame_upper'][0]
        rad_e = max([rad - lo, up - rad])
        return rad, 5 * rad_e

    @staticmethod
    def _get_teff(res):
        """Extract Teff from DR3 GSP-Phot results."""
        teff = res['teff_gspphot'][0]
        if np.ma.is_masked(teff):
            CatalogWarning('teff', 1).warn()
            return 0, 0
        lo = res['teff_gspphot_lower'][0]
        up = res['teff_gspphot_upper'][0]
        teff_e = max([teff - lo, up - teff])
        return teff, teff_e

    @staticmethod
    def _get_lum(res):
        """Extract luminosity from DR3 FLAME results."""
        lum = res['lum_flame'][0]
        if np.ma.is_masked(lum):
            CatalogWarning('lum', 1).warn()
            return 0, 0
        lo = res['lum_flame_lower'][0]
        up = res['lum_flame_upper'][0]
        lum_e = max([lum - lo, up - lum])
        return lum, lum_e

    @staticmethod
    def _get_mass(res):
        """Extract mass from DR3 FLAME results."""
        mass = res['mass_flame'][0]
        if np.ma.is_masked(mass):
            CatalogWarning('mass', 1).warn()
            return 0, 0
        lo = res['mass_flame_lower'][0]
        up = res['mass_flame_upper'][0]
        mass_e = max([mass - lo, up - mass])
        return mass, mass_e

    @staticmethod
    def _get_age(res):
        """Extract age from DR3 FLAME results."""
        age = res['age_flame'][0]
        if np.ma.is_masked(age):
            CatalogWarning('age', 1).warn()
            return 0, 0
        lo = res['age_flame_lower'][0]
        up = res['age_flame_upper'][0]
        age_e = max([age - lo, up - age])
        return age, age_e

    @staticmethod
    def _get_gaia_id(ra, dec, radius):
        """Find the nearest Gaia DR3 source ID via Vizier cone search (no Gaia TAP)."""
        c = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
        cats = Vizier(columns=['Source', '+_r']).query_region(
            c, radius=radius, catalog='I/355/gaiadr3'
        )
        if not cats or len(cats[0]) == 0:
            raise IndexError(f"No Gaia DR3 source found within {radius} of RA={ra}, Dec={dec}")
        cats[0].sort('_r')
        return int(cats[0]['Source'][0])

    @staticmethod
    def get_catalogs(ra, dec, radius, catalogs):
        """Retrieve available catalogs for a star from Vizier."""
        tries = [0.1, 0.25, 0.5, 1][::-1]
        for t in tries:
            cats = Vizier.query_region(
                SkyCoord(
                    ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'
                ), radius=radius / t, catalog=catalogs
            )
            if len(cats):
                break
        return cats

    @staticmethod
    def get_TIC(ra, dec, radius):
        """Retrieve TIC from MAST."""
        cat = Catalogs.query_region(
            SkyCoord(
                ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'
            ), radius=radius, catalog='TIC'
        )

        return cat

    @staticmethod
    def _qc_mags(mag, err, m):
        if np.ma.is_masked(mag):
            CatalogWarning(m, 2).warn()
            return False
        if np.ma.is_masked(err):
            CatalogWarning(m, 3).warn()
            return True
        if err == 0:
            CatalogWarning(m, 4).warn()
            return True
        if err > 1:
            return False
        return True

    @staticmethod
    def _gaia_galex_xmatch(cats, ra, dec, radius):
        galex = cats['II/312/ais']
        coord = SkyCoord(ra=ra * u.deg,
                         dec=dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=radius)
        xm = XMatch.query(cat1='vizier:I/355/gaiadr3', cat2=galex,
                          colRA2='RAJ2000', colDec2='DEJ2000',
                          area=region, max_distance=radius)
        xm.sort('angDist')
        return xm

    @staticmethod
    def _gaia_mermilliod_xmatch(ra, dec, radius):
        coord = SkyCoord(ra=ra * u.deg,
                         dec=dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=5 * u.arcmin)
        xm = XMatch.query(cat1='vizier:I/355/gaiadr3',
                          cat2='vizier:II/168/ubvmeans',
                          colRA2='_RA', colDec2='_DE',
                          area=region, max_distance=3 * u.arcmin)
        xm.sort('angDist')
        return xm

    @staticmethod
    def _gaia_paunzen_xmatch(ra, dec, radius):
        coord = SkyCoord(ra=ra * u.deg,
                         dec=dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=radius)
        xm = XMatch.query(cat1='vizier:I/355/gaiadr3',
                          cat2='vizier:J/A+A/580/A23/catalog',
                          colRA2='RAICRS', colDec2='DEICRS',
                          area=region, max_distance=3 * u.arcmin)
        xm.sort('angDist')
        return xm

    @staticmethod
    def _gaia_hauck_xmatch(ra, dec, radius):
        coord = SkyCoord(ra=ra * u.deg,
                         dec=dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=radius)
        xm = XMatch.query(cat1='vizier:I/355/gaiadr3',
                          cat2='vizier:II/215/catalog',
                          colRA2='_RA.icrs', colDec2='_DE.icrs',
                          area=region, max_distance=radius)
        xm.sort('angDist')
        return xm

    def create_logfile(self):
        """Activate log file."""
        self.old_stdout = sys.stdout

        self.log_file = open(os.getcwd() + '/' +
                             self.starname + 'output.log', 'w+')
        sys.stdout = self.log_file

    def close_logfile(self):
        """Deactivate log file."""
        sys.stdout = self.old_stdout
        self.log_file.close()
