# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*elif/ /^\s*def/
"""Helper class to look up broadband photometry and stellar parameters."""

__all__ = ['Librarian']

import os
import sys
import warnings

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
    __apass_mags = ['Vmag', 'Bmag', 'g_mag', 'r_mag', 'i_mag']
    __apass_errs = ['e_Vmag', 'e_Bmag', 'e_g_mag', 'e_r_mag', 'e_i_mag']
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
    __skymapper_mags = ['uPSF', 'vPSF', 'gPSF', 'rPSF', 'iPSF', 'zPSF']
    __skymapper_errs = ['e_uPSF', 'e_vPSF', 'e_gPSF',
                        'e_rPSF', 'e_iPSF', 'e_zPSF']
    __skymapper_filters = ['SkyMapper_u', 'SkyMapper_v', 'SkyMapper_g',
                           'SkyMapper_r', 'SkyMapper_i', 'SkyMapper_z']

    # APASS DR9, WISE, PAN-STARRS DR1, GAIA DR2, 2MASS, SDSS DR9
    catalogs = {
        'APASS': [
            'II/336/apass9', list(zip(__apass_mags, __apass_errs,
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
            'II/358/smss', list(zip(__skymapper_mags, __skymapper_errs,
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
        self.dr2_id = None

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
            self.get_magnitudes()
            idx = self.used_filters >= 1
            self.used_filters[idx] = 1

        # self.close_logfile()
        pass


    def gaia_params(self):
        """Retrieve parallax, radius, teff and lum from Gaia."""
        # If gaia DR3 id is provided, query by id
        query = f"""
            SELECT
                dr3.parallax, dr3.parallax_error,
                dr3.pmra, dr3.pmra_error,
                dr3.pmdec, dr3.pmdec_error,
                dr3.radial_velocity, dr3.radial_velocity_error,
                dr2.teff_val,
                dr2.teff_percentile_lower,
                dr2.teff_percentile_upper,
                dr2.radius_val,
                dr2.radius_percentile_lower,
                dr2.radius_percentile_upper,
                dr2.lum_val,
                dr2.lum_percentile_lower,
                dr2.lum_percentile_upper,
                dr2.source_id2 AS source_id
            FROM
                gaiadr3.gaia_source AS dr3
            JOIN
                (SELECT
                    n.dr3_source_id AS source_id,
                    n.dr2_source_id AS source_id2,
                    dr2.teff_val,
                    dr2.teff_percentile_lower,
                    dr2.teff_percentile_upper,
                    dr2.radius_val,
                    dr2.radius_percentile_lower,
                    dr2.radius_percentile_upper,
                    dr2.lum_val,
                    dr2.lum_percentile_lower,
                    dr2.lum_percentile_upper
                FROM
                    gaiadr3.dr2_neighbourhood AS n
                JOIN
                    gaiadr2.gaia_source AS dr2
                ON
                    n.dr2_source_id = dr2.source_id
                WHERE
                    n.dr3_source_id = {self.g_id}
                ) AS dr2
            ON
                dr3.source_id = dr2.source_id
            WHERE
                dr3.source_id = {self.g_id}
            """
        j = Gaia.launch_job_async(query)
        res = j.get_results()
        if len(res) > 1:
            res = Table.from_pandas(res.to_pandas().dropna())
        self.dr2_id = res['source_id'][0]
        self.plx, self.plx_e = self._get_parallax(res)
        self.temp, self.temp_e = self._get_teff(res)
        self.rad, self.rad_e = self._get_radius(res)
        self.lum, self.lum_e = self._get_lum(res)
        self.dist, self.dist_e = self._get_distance(self.ra, self.dec,
                                                    self.radius, self.g_id)
        pass

    def gaia_query(self):
        """Query Gaia to get different catalog IDs."""
        cats = ['tycho2', 'panstarrs1', 'sdssdr9',
                'allwise', 'tmass', 'apassdr9']
        names = ['tycho', 'ps', 'sdss', 'allwise', 'tmass', 'apass']
        IDS = {
            'TYCHO2': '',
            'APASS': '',
            '2MASS': '',
            'Pan-STARRS': '',
            'SDSS': '',
            'Wise': '',
            'Gaia': self.g_id,
            'SkyMapper': self.g_id,
        }
        for c, n in zip(cats, names):
            if c == 'apassdr9':
                cat = 'APASS'
            elif c == 'tmass':
                cat = '2MASS'
                c = 'tmass'
            elif c == 'panstarrs1':
                cat = 'Pan-STARRS'
            elif c == 'sdssdr9':
                cat = 'SDSS'
            elif c == 'allwise':
                cat = 'Wise'
            elif c == 'tycho2':
                cat = 'TYCHO2'
            if cat in self.ignore:
                IDS[cat] = 'skipped'
                CatalogWarning(cat, 7).warn()
                continue
            query = f"""
            SELECT
                {n}.original_ext_source_id
            FROM
                gaiadr2.gaia_source AS gaia
            JOIN
                gaiadr2.{c}_best_neighbour AS {n}
            ON gaia.source_id={n}.source_id
            WHERE
                gaia.source_id={self.dr2_id}
            """
            j = Gaia.launch_job_async(query)
            r = j.get_results()
            if len(r):
                IDS[cat] = r[0][0]
            else:
                IDS[cat] = 'skipped'
                print('Star not found in catalog ' + cat, end='.\n')
        IDS['GALEX'] = ''
        IDS['TESS'] = ''
        IDS['MERMILLIOD'] = ''
        IDS['STROMGREN_PAUNZ'] = ''
        IDS['STROMGREN_HAUCK'] = ''
        self.ids = IDS

    def get_magnitudes(self):
        """Retrieve the magnitudes of the star.

        Looks into APASS, WISE, Pan-STARRS, Gaia, 2MASS and SDSS surveys
        looking for different magnitudes for the star, along with the
        associated uncertainties.
        """
        print('Looking online for archival magnitudes for star', end=' ')
        print(self.starname)
        catalogs = [c[1][0] for c in self.catalogs.items()]
        cats = self.get_catalogs(self.ra, self.dec, self.radius, catalogs)
        skips = ['ASCC', 'GLIMPSE']

        for c in self.catalogs.keys():
            if c in skips:
                continue
            if c in self.ignore:
                CatalogWarning(c, 7).warn()
                continue

            if self.ids[c] == 'skipped':
                continue
            if c != 'TESS':
                try:
                    current_cat = cats[self.catalogs[c][0]]
                    current_cat.sort('_r')
                except TypeError:
                    CatalogWarning(c, 5).warn()
                    continue
            else:
                self._retrieve_from_tess()
                continue
            if c == 'APASS':
                self._get_apass(current_cat)
                continue
            elif c == 'Wise':
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
            elif c == 'SkyMapper':
                self._get_skymapper(current_cat)
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
        mask = cat['source_id'] == self.dr2_id
        cat = cat[mask][0]
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
        mask = cat['source_id'] == self.dr2_id
        cat = cat[mask][0]
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
        mask = cat['source_id'] == self.dr2_id
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

    def _get_apass(self, cat):
        print('Checking catalog APASS')
        CatalogWarning('APASS', 5).warn()
        # mask = cat['recno'] == int(self.ids['APASS'])
        # self._retrieve_from_cat(cat[mask], 'APASS')

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
        if name == '2MASS':
            mask = cat['_2MASS'] == self.ids['2MASS']
            self._retrieve_from_2mass(cat[mask], '2MASS')
        else:
            mask = cat['_2MASS'] == self.ids['2MASS']
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

    def _get_skymapper(self, cat):
        print('Checking catalog SkyMapper DR1.1')
        mask = cat['Gaiadr2Id1'] == self.ids['Gaia']
        is_good_quality = cat[mask]['flags'] == 0
        if is_good_quality:
            self._retrieve_from_cat(cat[mask], 'SkyMapper')
        else:
            CatalogWarning('SkyMapper', 8).warn()

    @staticmethod
    def _get_distance(ra, dec, radius, g_id):
        """Retrieve Bailer-Jones EDR3 distance."""
        tries = [0.5, 0.25, 0.1, 1, 2, 3, 4][::-1]
        for t in tries:
            try:
                failed = False
                cat = Vizier.query_region(
                    SkyCoord(
                        ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'),
                    radius=radius / t,
                    catalog='I/352/gedr3dis')['I/352/gedr3dis']
                break
            except TypeError:
                failed = True
                continue
        if failed:
            CatalogWarning(-1, 9).warn()
            return -999, -999
        cat.sort('_r')
        idx = np.where(cat['Source'] == g_id)[0]
        if len(idx) == 0:
            # Raise exception, for now do nothing
            return -1, -1
        dist = cat[idx]['rgeo'][0]
        lo = dist - cat[idx]['b_rgeo'][0]
        hi = cat[idx]['B_rgeo'][0] - dist
        return dist, max(lo, hi)

    @staticmethod
    def _get_parallax(res):
        plx = res['parallax'][0]
        if plx <= 0:
            CatalogWarning(0, 0).warn()
            return -1, -1
        plx_e = res['parallax_error'][0]
        # Parallax correction −52.8 ± 2.4 µas from Zinn+19
        return plx + 0.0528, np.sqrt(plx_e ** 2 + 0.0024 ** 2)

    @staticmethod
    def _get_radius(res):
        rad = res['radius_val'][0]
        if np.ma.is_masked(rad):
            CatalogWarning('radius', 1).warn()
            return 0, 0
        lo = res['radius_percentile_lower'][0]
        up = res['radius_percentile_upper'][0]
        rad_e = max([rad - lo, up - rad])
        return rad, 5 * rad_e

    @staticmethod
    def _get_teff(res):
        teff = res['teff_val'][0]
        if np.ma.is_masked(teff):
            CatalogWarning('teff', 1).warn()
            return 0, 0
        lo = res['teff_percentile_lower'][0]
        up = res['teff_percentile_upper'][0]
        teff_e = max([teff - lo, up - teff])
        return teff, teff_e

    @staticmethod
    def _get_lum(res):
        lum = res['lum_val'][0]
        if np.ma.is_masked(lum):
            CatalogWarning('lum', 1).warn()
            return 0, 0
        lo = res['lum_percentile_lower'][0]
        up = res['lum_percentile_upper'][0]
        lum_e = max([lum - lo, up - lum])
        return lum, lum_e

    @staticmethod
    def _get_gaia_id(ra, dec, radius):
        c = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
        j = Gaia.cone_search_async(c, radius, table_name='gaiadr3.gaia_source')
        res = j.get_results()
        return res['source_id'][0]

    @staticmethod
    def get_catalogs(ra, dec, radius, catalogs):
        """Retrieve available catalogs for a star from Vizier."""
        tries = [0.5, 0.25, 0.1, 1, 2, 3, 4][::-1]
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
        xm = XMatch.query(cat1='vizier:I/345/gaia2', cat2=galex,
                          colRA2='RAJ2000', colDec2='DEJ2000',
                          area=region, max_distance=radius)
        xm.sort('angDist')
        return xm

    @staticmethod
    def _gaia_mermilliod_xmatch(ra, dec, radius):
        coord = SkyCoord(ra=ra * u.deg,
                         dec=dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=radius)
        xm = XMatch.query(cat1='vizier:I/345/gaia2',
                          cat2='vizier:II/168/ubvmeans',
                          colRA2='_RA', colDec2='_DE',
                          area=region, max_distance=radius)
        xm.sort('angDist')
        return xm

    @staticmethod
    def _gaia_paunzen_xmatch(ra, dec, radius):
        coord = SkyCoord(ra=ra * u.deg,
                         dec=dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=radius)
        xm = XMatch.query(cat1='vizier:I/345/gaia2',
                          cat2='vizier:J/A+A/580/A23/catalog',
                          colRA2='RAICRS', colDec2='DEICRS',
                          area=region, max_distance=radius)
        xm.sort('angDist')
        return xm

    @staticmethod
    def _gaia_hauck_xmatch(ra, dec, radius):
        coord = SkyCoord(ra=ra * u.deg,
                         dec=dec * u.deg, frame='icrs')
        region = CircleSkyRegion(coord, radius=radius)
        xm = XMatch.query(cat1='vizier:I/345/gaia2',
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
