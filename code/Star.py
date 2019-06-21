"""Star.py contains the Star class which contains the data regarding a star."""


class Star:

    # pyphot filter names: currently unused are U R I PS1_w

    #     '2MASS_H', '2MASS_J', '2MASS_Ks',
    #     'GROUND_JOHNSON_U', 'GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
    #     'GROUND_COUSINS_R', 'GROUND_COUSINS_I',
    #     'GaiaDR2v2_G', 'GaiaDR2v2_RP', 'GaiaDR2v2_BP',
    #     'PS1_g', 'PS1_i', 'PS1_r', 'PS1_w', 'PS1_y',  'PS1_z',
    #     'SDSS_g', 'SDSS_i', 'SDSS_r', 'SDSS_u', 'SDSS_z',
    #     'WISE_RSR_W1', 'WISE_RSR_W2'

    # Catalogs magnitude names
    apass_mags = ['Vmag', 'Bmag', 'g_mag', 'r_mag', 'i_mag']
    apass_errs = ['e_Vmag', 'e_Bmag', 'e_g_mag', 'e_r_mag', 'e_i_mag']
    apass_filters = ['GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
                     'SDSS_g', 'SDSS_r', 'SDSS_i']
    wise_mags = ['W1mag', 'W2mag']
    wise_errs = ['e_W1mag', 'e_W2mag']
    wise_filters = ['WISE_RSR_W1', 'WISE_RSR_W2']
    ps1_mags = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    ps1_errs = ['e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_ymag']
    ps1_filters = ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']
    twomass_mags = ['Jmag', 'Hmag', 'Kmag']
    twomass_errs = ['e_Jmag', 'e_Hmag', 'e_Kmag']
    twomass_filters = ['2MASS_J', '2MASS_H', '2MASS_Ks']
    gaia_mags = ['Gmag', 'BPmag', 'RPmag']
    gaia_errs = ['e_Gmag', 'e_BPmag', 'e_RPmag']
    gaia_filters = ['GaiaDR2v2_G',  'GaiaDR2v2_BP', 'GaiaDR2v2_RP']
    sdss_mags = ['umag', 'zmag']
    sdss_errs = ['e_umag', 'e_zmag']
    sdss_filters = ['SDSS_u', 'SDSS_z']

    # APASS DR9, WISE, PAN-STARRS DR1, GAIA DR2, 2MASS, SDSS DR9
    catalogs = {
        'apass': ['II/336/apass9', zip(apass_mags, apass_errs, apass_filters)],
        'Wise': ['II/311/wise', zip(wise_mags, wise_errs, wise_filters)],
        'Pan-STARRS': ['II/349/ps1', zip(ps1_mags, ps1_errs, ps1_filters)],
        'Gaia': ['I/345/gaia2', zip(gaia_mags, gaia_errs, gaia_filters)],
        '2MASS': [
            'II/246/out',
            zip(twomass_mags, twomass_errs, twomass_filters)
        ],
        'SDSS': ['V/139/sdss9', zip(sdss_mags, sdss_errs, sdss_filters)]
    }

    def __init__(self, starname, coords=None, ra=None, dec=None,
                 fixed_Z=False):
        """ra/dec units must be in deg."""
        self.full_grid = sp.loadtxt('test_grid.dat')
        self.teff = self.full_grid[:, 0]
        self.logg = self.full_grid[:, 1]
        self.z = self.full_grid[:, 2] if not fixed_Z else fixed_Z
        self.starname = starname
        self.coords = coords
        if coords:
            self.ra = ra
            self.ra_units = ra_units
            self.dec = dec
            self.dec_units = dec_units
        self.get_magnitudes()

    def get_magnitudes(self):
        """Retrieve the magnitudes of the star.

        Looks into APASS, WISE, Pan-STARRS, Gaia, 2MASS and SDSS surveys
        looking for different magnitudes for the star, along with the
        associated uncertainties.
        """
        if self.coords:
            cats = Vizier.query_region(
                coord.SkyCoord(
                    ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'
                ), radius=Angle(.01, "deg")
            )
        else:
            cats = Vizier.query_object(self.starname)

        filters = []
        magnitudes = []
        errors = []

        for c in self.catalogs.keys():
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
        if self.coords:
            cats = Vizier.query_region(
                coord.SkyCoord(
                    ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'
                ), radius=Angle(.01, "deg")
            )
        else:
            cats = Vizier.query_object(self.starname)
        try:
            plx = cats[self.plx_catalogs['Gaia']
                       [0]][self.plx_catalogs['Gaia'][1]]
            plx_e = cats[self.plx_catalogs['Gaia']
                         [0]][self.plx_catalogs['Gaia'][2]]
        except Exception as e:
            print('No Gaia parallax found for this star.', end=' ')
            print('Retrying with Hipparcos.')
            try:
                plx = cats[self.plx_catalogs['Hipparcos']
                           [0]][self.plx_catalogs['Hipparcos'][1]]
                plx_e = cats[self.plx_catalogs['Hipparcos']
                             [0]][self.plx_catalogs['Hipparcos'][2]]
            except Exception as e:
                print('No Hipparcos parallax found.', end=' ')
                print('Input the parallax manually.')

        self.plx = plx
        self.plx_e = plx_e
