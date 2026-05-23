"""Regression test: per-field null-safe spectroscopic priors.

The new librarian (default ``feh_source="hypatia"``) can return
``star.spectroscopic_params`` as a *partial* dict where ``feh``/``feh_err``
are set but ``teff``/``teff_err``/``logg``/``logg_err`` are ``None`` (Hypatia
has [Fe/H] but the survey chain found no Teff/logg).

The old librarian never produced such a dict, so ``Fitter._default_priors``
guarded only on ``spec is not None`` and then did
``st.norm(loc=spec['teff'], scale=spec['teff_err'])`` -> ``TypeError`` when
those are ``None``.

This test reproduces that crash mode against an OFFLINE Star (no network) and
asserts the fix:
  (a) ``_default_priors`` does NOT raise;
  (b) the [Fe/H] (``z``) prior is the star-specific normal centred on the
      Hypatia value (0.12);
  (c) the ``teff`` and ``logg`` priors fall back to the population/default
      priors (NOT a None-based ``st.norm``).
"""
import os
import sys

# Ensure the worktree package is imported, not an installed egg that may
# shadow it when this file's directory lands first on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.stats as st

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter
from astroARIADNE.star import Star


# A spectroscopic dict as the Hypatia-only path produces it: [Fe/H] present,
# Teff and logg (and source-derived) None.
PARTIAL_SPEC = {
    'teff': None, 'teff_err': None,
    'logg': None, 'logg_err': None,
    'feh': 0.12, 'feh_err': 0.10,
    'source': 'Hypatia',
}


def _build_offline_star():
    """Construct an offline Star (no network) with enough photometry for
    ``_default_priors`` to run, then attach the partial spec dict.

    ``dist`` and ``Av`` are supplied so the constructor never touches the
    network or dustmaps.
    """
    mag_dict = {
        '2MASS_J': (10.0, 0.02),
        '2MASS_H': (9.7, 0.02),
        '2MASS_Ks': (9.6, 0.02),
        'GaiaDR2v2_G': (10.5, 0.01),
    }
    star = Star(
        'partial_spec_test', ra=180.0, dec=0.0,
        dist=50.0, dist_e=1.0,
        Av=0.05,
        offline=True, mag_dict=mag_dict, verbose=False,
    )
    # The offline path leaves spectroscopic_params None; emulate the new
    # librarian's Hypatia-only result.
    star.spectroscopic_params = dict(PARTIAL_SPEC)
    star.rave_params = None
    # When Av is supplied the constructor sets Av (not Av_e); pin Av_e so the
    # extinction branch (orthogonal to this test) is deterministic.
    if not hasattr(star, 'Av_e'):
        star.Av_e = None
    return star


def _build_fitter(star):
    """Minimal Fitter wired up to call ``_default_priors`` in isolation.

    Bypasses ``Fitter.__init__`` (which loads heavy model grids) and sets only
    the attributes ``_default_priors`` reads, mirroring what ``initialize``
    sets up before it calls ``_default_priors``.
    """
    f = Fitter.__new__(Fitter)
    f._star = star
    f.norm = False          # property setter -> also sets f._norm
    f.prior_sources = {}

    npars = 6 + int(star.used_filters.sum())
    f.coordinator = np.zeros(npars)
    f.fixed = np.zeros(npars)

    # ``_default_priors`` appends noise params to the module-level ``order``
    # (normally initialised inside ``Fitter.initialize``).
    fitter_mod.order = np.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av'])
    return f


def test_partial_spec_does_not_raise_and_falls_back_per_field():
    star = _build_offline_star()
    f = _build_fitter(star)

    # (a) must not raise (the old code crashed with TypeError here).
    defaults = f._default_priors()

    # (b) [Fe/H] prior is the star-specific normal centred on the Hypatia
    # value, with the Hypatia error as scale.
    z_prior = defaults['z']
    assert isinstance(z_prior, st._distn_infrastructure.rv_frozen)
    assert z_prior.dist.name == 'norm'
    loc, scale = z_prior.args if z_prior.args else (
        z_prior.kwds['loc'], z_prior.kwds['scale'])
    assert np.isclose(loc, 0.12)
    assert np.isclose(scale, 0.10)
    assert f.prior_sources['z'] == 'Hypatia'

    # (c) Teff falls back to the population prior (the pickled ppf spline, NOT
    # a None-based norm). The population prior is a callable spline (ARIADNE
    # samples it via ``prior(u)``); a None-based ``st.norm`` would instead be a
    # frozen rv and would have raised above. It must produce a finite value.
    teff_prior = defaults['teff']
    assert f.prior_sources['teff'] == 'rave_population'
    assert not isinstance(teff_prior, st._distn_infrastructure.rv_frozen)
    assert callable(teff_prior)
    assert np.isfinite(float(teff_prior(0.5)))

    # (c) logg falls back to the population/default uniform (get_logg is False
    # for an offline star), NOT a None-based norm.
    logg_prior = defaults['logg']
    assert f.prior_sources['logg'] in ('uniform_default', 'isochrone')
    assert logg_prior.dist.name in ('uniform', 'norm')
    logg_sample = logg_prior.ppf(0.5)
    assert np.isfinite(logg_sample)


if __name__ == '__main__':
    test_partial_spec_does_not_raise_and_falls_back_per_field()
    print('PASS: test_partial_spec_does_not_raise_and_falls_back_per_field')
