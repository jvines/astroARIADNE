"""Fitter.to_netcdf must write a readable InferenceData file on arviz 0.x and 1.x.

arviz < 1.0 takes one keyword per standard group in ``from_dict``; passing the
whole group dict positionally made it the posterior, and the write failed with
``unable to infer dtype on variable 'posterior'``. arviz >= 1.0 (Python >= 3.12)
takes the group dict directly.
"""
from types import SimpleNamespace

import arviz as az
import numpy as np
import xarray as xr

from astroARIADNE.fitter import Fitter

N_DRAWS = 400


def _stub_fitter():
    rng = np.random.default_rng(42)
    n_bands = 6
    star = SimpleNamespace(
        filter_mask=np.ones(n_bands, dtype=bool),
        wave=rng.uniform(0.4, 12.0, n_bands),
        flux=rng.uniform(1e-12, 1e-10, n_bands),
        flux_er=rng.uniform(1e-14, 1e-12, n_bands),
        filter_names=np.array(['GaiaDR2v2_G', 'GaiaDR2v2_BP', 'GaiaDR2v2_RP',
                               '2MASS_J', '2MASS_H', '2MASS_Ks']),
        bandpass=rng.uniform(0.05, 0.5, n_bands),
    )
    params = ('teff', 'logg', 'z', 'dist', 'rad', 'Av')
    out = {
        'weighted_samples': {p: rng.normal(size=N_DRAWS) for p in params},
        'weights': {'phoenix': 0.7, 'btsettl': 0.3},
        'lnZ': {'phoenix': -100.0, 'btsettl': -101.2},
        'originals': {
            'phoenix': {**{p: rng.normal(size=300) for p in params},
                        'loglike': rng.normal(size=300)},
            'btsettl': {**{p: rng.normal(size=250) for p in params},
                        'loglike': rng.normal(size=250)},
        },
        'mist_samples': {'age': rng.uniform(1, 10, N_DRAWS),
                         'iso_mass': rng.uniform(0.5, 1.5, N_DRAWS),
                         'eep': rng.uniform(300, 500, N_DRAWS)},
        'model_sed': {'model_flux': rng.uniform(1e-12, 1e-10, n_bands)},
        'best_fit_averaged': {'teff': 5700.0, 'rad': 1.0},
        'uncertainties_averaged': {'teff': (40.0, 45.0), 'rad': (0.02, 0.03)},
        'confidence_interval_averaged': {'teff': (5650.0, 5750.0),
                                         'rad': (0.97, 1.03)},
    }
    stub = SimpleNamespace(out=out, star=star)
    stub.to_dict = lambda: Fitter.to_dict(stub)
    return stub


def _read_groups(path):
    if int(az.__version__.split('.')[0]) >= 1:
        tree = xr.open_datatree(path)
        return {g.lstrip('/'): tree[g].to_dataset()
                for g in tree.groups if g != '/'}
    idata = az.from_netcdf(path)
    return {g: idata[g] for g in idata.groups()}


def test_to_netcdf_writes_and_round_trips(tmp_path):
    stub = _stub_fitter()
    path = tmp_path / 'ariadne_result.nc'

    Fitter.to_netcdf(stub, str(path))

    groups = _read_groups(str(path))
    for g in ('posterior', 'observed_data', 'constant_data',
              'posterior_phoenix', 'posterior_btsettl'):
        assert g in groups, f'missing group {g}; have {sorted(groups)}'

    post = groups['posterior']
    assert np.allclose(post['Teff'].values.ravel(),
                       stub.out['weighted_samples']['teff'])
    assert post['age'].values.size == N_DRAWS

    obs = groups['observed_data']
    assert list(obs['filter_names'].values) == list(stub.star.filter_names)
    assert np.allclose(obs['model_flux'].values,
                       stub.out['model_sed']['model_flux'])

    const = groups['constant_data']
    assert list(const['model_names'].values) == ['phoenix', 'btsettl']
    assert np.allclose(const['model_weights'].values, [0.7, 0.3])
    # arviz < 1 stores scalars here with shape (1,); arviz >= 1 with shape ()
    assert np.isclose(const['log_evidence'].values.item(),
                      0.7 * -100.0 + 0.3 * -101.2)
    assert np.allclose(
        const['confidence_interval_averaged__teff'].values, [5650.0, 5750.0])

    phx = groups['posterior_phoenix']
    assert np.allclose(phx['Teff'].values.ravel(),
                       stub.out['originals']['phoenix']['teff'])
    assert phx['log_likelihood'].values.size == 300
