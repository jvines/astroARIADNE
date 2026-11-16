"""SEDPlotter._load_from_cache must not draw a spectrum at the wrong Teff.

Regression: the btsettl cache group starts at 2600 K, and a 1700 K fit was
drawn from the 2600 K node (bracketing clamped to the grid edge), so the best-fit
curve sat 2-5x above the model's own synthetic photometry.
"""
import types

import h5py
import numpy as np
import pytest

import astroARIADNE.plotter as P


def _group(h, name, teffs):
    g = h.create_group(name)
    wave = np.linspace(0.5, 5.0, 10)
    rows = [(t, 5.0, 0.0) for t in teffs]
    g['wavelength'] = wave
    g['teff'] = np.array([r[0] for r in rows], float)
    g['logg'] = np.array([r[1] for r in rows], float)
    g['z'] = np.array([r[2] for r in rows], float)
    # flux encodes the node temperature so the test can see which node was used
    g['flux'] = np.array([np.full(wave.size, r[0]) for r in rows], float)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / 'spectra_cache.h5'
    with h5py.File(path, 'w') as h:
        _group(h, 'btsettl', [2600, 2700, 2800])
        _group(h, 'btsettl_cool', [1600, 1700, 1800, 2600])
        _group(h, 'phoenix', [2300, 2400])
    monkeypatch.setattr(P, 'spectra_cache', str(path))
    return path


def _load(model, teff):
    fake = types.SimpleNamespace(theta=np.array([teff, 5.0, 0.0]))
    return P.SEDPlotter._load_from_cache(fake, model)


def test_in_range_interpolates(cache):
    w, f = _load('btsettl', 2650)
    assert np.allclose(f, 2650)


def test_below_range_uses_cool_group(cache):
    w, f = _load('btsettl', 1700)
    assert np.allclose(f, 1700)


def test_below_range_without_cool_group_declines(cache):
    assert _load('phoenix', 1700) is None
