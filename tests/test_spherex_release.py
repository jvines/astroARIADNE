"""SPHEREx release selection: DR1 first, QR3 fallback, no-planes never cached.

Offline: astroquery's Irsa.query_sia is replaced by a stub keyed on collection.
"""
import os

import numpy as np
import pytest

from astroARIADNE.librarian import _spherex


class _Res(dict):
    pass


def _stub(monkeypatch, answers, calls):
    from astroquery.ipac.irsa import Irsa

    def query_sia(pos=None, collection=None):
        coord, _ = pos
        calls.append((collection, float(coord.ra.deg), float(coord.dec.deg)))
        a = answers[collection]
        if isinstance(a, Exception):
            raise a
        return _Res(access_url=a)

    monkeypatch.setattr(Irsa, 'query_sia', staticmethod(query_sia))


def test_dr1_is_primary(monkeypatch):
    calls = []
    _stub(monkeypatch, {'spherex_dr1': ['u1', 'u2'], 'spherex_qr3': ['q']}, calls)
    urls, coll, ra, dec = _spherex._query_planes(10.0, -20.0)
    assert (urls, coll, ra, dec) == (['u1', 'u2'], 'spherex_dr1', 10.0, -20.0)
    assert [c for c, *_ in calls] == ['spherex_dr1']


def test_qr3_fallback_when_dr1_empty_repropagates(monkeypatch):
    calls = []
    _stub(monkeypatch, {'spherex_dr1': [], 'spherex_qr3': ['q1']}, calls)
    seen = []

    def position(ep):
        seen.append(ep)
        return 10.5, -20.5

    urls, coll, ra, dec = _spherex._query_planes(10.0, -20.0, position=position)
    assert (urls, coll, ra, dec) == (['q1'], 'spherex_qr3', 10.5, -20.5)
    assert seen == [dict(_spherex.RELEASES)['spherex_qr3']]
    assert calls[1] == ('spherex_qr3', 10.5, -20.5)


def test_qr3_fallback_when_dr1_fails(monkeypatch):
    calls = []
    _stub(monkeypatch, {'spherex_dr1': RuntimeError('down'), 'spherex_qr3': ['q1']}, calls)
    urls, coll, _, _ = _spherex._query_planes(10.0, -20.0)
    assert (urls, coll) == (['q1'], 'spherex_qr3')


def test_all_failed_raises_first_error(monkeypatch):
    _stub(monkeypatch, {'spherex_dr1': RuntimeError('dr1 down'),
                        'spherex_qr3': RuntimeError('qr3 down')}, [])
    with pytest.raises(RuntimeError, match='dr1 down'):
        _spherex._query_planes(10.0, -20.0)


def test_no_planes_anywhere_is_not_cached(monkeypatch, tmp_path):
    _stub(monkeypatch, {'spherex_dr1': [], 'spherex_qr3': []}, [])
    monkeypatch.setattr(_spherex, 'CACHE_DIR', str(tmp_path))
    meta = {}
    lam, f, e = _spherex.extract_spectrum(10.0, -20.0, 42, meta=meta)
    assert len(lam) == len(f) == len(e) == 0
    assert meta['release'] is None
    assert not os.path.exists(os.path.join(str(tmp_path), '42.csv'))


def test_epoch_is_dr1_mid_epoch():
    lo, hi = 2025.31, 2026.35          # DR1 observation span
    ep = dict(_spherex.RELEASES)['spherex_dr1']
    assert _spherex.EPOCH == ep
    assert max(ep - lo, hi - ep) < 0.53
    assert np.isclose(ep, 0.5 * (lo + hi), atol=0.01)


def test_neighbours_at_propagates_by_proper_motion():
    n = dict(ra=10.0, dec=60.0, sep=30.0, ks=9.0, g=np.nan,
             pmra=1000.0, pmde=-500.0, epoch=2025.83)
    m, = _spherex._neighbours_at([n], 2026.83)       # one year later
    assert np.isclose((m['ra'] - 10.0) * 3.6e6 * np.cos(np.radians(60.0)), 1000.0)
    assert np.isclose((m['dec'] - 60.0) * 3.6e6, -500.0)
    assert m['epoch'] == 2026.83 and n['epoch'] == 2025.83     # input untouched


def test_qr3_fallback_moves_the_fitted_neighbours_too(monkeypatch, tmp_path):
    _stub(monkeypatch, {'spherex_dr1': [], 'spherex_qr3': ['q1']}, [])
    monkeypatch.setattr(_spherex, 'CACHE_DIR', str(tmp_path / 'ap'))
    monkeypatch.setattr(_spherex, 'PSF_CACHE_DIR', str(tmp_path / 'psf'))
    monkeypatch.setattr(_spherex, 'USE_RANGED_READS', False)
    dr1 = dict(_spherex.RELEASES)['spherex_dr1']
    qr3 = dict(_spherex.RELEASES)['spherex_qr3']
    nb = dict(ra=10.001, dec=-20.0, sep=4.0, ks=9.0, g=np.nan,
              pmra=2000.0, pmde=0.0, epoch=dr1)       # bright, inside the aperture
    monkeypatch.setattr(_spherex, 'neighbours', lambda *a, **k: ([nb], 0))
    seen = []
    monkeypatch.setattr(_spherex, '_fetch_one', lambda args: seen.append(args[6]))
    meta = {}
    _spherex.extract_spectrum(10.0, -20.0, 7, ks=9.0, epoch=dr1, meta=meta,
                              position=lambda ep: (10.0, -20.0))
    assert meta['method'] == 'psf' and meta['release'] == 'spherex_qr3'
    (moved,), = seen
    assert moved['epoch'] == qr3
    assert np.isclose((moved['ra'] - nb['ra']) * 3.6e6 * np.cos(np.radians(-20.0)),
                      2000.0 * (qr3 - dr1))
