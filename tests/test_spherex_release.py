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
