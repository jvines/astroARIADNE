"""N5: extinction-law selector should accept fm07 (fixed Rv=3.1).

Runs as a plain script (the project has no pytest dependency):
    python tests/test_extinction_laws.py
"""
import os
import sys

# Ensure we test the source tree, not any stale installed build.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import astroARIADNE as _aa
assert 'site-packages' not in _aa.__file__, (
    f'testing the installed build, not source: {_aa.__file__}')
from astroARIADNE.fitter import Fitter
from astroARIADNE.sed_library import _ext_attenuation_base


def _set_law(law):
    f = Fitter.__new__(Fitter)   # bypass __init__/network; exercise the setter only
    f.av_law = law
    return f._av_law


def test_existing_laws_still_work():
    for law in ('cardelli', 'odonnell', 'calzetti', 'fitzpatrick'):
        fn = _set_law(law)
        out = fn(np.array([3000., 5000., 10000.]), 1.0, 3.1)
        assert np.all(np.isfinite(out)), law


def test_fm07_selectable_and_fixed_rv():
    fn = _set_law('fm07')
    out = fn(np.array([3000., 5000., 10000.]), 1.0, 3.1)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_fm07_differs_from_fitzpatrick():
    w = np.array([2500., 4000., 8000., 20000.])
    fm = _set_law('fm07')(w, 1.0, 3.1)
    fitz = _set_law('fitzpatrick')(w, 1.0, 3.1)
    assert not np.allclose(fm, fitz)


def test_fm07_cache_key_stable():
    fn1 = _set_law('fm07')
    fn2 = _set_law('fm07')
    assert fn1 is fn2, 'fm07 wrapper must be a single stable object for caching'
    wave = np.linspace(0.3, 2.0, 50)
    b1 = _ext_attenuation_base(wave, fn1)
    b2 = _ext_attenuation_base(wave, fn2)
    assert b1 is b2


def test_unknown_law_still_rejected():
    raised = False
    try:
        _set_law('nonsense_law')
    except Exception:
        raised = True
    assert raised, 'unknown law must raise'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    n = 0
    for fn in fns:
        fn()
        n += 1
        print(f'  PASS {fn.__name__}')
    print(f'ALL {n} EXTINCTION TESTS PASSED')
