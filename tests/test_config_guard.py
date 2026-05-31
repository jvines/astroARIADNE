"""N3: Fitter rejects mistyped config attributes without breaking its own writes.

Runs as a plain script (no pytest dep):
    python tests/test_config_guard.py
"""
import copy
import os
import pickle
import sys

# Ensure we test the source tree, not any stale installed build.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import astroARIADNE as _aa
assert 'site-packages' not in _aa.__file__, (
    f'testing the installed build, not source: {_aa.__file__}')
from astroARIADNE.fitter import Fitter


def test_construction_works():
    f = Fitter()
    assert f.bma is False
    assert f.grid == 'phoenix'
    assert f._guard_active is True


def test_all_config_setters_still_work():
    f = Fitter()
    f.out_folder = '/tmp/x'
    f.verbose = False
    f.norm = True
    f.grid = 'btsettl'
    f.bma = True
    f.sequential = False
    f.n_samples = 1000
    f.av_law = 'fm07'
    f.estimate_logg = True
    f.estimate_age = False
    f.experimental = True
    f.walks = 30
    f.n_grid_jobs = 2
    f.isochrone_dlogz = 0.05
    f.prior_setup = {'teff': ('default',)}
    f.setup = ['dynesty']
    assert f.bma is True and f.grid == 'btsettl' and f.walks == 30


def test_typo_is_rejected():
    f = Fitter()
    for bad in ('bmaa', 'noooorm', 'gird', 'estimate_aage', 'nsamples'):
        raised = False
        try:
            setattr(f, bad, True)
        except AttributeError:
            raised = True
        assert raised, f'typo {bad!r} should have raised AttributeError'
    # and it did NOT silently create the dead attribute
    assert not hasattr(f, 'bmaa')
    # the real attribute is untouched
    assert f.bma is False


def test_dynamic_postinit_attrs_allowed():
    # the attributes initialize()/fit()/the worker set must NOT be rejected
    f = Fitter()
    f.start = 0.0
    f.coordinator = np.zeros(3)
    f.fixed = np.zeros(3)
    f.default_priors = {}
    f.priors = {}
    f.ndim = 3
    f.prior_sources = {}
    f.sampler = object()
    f.out = {}
    assert f.ndim == 3


def test_private_attrs_always_allowed():
    # the fork BMA worker writes fitter._threads / fitter._grid
    f = Fitter()
    f._threads = 1
    f._grid = 'phoenix'
    f._interpolators = []
    assert f._grid == 'phoenix'


def test_copy_roundtrip():
    f = Fitter()
    f.bma = True
    g = copy.copy(f)
    assert g.bma is True
    # the copy is still guarded
    raised = False
    try:
        g.bmaa = True
    except AttributeError:
        raised = True
    assert raised


def test_pickle_roundtrip():
    f = Fitter()
    f.grid = 'kurucz'
    f.walks = 42
    g = pickle.loads(pickle.dumps(f))
    assert g.grid == 'kurucz' and g.walks == 42
    # still guarded after unpickle
    raised = False
    try:
        g.totally_new_attr = 1
    except AttributeError:
        raised = True
    assert raised


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS {fn.__name__}')
    print(f'ALL {len(fns)} CONFIG-GUARD TESTS PASSED')
