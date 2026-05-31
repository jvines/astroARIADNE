"""N4: fit-health diagnostics (logz+-err, eff samples, grid-edge pegging).

Runs as a plain script (no pytest dep):
    python tests/test_fit_health.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import astroARIADNE as _aa
assert 'site-packages' not in _aa.__file__, (
    f'testing the installed build, not source: {_aa.__file__}')
from astroARIADNE.fitter import Fitter


class _MockResults(dict):
    """Minimal stand-in for a dynesty results object (attr + item access)."""
    def __init__(self, logz, logzerr, logwt):
        super().__init__(logwt=np.asarray(logwt))
        self.logz = np.asarray(logz)
        self.logzerr = np.asarray(logzerr)


def _mock(n=2000, lnz=-1240.0):
    logz = np.linspace(lnz - 5, lnz, n)            # accumulating evidence
    logzerr = np.linspace(1.0, 0.2, n)
    logwt = np.full(n, -np.log(n)) + logz          # ~uniform weights
    return _MockResults(logz, logzerr, logwt)


def test_basic_evidence_and_ess():
    d = Fitter._fit_health(_mock())
    assert abs(d['logz'] - (-1240.0)) < 1e-6
    assert abs(d['logzerr'] - 0.2) < 1e-6
    assert d['eff_samples'] is not None and d['eff_samples'] > 0
    assert d['dlogz_final'] is not None
    assert d['edge_flags'] == []


def test_edge_pegging_upper_flagged():
    grid_axes = {'teff': np.linspace(2300, 12000, 50),
                 'logg': np.linspace(0.0, 5.5, 12),
                 'z': np.linspace(-4.0, 0.5, 10)}
    # teff pegged at the ceiling, logg comfortably interior
    samples = {'teff': np.full(500, 11990.0),
               'logg': np.full(500, 4.5),
               'z': np.full(500, 0.0)}
    d = Fitter._fit_health(_mock(), param_samples=samples, grid_axes=grid_axes)
    flagged = {f['param']: f['edge'] for f in d['edge_flags']}
    assert flagged.get('teff') == 'upper', flagged
    assert 'logg' not in flagged
    assert 'z' not in flagged          # z=0.0 is interior of [-4, 0.5]


def test_edge_pegging_lower_flagged():
    grid_axes = {'teff': np.linspace(2300, 12000, 50)}
    samples = {'teff': np.full(500, 2305.0)}   # within 2% of the floor
    d = Fitter._fit_health(_mock(), param_samples=samples, grid_axes=grid_axes)
    assert any(f['param'] == 'teff' and f['edge'] == 'lower'
               for f in d['edge_flags'])


def test_no_grid_axes_means_no_flags_no_crash():
    d = Fitter._fit_health(_mock(), param_samples={'teff': np.zeros(10)},
                           grid_axes=None)
    assert d['edge_flags'] == []


def test_fixed_param_not_flagged():
    grid_axes = {'teff': np.linspace(2300, 12000, 50)}
    samples = {'teff': 11990.0}   # scalar => fixed, must be ignored
    d = Fitter._fit_health(_mock(), param_samples=samples, grid_axes=grid_axes)
    assert d['edge_flags'] == []


def test_print_does_not_crash():
    d = Fitter._fit_health(
        _mock(), param_samples={'teff': np.full(10, 11990.0)},
        grid_axes={'teff': np.linspace(2300, 12000, 50)})
    Fitter._print_fit_health(d, 'phoenix')   # should print a warning, not raise


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS {fn.__name__}')
    print(f'ALL {len(fns)} FIT-HEALTH TESTS PASSED')
