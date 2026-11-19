"""fit_bma reuses compatible <grid>_out.pkl files unless refit is set (#56).

Offline: dynesty, the averaging and the output step are stubbed out.
"""
import os
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter

FILTS = ['2MASS_J', '2MASS_H', '2MASS_Ks', 'GaiaDR2v2_G', 'WISE_RSR_W1',
         'WISE_RSR_W2', 'SDSS_g']


def _star(filts=FILTS):
    return SimpleNamespace(filter_names=np.array(filts),
                           filter_mask=np.ones(len(filts), dtype=bool))


def _fitter(tmp_path, grids, refit=False, norm=False):
    f = Fitter.__new__(Fitter)
    f.out_folder = str(tmp_path)
    f.star = _star()
    f._grids = list(grids)
    f._interpolators = [None] * len(grids)
    f._norm = norm
    f._refit = refit
    f.n_grid_jobs = 1
    f.n_samples = 10
    f.colors = ['white']
    f.coordinator = None
    f.start = 0
    f._binary = False
    f.binary = False
    return f


def _fake_out(tmp_path, grid, norm=False, star=None):
    with open(tmp_path / f'{grid}_out.pkl', 'wb') as fh:
        pickle.dump({'global_lnZ': 1.0, 'posterior_samples': {},
                     'star': star or _star(), 'norm': norm,
                     'model_grid': grid, 'binary': False}, fh)


def _run(monkeypatch, f):
    fitted, averaged = [], []
    monkeypatch.setattr(Fitter, 'fit_dynesty',
                        lambda self, out_file=None: fitted.append(self.grid))
    monkeypatch.setattr(
        Fitter, 'bayesian_model_average',
        staticmethod(lambda outs, grids, *a: averaged.append(list(grids))))
    monkeypatch.setattr(Fitter, 'save_bma', lambda self, avgd: None)
    monkeypatch.setattr(fitter_mod, 'execution_time', lambda s: 0)
    monkeypatch.setattr(fitter_mod, 'end', lambda *a, **k: None)
    f.fit_bma()
    return fitted, averaged


def test_reuses_existing_fits_and_fits_missing(tmp_path, monkeypatch):
    _fake_out(tmp_path, 'phoenix')
    fitted, averaged = _run(monkeypatch, _fitter(tmp_path, ['phoenix', 'btsettl']))
    assert fitted == ['btsettl']
    assert averaged == [['phoenix', 'btsettl']]


def test_refit_fits_everything(tmp_path, monkeypatch):
    _fake_out(tmp_path, 'phoenix')
    f = _fitter(tmp_path, ['phoenix', 'btsettl'])
    f.refit = True
    fitted, _ = _run(monkeypatch, f)
    assert fitted == ['phoenix', 'btsettl']


def test_incompatible_fit_is_refitted(tmp_path, monkeypatch, capsys):
    _fake_out(tmp_path, 'phoenix', star=_star(FILTS[:-1]))   # filters differ
    _fake_out(tmp_path, 'btsettl', norm=True)                # norm differs
    _fake_out(tmp_path, 'kurucz')                            # compatible
    f = _fitter(tmp_path, ['phoenix', 'btsettl', 'kurucz'])
    fitted, _ = _run(monkeypatch, f)
    assert fitted == ['phoenix', 'btsettl']
    out = capsys.readouterr().out
    assert 'filters changed' in out and 'norm vs dist/rad' in out
