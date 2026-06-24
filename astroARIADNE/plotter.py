"""plot_utils module for plotting SEDs."""

__all__ = ['SEDPlotter']

import glob
import logging
import os
import pickle
from random import choice

logger = logging.getLogger(__name__)

# Cool/BD plotting spectra, shipped inside the standard spectra cache as the
# groups btsettl_cool / atmo / sonora (added by
# notebooks/build_cool_spectra_cache.py, distributed with the cache via
# Zenodo). Loaded lazily per grid; None marks a grid (or the cache) as
# unavailable so the companion overlay falls back to band-resolution curves.
_COOL_SPEC_CACHE: dict = {}
_COOL_GROUPS = {'btsettl': 'btsettl_cool', 'atmo': 'atmo', 'sonora': 'sonora',
                'btcond': 'btcond', 'btnextgen': 'btnextgen',
                'btdusty': 'btdusty'}


def _load_cool_cache(gname):
    if gname in _COOL_SPEC_CACHE:
        return _COOL_SPEC_CACHE[gname]
    out = None
    group = _COOL_GROUPS.get(gname)
    try:
        import h5py
        if group and spectra_cache and os.path.isfile(spectra_cache):
            with h5py.File(spectra_cache, 'r') as h:
                if group in h:
                    g = h[group]
                    out = (g['wavelength'][:].astype(float), g['teff'][:],
                           g['logg'][:], g['z'][:], g['flux'][:])
    except Exception:
        out = None
    _COOL_SPEC_CACHE[gname] = out
    return out

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from extinction import apply
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from PyAstronomy import pyasl
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm

import corner
from dynesty import plotting as dyplot

from .config import (filesdir, gridsdir, modelsdir, spectra_cache)
from .isochrone import get_isochrone
from .phot_utils import *
from .sed_library import *
from .utils import *


class SEDPlotter:
    """Artist class for all things SED.

    Parameters
    ----------
    input_files : str
        Directory containing the code's output files.
    out_folder : type
        Directory where to put the output plots.
    pdf : type
        Set to True to output plots in pdf.
    png : type
        Set to True to output plots in png.
    model : type
        Set to override the SED model that's going to be plotted.
        Possible values are:
            - Phoenix
            - BTSettl
            - NextGen
            - CK04 (Castelli & Kurucz 2004)
            - Kurucz (Kurucz 1993)

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    Attributes
    ----------
    chain_out : str
        Output directory for chain plot.
    like_out : str
        Output directory for likelihood plot.
    post_out : str
        Output directory for posteriors plot.
    moddir : type
        Directory wheere the SED models are located.
    out : dict
        SED fitting routine output.
    engine : str
        Selected fitting engine.
    star : Star
        The fitted Star object.
    coordinator : array_like
        Array coordinating fixed parameters.
    fixed : array_like
        Array coordinating fixed parameters.
    norm : bool
        norm is set to True if a normalization constant is fitted instead of
        radius + distance.
    grid : str
        Selected model grid.
    av_law : function
        Exticntion law chosen for the fit.
    order : array_like
        Array coordinating parameter order.
    interpolator : function
        Interpolator function.
    theta : array_like
        `Best fit` parameter vector

    """

    __wav_file = 'PHOENIXv2/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

    def __init__(self, input_files, out_folder, pdf=False,
                 model=None, settings=None, method='averaged',
                 save_model=False, ir_excess=False):
        """See class docstring."""
        print('\nInitializing plotter.\n')
        # General setup
        self.pdf = pdf
        png = True if not pdf else False
        self.png = png
        self.out_folder = out_folder
        self.bma = False
        self.method = method
        self.save = save_model
        self.irx = ir_excess

        traces = f'{out_folder}/traces'
        histograms = f'{out_folder}/histograms'
        self.traces_out = traces
        self.hist_out = histograms
        self.moddir = modelsdir
        self.settings_dir = settings

        # Read output files.
        if input_files != 'raw':
            out = pickle.load(open(input_files, 'rb'))
            self.out = out
            self.engine = out['engine']
            self.star = out['star']
            self.coordinator = out['coordinator']
            self.fixed = out['fixed']
            self.norm = out['norm']
            self.binary = out.get('binary', False)
            if model is None:
                if self.engine != 'Bayesian Model Averaging':
                    self.grid = out['model_grid']
                else:
                    self.bma = True
                    zs = np.array([out['lnZ'][key]
                                   for key in out['lnZ'].keys()])
                    keys = np.array([key for key in out['lnZ'].keys()])
                    grid = keys[np.argmax(zs)]
                    self.grid = grid
            else:
                self.grid = model
            self.av_law = out['av_law']

            # Create target folders
            create_dir(out_folder)
            if self.engine != 'Bayesian Model Averaging':
                create_dir(traces)
            create_dir(histograms)

            self.star.load_grid(self.grid)

            if not self.norm:
                self.order = np.array(
                    [
                        'teff', 'logg', 'z',
                        'dist', 'rad', 'Av',
                    ]
                )
                if self.binary:
                    self.order = np.append(self.order,
                                           ['teff2', 'logg2', 'rad2'])
            else:
                self.order = np.array(
                    ['teff', 'logg', 'z', 'norm', 'Av'])

            mask = self.star.filter_mask
            flxs = self.star.flux[mask]
            errs = self.star.flux_er[mask]
            filters = self.star.filter_names[mask]
            wave = self.star.wave[mask]
            if self.irx:
                irx_mask = self.star.irx_filter_mask
                irx_filt = self.star.filter_names[irx_mask]
                irx_wave = self.star.wave[irx_mask]
                # The IR-excess characterization (disk/companion fit) from the
                # fit output, drawn as an overlay by IRX_SED (never plot_SED).
                _irx = out.get('irx') or {}
                self.irx_fit = _irx.get('_summary')
                # Per-band excess significance as reported by _compute_irx --
                # (obs-model)/hypot(obs_err, model_err), i.e. including the
                # photosphere model uncertainty. Used for the residual panel so
                # it matches the reported detection significance (the bare
                # (obs-model)/obs_err overstates deviations for the IR bands).
                self.irx_sigma = np.array([
                    float((_irx.get(f, {}) or {}).get('sigma', np.nan))
                    for f in irx_filt])
                # Per-band error used for the residual panel: floored obs error
                # in quadrature with the photosphere model uncertainty (the same
                # denominator as the stored significance). The residual panel
                # shows (obs - DISPLAYED total model)/err so it agrees with the
                # top panel rather than the bare-photosphere excess sigma.
                self.irx_band_err = np.array([
                    float(np.hypot(
                        (_irx.get(f, {}) or {}).get(
                            'obs_err_eff',
                            (_irx.get(f, {}) or {}).get('obs_err', np.nan)),
                        (_irx.get(f, {}) or {}).get('model_err', 0.0)))
                    for f in irx_filt])
            for filt, flx, flx_e in zip(filters, flxs, errs):
                p_ = get_noise_name(filt) + '_noise'
                self.order = np.append(self.order, p_)

            self.interpolator = load_grid(self.grid)

            # Get best fit parameters. Single-grid outputs carry only
            # 'best_fit_averaged'; fall back to it so single-grid SEDs plot too.
            bf_samples = out.get('best_fit_samples', out['best_fit_averaged'])
            theta_samples = np.zeros(self.order.shape[0])
            theta_average = np.zeros(self.order.shape[0])
            for i, param in enumerate(self.order):
                if param != 'inflation':
                    theta_samples[i] = bf_samples[param]
                    theta_average[i] = out['best_fit_averaged'][param]

            # The plot theta (posterior medians) can land in a NaN pocket of
            # the grid even though the fit was fine: the cool-splice nodes
            # (Teff <= 2500 K) exist only at [Fe/H]=0, so a median like
            # z=-0.02 NaNs every synthetic flux -- no model diamonds, no
            # residuals (-inf samples were simply rejected during the fit).
            # Snap the grid axes to the nearest finite node when that
            # happens; the cache-backed spectrum curve already snaps the
            # same way.
            theta_samples = self._snap_theta_finite(theta_samples, filters,
                                                    wave)
            theta_average = self._snap_theta_finite(theta_average, filters,
                                                    wave)

            # Calculate best fit model.
            model_samples = model_grid(theta_samples, filters, wave,
                                       self.interpolator, self.norm,
                                       self.av_law, binary=self.binary)
            model_average = model_grid(theta_average, filters, wave,
                                       self.interpolator, self.norm,
                                       self.av_law, binary=self.binary)
            if self.irx:
                # Per-band photosphere model at the IR-excess bands. PREFER the
                # fitter's stored posterior-predictive model (_compute_irx, in
                # the pickle's irx dict): it is the evidence-weighted value over
                # the FULL BMA ensemble and is finite for every quantified band.
                # Recomputing from the single forced plotting grid (btsettl, set
                # for every IRX plot) leaves NaN holes on bands that grid does
                # not reach or has NaN nodes for -- which silently dropped the
                # model diamond on those bands. Fall back to the recomputed grid
                # model only where the fitter stored none (e.g. plot-only bands
                # outside the ensemble's common reach), so model_grid never
                # KeyErrors on a missing band and provisional bands still draw.
                irx_stored = np.array([
                    float((_irx.get(f, {}) or {}).get('model', np.nan))
                    for f in irx_filt])
                reach = grid_irx_reach(self.grid, self.interpolator)
                keep = np.array([f in reach for f in irx_filt], dtype=bool)
                irx_recomp_sam = np.full(len(irx_filt), np.nan)
                irx_recomp_avg = np.full(len(irx_filt), np.nan)
                if keep.any():
                    irx_recomp_sam[keep] = model_grid(
                        theta_samples, irx_filt[keep], irx_wave[keep],
                        self.interpolator, self.norm, self.av_law,
                        binary=self.binary)
                    irx_recomp_avg[keep] = model_grid(
                        theta_average, irx_filt[keep], irx_wave[keep],
                        self.interpolator, self.norm, self.av_law,
                        binary=self.binary)
                _fin = np.isfinite(irx_stored)
                irx_model_sam = np.where(_fin, irx_stored, irx_recomp_sam)
                irx_model_avg = np.where(_fin, irx_stored, irx_recomp_avg)
            if method == 'averaged':
                self.theta = theta_average
                self.model = model_average
                if self.irx:
                    self.irx_model = irx_model_avg
            elif method == 'samples':
                self.theta = theta_samples
                self.model = model_samples
                if self.irx:
                    self.irx_model = irx_model_sam

            # Get archival fluxes.
            self.__extract_info()
        else:
            self.star = None

        # Setup plots.
        self.__read_config()
        print('\nPlotter initialized.\n')

    def __extract_info(self):
        self.flux = []
        self.flux_er = []
        self.wave = []
        self.bandpass = []

        for i, f in zip(self.star.used_filters, self.star.flux):
            if i:
                self.flux.append(f)
        for i, e in zip(self.star.used_filters, self.star.flux_er):
            if i:
                self.flux_er.append(e)
        for i, w in zip(self.star.used_filters, self.star.wave):
            if i:
                self.wave.append(w)
        for i, bp in zip(self.star.used_filters, self.star.bandpass):
            if i:
                self.bandpass.append(bp)

        self.flux = np.array(self.flux)
        self.flux_er = np.array(self.flux_er)
        self.wave = np.array(self.wave)
        self.bandpass = np.array(self.bandpass).T
        # Do the same for IR bandpasses
        self.irx_flux = []
        self.irx_flux_er = []
        self.irx_wave = []
        self.irx_bandpass = []

        for i, f in zip(self.star.irx_used_filters, self.star.flux):
            if i:
                self.irx_flux.append(f)
        for i, e in zip(self.star.irx_used_filters, self.star.flux_er):
            if i:
                self.irx_flux_er.append(e)
        for i, w in zip(self.star.irx_used_filters, self.star.wave):
            if i:
                self.irx_wave.append(w)
        for i, bp in zip(self.star.irx_used_filters, self.star.bandpass):
            if i:
                self.irx_bandpass.append(bp)

        self.irx_flux = np.array(self.irx_flux)
        self.irx_flux_er = np.array(self.irx_flux_er)
        self.irx_wave = np.array(self.irx_wave)
        self.irx_bandpass = np.array(self.irx_bandpass).T

    def _snap_theta_finite(self, theta, filters, wave):
        """Return ``theta`` with (z, teff, logg) snapped to the nearest grid
        node, axis by axis, until the synthetic photometry is finite in every
        fitted band. Returns the original theta untouched when it is already
        fine (the normal case) or when no snap helps."""
        def _all_finite(th):
            m = model_grid(th, filters, wave, self.interpolator, self.norm,
                           self.av_law, binary=self.binary)
            return bool(np.isfinite(np.asarray(m, dtype=float)).all())
        try:
            if _all_finite(theta):
                return theta
            lg, tf, fe = self.interpolator.index_columns
            th = np.array(theta, dtype=float)
            # z first (the splice's restricted axis), then teff, then logg.
            for idx, nodes in ((2, np.unique(fe)), (0, np.unique(tf)),
                               (1, np.unique(lg))):
                th[idx] = float(nodes[np.argmin(np.abs(nodes - th[idx]))])
                if _all_finite(th):
                    logger.warning(
                        'plot theta sat on a NaN grid pocket; snapped '
                        '(teff, logg, z) to (%.1f, %.2f, %.2f) for the '
                        'synthetic photometry.', th[0], th[1], th[2])
                    return th
        except Exception as e:
            logger.warning('theta finite-snap failed (%s); plotting with the '
                           'raw theta.', e)
        return theta

    def _load_from_cache(self, model_key):
        """Return (wavelength, flux) from pre-computed HDF5 cache, or None."""
        if not spectra_cache:
            return None
        import h5py
        import os as _os
        if not _os.path.isfile(spectra_cache):
            return None
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        try:
            with h5py.File(spectra_cache, 'r') as f:
                if model_key not in f:
                    return None
                g = f[model_key]
                teffs = g['teff'][:]
                loggs = g['logg'][:]
                zs = g['z'][:]
                t_val = np.unique(teffs)[np.argmin(np.abs(np.unique(teffs) - teff))]
                l_val = np.unique(loggs)[np.argmin(np.abs(np.unique(loggs) - logg))]
                z_val = np.unique(zs)[np.argmin(np.abs(np.unique(zs) - z))]
                mask = (teffs == t_val) & (loggs == l_val) & (zs == z_val)
                idx = np.where(mask)[0][0]
                wave = g['wavelength'][:]
                flux = g['flux'][idx]
            return wave.astype(float), flux.astype(float)
        except Exception as e:
            logger.warning(f'Failed to load spectra cache for {model_key}: {e}')
            return None

    @staticmethod
    def _extend_rj(wave, flux, lam_max=30.0):
        """Extend a photosphere spectrum to lam_max with a Rayleigh-Jeans tail
        (F_lambda ~ lambda^-4). The plotting spectra cache stops at ~4.63 um, but
        an IRX SED shows excess bands out to ~22 um (W4); a stellar photosphere
        is firmly in the RJ regime there, so the tail is an excellent stand-in
        for the unavailable full-range model spectrum. No-op if already long."""
        wave = np.asarray(wave, float); flux = np.asarray(flux, float)
        if wave[-1] >= lam_max:
            return wave, flux
        ext_w = np.geomspace(wave[-1] * 1.001, lam_max, 80)
        ext_f = flux[-1] * (wave[-1] / ext_w) ** 4
        return np.concatenate([wave, ext_w]), np.concatenate([flux, ext_f])

    @staticmethod
    def _convolve_to_R(wave, flux, R_hi=1500.0, R_lo=100.0,
                       lam_split=5.0, width=0.45):
        """Convolve a spectrum to a WAVELENGTH-DEPENDENT resolving power and
        return it on a uniform log grid: R_hi in the optical/NIR (keep the real
        molecular bands) tapering smoothly to R_lo across lam_split into the
        mid-IR (where the NewEra extension is an unresolved mess). The smooth
        taper means there is still no resolution seam at the cache/NewEra
        splice. Constant-R Gaussian in log-wavelength."""
        wave = np.asarray(wave, float); flux = np.asarray(flux, float)
        g = np.isfinite(wave) & np.isfinite(flux) & (wave > 0) & (flux > 0)
        if g.sum() < 10:
            return wave, flux
        w = wave[g]; f = flux[g]
        o = np.argsort(w); w, f = w[o], f[o]
        lo, hi = np.log(w[0]), np.log(w[-1])
        npts = int(np.clip(8 * R_hi * (hi - lo), 64, 16000))  # oversample, cap
        lg = np.linspace(lo, hi, npts)
        fg = np.interp(lg, np.log(w), np.log10(np.maximum(f, 1e-300)))
        try:
            from scipy.ndimage import gaussian_filter1d
            dln = (hi - lo) / (npts - 1)
            f_hi = gaussian_filter1d(fg, (1.0 / R_hi) / 2.3548 / dln,
                                     mode='nearest')
            f_lo = gaussian_filter1d(fg, (1.0 / R_lo) / 2.3548 / dln,
                                     mode='nearest')
            wgt = 0.5 * (1.0 - np.tanh((lg - np.log(lam_split)) / width))
            fg = f_hi * wgt + f_lo * (1.0 - wgt)
        except Exception:
            pass
        return np.exp(lg), 10 ** fg

    def _extend_photosphere(self, wave, flux, teff, logg, z, lam_max=30.0):
        """Extend a photosphere spectrum past the cache's red end (~4.63 um)
        using the NewEra grid, which reaches 700 um and covers 2300-11800 K and
        therefore carries the real molecular mid-IR structure -- so the line
        follows a model rather than a straight Rayleigh-Jeans extrapolation. The
        NewEra tail is scaled to match the spectrum at the splice. Falls back to
        the RJ tail when NewEra does not cover the star (e.g. a hot WD)."""
        wave = np.asarray(wave, float); flux = np.asarray(flux, float)
        if wave[-1] >= lam_max:
            return self._convolve_to_R(wave, flux)
        ne = None
        try:
            import h5py
            import os as _os
            if spectra_cache and _os.path.isfile(spectra_cache):
                with h5py.File(spectra_cache, 'r') as f:
                    if 'newera' in f:
                        g = f['newera']
                        tt = g['teff'][:]; ll = g['logg'][:]; zz = g['z'][:]
                        if tt.min() <= teff <= tt.max():
                            ut = np.unique(tt); ul = np.unique(ll); uz = np.unique(zz)
                            tv = ut[np.argmin(np.abs(ut - teff))]
                            lv = ul[np.argmin(np.abs(ul - logg))]
                            zv = uz[np.argmin(np.abs(uz - z))]
                            m = (tt == tv) & (ll == lv) & (zz == zv)
                            ix = np.where(m)[0]
                            if len(ix):
                                ne = (g['wavelength'][:].astype(float),
                                      g['flux'][ix[0]].astype(float))
        except Exception:
            ne = None
        if ne is None:
            return self._convolve_to_R(*self._extend_rj(wave, flux, lam_max))
        nw, nf = ne
        good = np.isfinite(nf) & (nf > 0)
        ext = good & (nw > wave[-1]) & (nw <= lam_max)
        if ext.sum() < 2 or good.sum() < 2:
            return self._convolve_to_R(*self._extend_rj(wave, flux, lam_max))
        # Splice the NewEra tail (scaled to match at the join), then convolve the
        # WHOLE spectrum to a single resolving power so the optical/NIR cache and
        # the mid-IR extension share the same smoothness -- no jarring break.
        f_splice = np.interp(wave[-1], nw[good], nf[good])
        scale = flux[-1] / f_splice if f_splice > 0 else 1.0
        ew = np.concatenate([wave, nw[ext]])
        ef = np.concatenate([flux, nf[ext] * scale])
        return self._convolve_to_R(ew, ef)

    def _get_raw_spectrum(self, model_key, teff, logg, z):
        """Fetch broadened spectrum for given model and stellar params.

        Returns (wave, flux) trimmed to the standard wavelength range and
        instrumentally broadened where applicable, but NOT extincted or
        normalized.  Returns None on failure.
        """
        orig_theta = self.theta.copy()
        self.theta[0], self.theta[1], self.theta[2] = teff, logg, z
        try:
            cached = self._load_from_cache(model_key)
            if cached is not None:
                return cached

            if self.moddir is None:
                return None

            if model_key == 'phoenix':
                if not hasattr(self, '_phoenix_wave'):
                    self._phoenix_wave = fits.open(
                        self.moddir + self.__wav_file
                    )[0].data * u.angstrom.to(u.um)
                wave = self._phoenix_wave
                mask = (0.125 < wave) & (wave < 4.629296073126975)
                flux = self.fetch_Phoenix()
                new_w = wave[mask]
                new_ww = np.linspace(new_w[0], new_w[-1], len(new_w))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_ww, flux, 1500,
                    edgeHandling="firstlast", fullout=True, maxsig=8
                )
                return new_w, brf[mask]

            elif model_key in ('btsettl', 'btnextgen', 'btcond'):
                fetch = {
                    'btsettl': self.fetch_btsettl,
                    'btnextgen': self.fetch_btnextgen,
                    'btcond': self.fetch_btcond,
                }[model_key]
                wave, flux = fetch()
                mask = (0.125 < wave) & (wave < 4.629296073126975)
                wave, flux = wave[mask], flux[mask]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast", fullout=True, maxsig=8
                )
                return wave, brf

            elif model_key in ('ck04', 'kurucz', 'coelho'):
                fetch = {
                    'ck04': self.fetch_ck04,
                    'kurucz': self.fetch_kurucz,
                    'coelho': self.fetch_coelho,
                }[model_key]
                wave, flux = fetch()
                lo = 0.15 if model_key in ('kurucz', 'coelho') else 0.125
                mask = (lo < wave) & (wave < 4.629296073126975)
                return wave[mask], flux[mask]

        except Exception as e:
            logger.warning(
                f'Failed to get spectrum for {model_key} '
                f'(Teff={teff}, logg={logg}, z={z}): {e}'
            )
            return None
        finally:
            self.theta[:] = orig_theta

    def _plot_bma_gray_spectra(self, ax, n_samples=100):
        """Draw gray sampled model spectra to show the posterior uncertainty.

        For BMA the draws are spread across grids by their weights; for a single
        grid they are drawn straight from the posterior. For a binary fit each
        draw is the summed two-component model (primary + secondary), matching
        the best-fit overlay. Component spectra are fetched per draw -- the HDF5
        cache makes the lookups cheap.
        """
        rng = np.random.default_rng()
        Rv = 3.1

        # (grid name, weight, posterior-sample dict) sources to draw from.
        if 'weights' in self.out and 'originals' in self.out:
            models = [(m, w, self.out['originals'][m])
                      for m, w in self.out['weights'].items() if w >= 1e-4]
        else:
            models = [(self.grid, 1.0, self.out['posterior_samples'])]

        def _sv(v, i):
            # fixed parameters are stored as scalars, free ones as arrays
            return v[i] if np.ndim(v) else v

        for model_name, weight, samples in models:
            n_model = max(1, round(weight * n_samples))
            indices = rng.choice(len(samples['teff']), size=n_model,
                                 replace=True)
            self.star.load_grid(model_name)
            for idx in indices:
                z_i = _sv(samples['z'], idx)
                Av = _sv(samples['Av'], idx)
                if not self.norm:
                    dist = _sv(samples['dist'], idx) * u.pc.to(u.solRad)
                    norm1 = (_sv(samples['rad'], idx) / dist) ** 2
                else:
                    norm1 = _sv(samples['norm'], idx)
                s1 = self._get_raw_spectrum(model_name, _sv(samples['teff'], idx),
                                            _sv(samples['logg'], idx), z_i)
                if s1 is None:
                    continue
                wave, f1 = s1
                ext = self.av_law(wave * 1e4, Av, Rv)
                flx = apply(ext, f1.copy()) * norm1 * wave
                if self.binary:
                    s2 = self._get_raw_spectrum(
                        model_name, _sv(samples['teff2'], idx),
                        _sv(samples['logg2'], idx), z_i)
                    if s2 is not None:
                        norm2 = (_sv(samples['rad2'], idx) / dist) ** 2
                        flx = flx + apply(ext, s2[1].copy()) * norm2 * wave
                # Convolve each draw to the same resolution as the best-fit line
                # (IRX view) so the posterior cloud matches its smoothness.
                if getattr(self, 'irx', False):
                    wave, flx = self._convolve_to_R(wave, flx)
                ax.plot(wave, flx, color='gray', alpha=0.15, lw=0.5, zorder=-1)

        # Restore winning model's grid
        self.star.load_grid(self.grid)

    def plot_SED_no_model(self, s=None):
        """Plot raw photometry."""
        if self.star is None:
            self.star = s
        self.__extract_info()
        # Get plot ylims.
        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        f, ax = plt.subplots(figsize=self.figsize)

        # Model plot
        used_f = self.star.filter_names[self.star.filter_mask]
        # Bands flagged by the blackbody QC are outlined in red so outliers
        # can be eyeballed against the BB fit drawn below.
        flagged = set(getattr(self.star, '_qc_bb_flagged_bands', None) or [])

        # Colour each point by its effective wavelength so the SED shape reads
        # directly off the plot: short wavelengths -> violet/blue, long
        # wavelengths -> orange/red.
        lam_log = np.log10(self.wave)
        if lam_log.max() > lam_log.min():
            cnorm = plt.Normalize(vmin=lam_log.min(), vmax=lam_log.max())
        else:
            cnorm = plt.Normalize(vmin=lam_log.min() - 1, vmax=lam_log.max() + 1)
        point_colors = plt.cm.rainbow(cnorm(lam_log))

        for c, w, fl, fe, bp, fi in zip(
                point_colors,
                self.wave, self.flux, self.flux_er,
                self.bandpass, used_f):
            ax.errorbar(w, fl * w,
                        xerr=bp, yerr=fe * w,
                        fmt='',
                        ecolor=c,
                        marker=None)

            ax.scatter(w, fl * w,
                       edgecolors='black',
                       linewidths=0.5,
                       marker=self.marker,
                       c=[c],
                       s=self.scatter_size,
                       alpha=self.scatter_alpha, label=fi)

            # QC-flagged bands get a hollow red halo drawn on top, which stays
            # visible regardless of the point's wavelength colour (the colormap
            # is itself red at long wavelengths).
            if fi in flagged:
                ax.scatter(w, fl * w,
                           edgecolors='red', facecolors='none',
                           linewidths=1.8, marker=self.marker,
                           s=self.scatter_size * 2.4, zorder=4)

        # Blackbody fit from the photometry QC, log-scaled to the non-flagged
        # bands so it overlays the data. Lets the user judge visually whether a
        # red-outlined band genuinely departs from the stellar continuum.
        bb_T = getattr(self.star, '_qc_bb_teff', None)
        if bb_T:
            h_, c_, k_ = 6.62607015e-34, 2.99792458e8, 1.380649e-23

            def _planck(lam_um_arr):
                lam_m = np.asarray(lam_um_arr) * 1e-6
                x = h_ * c_ / (lam_m * k_ * bb_T)
                return 1.0 / (lam_m ** 5 * (np.exp(np.clip(x, 0, 500)) - 1.0))

            wave_um = np.asarray(self.wave)
            F_lam = np.asarray(self.flux)
            trusted = np.array([fi not in flagged for fi in used_f])
            if trusted.sum() < 2:
                trusted = np.ones(len(used_f), dtype=bool)
            B_bands = _planck(wave_um)
            log_scale = np.median(np.log10(F_lam[trusted])
                                  - np.log10(B_bands[trusted]))
            scale = 10.0 ** log_scale
            lam_curve = np.logspace(np.log10(max(wave_um.min() * 0.5, 0.1)),
                                    np.log10(wave_um.max() * 2.0), 400)
            F_curve = _planck(lam_curve) * scale
            ax.plot(lam_curve, F_curve * lam_curve,
                    color='0.4', lw=1.0, ls='--', zorder=1,
                    label=f'BB fit, T={bb_T:.0f} K')

        ax.set_ylim([ymin * .8, ymax * 1.25])
        ax.set_xscale('log', nonpositive='clip')
        ax.set_yscale('log', nonpositive='clip')
        ax.set_ylabel(r'$\lambda$F$_\lambda$ (erg cm$^{-2}$s$^{-1}$)',
                      fontsize=self.fontsize,
                      fontname=self.fontname
                      )
        ax.legend(loc=0)

        ax.tick_params(
            axis='both', which='major',
            labelsize=self.tick_labelsize
        )
        ax.tick_params(
            axis='both', which='minor',
            labelsize=self.tick_labelsize
        )
        # X-axis built like the model SED plot: log scale in micron, integer
        # ticks, lower bound set by whether GALEX UV bands are present.
        ax.set_xticks(np.linspace(1, 10, 10))
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        fnames = self.star.filter_names[self.star.filter_mask]
        if 'GALEX_FUV' in fnames or 'GALEX_NUV' in fnames:
            ax.set_xlim([0.125, 6])
        else:
            ax.set_xlim([0.25, 6])
        ax.set_xlabel(r'$\lambda (\mu m)$',
                      fontsize=self.fontsize,
                      fontname=self.fontname)

        for tick in ax.get_yticklabels():
            tick.set_fontname(self.fontname)

        if self.pdf:
            plt.savefig(self.out_folder + '/SED_no_model.pdf',
                        bbox_inches='tight')
        if self.png:
            plt.savefig(self.out_folder + '/SED_no_model.png',
                        bbox_inches='tight')
        pass

    def plot_SED(self):
        """Create the plot of the SED.

        This is the original SED figure: the fit photometry and the model,
        covering the fitted bands only. IR-excess (irx) data are NOT shown
        here -- use ``IRX_SED`` for the SED + IR-excess view.
        """
        self._render_sed(irx=False, basename='SED')

    def IRX_SED(self):
        """Create the SED plot including the IR-excess (irx) photometry.

        Identical to ``plot_SED`` but additionally overlays the irx bands
        (and their residuals) and extends the wavelength range into the IR.
        """
        self._render_sed(irx=True, basename='IRX_SED')

    def _render_sed(self, irx=False, basename='SED'):
        """Render the SED figure to ``<out_folder>/<basename>.{png,pdf}``.

        With ``irx=False`` this reproduces the original fit-only SED; with
        ``irx=True`` the IR-excess photometry/residuals are added and the
        x-range is extended. Driven by :meth:`plot_SED` and :meth:`IRX_SED`.
        """
        if self.moddir is None and not spectra_cache:
            logger.warning('Models directory not provided and no spectra cache set, skipping SED plot')
            return
        print('Plotting SED')
        # Only show the irx bands when explicitly requested AND present.
        show_irx = bool(irx and getattr(self, 'irx', False))
        # Get plot ylims.
        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        if show_irx:
            # Scale the y-range to the DATA, including the IR excess points --
            # NOT the declining photosphere model at the irx bands, which (for a
            # WD especially) plunges far below the data and leaves a swathe of
            # empty axis with the points crammed at the top. Use the finite,
            # strictly positive lambda*F_lambda of the irx bands (a NaN or
            # exactly-0 model band must not collapse the log axis).
            _ird = np.asarray(self.irx_flux, float) * np.asarray(self.irx_wave, float)
            _ird = _ird[np.isfinite(_ird) & (_ird > 0)]
            if _ird.size:
                ymin = min(float(ymin), float(_ird.min()))
                ymax = max(float(ymax), float(_ird.max()))

        n_filt = self.star.used_filters.sum()
        n_pars = int(len(self.theta) - n_filt)

        # Get models residuals
        mask = self.star.filter_mask
        mags = self.star.mags[mask]
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        filters = self.star.filter_names[mask]
        wave = self.star.wave[mask]

        for i, th in enumerate(self.theta[n_pars:]):
            mag = mags[i]
            filt = filters[i]
            _, er = mag_to_flux(mag, th, filt)
            self.theta[n_pars + i] = er

        residuals, errors = get_residuals(
            self.theta, flxs, errs, wave, filters, self.interpolator, self.norm,
            self.av_law, binary=self.binary)

        norm_res = residuals / errors

        # Create plot layout

        f = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 0.75], hspace=0.05)

        ax = f.add_subplot(gs[0])
        ax_r = f.add_subplot(gs[1])

        # Gray posterior-draw spectra: BMA draws across grids by weight, binary
        # draws the summed two-component model, single-grid draws straight from
        # the posterior (the method handles all three; gating on bma/binary
        # silently dropped the single-grid draws).
        try:
            self._plot_bma_gray_spectra(ax)
        except Exception as e:
            logger.warning('gray posterior spectra skipped: %s', e)
        self.SED(ax)

        # Keep the whole model curve in frame. The y-range is scaled to the
        # data, but if the continuous model peaks above the data (e.g. the SED
        # peak falls between photometric bands) extend the upper limit to the
        # model so it isn't clipped. Also capture the photosphere spectrum (the
        # model line) for the SED+IRX total overlay. ax.get_lines() here is just
        # the model curve(s) from SED()/the gray spectra -- photometry follows.
        ph_w = ph_f = None
        for _ln in ax.get_lines():
            _yd = np.asarray(_ln.get_ydata())
            if len(_yd):
                ymax = max(ymax, float(np.nanmax(_yd)))
                if (_ln.get_color() == self.model_color
                        and (ph_f is None or len(_yd) > len(ph_f))):
                    ph_w, ph_f = np.asarray(_ln.get_xdata()), _yd

        # Model plot
        # Observed photometry (and its error bars) renders ABOVE every model
        # curve/draw (curves z=2, posterior draws z=1, synthetic-model markers
        # z=3): data on top.
        ax.errorbar(self.wave, self.flux * self.wave,
                    xerr=self.bandpass, yerr=errors,
                    fmt=',', ecolor=self.error_color, zorder=4,
                    marker=None)
        ax.scatter(self.wave, self.flux * self.wave,
                   edgecolors='black', marker=self.marker, c=self.marker_colors,
                   s=self.scatter_size, zorder=5, alpha=self.scatter_alpha)
        ax.scatter(self.wave, self.model * self.wave,
                   marker=self.marker_model,
                   edgecolors=self.marker_colors_model, s=self.scatter_size,
                   facecolor='none', zorder=3, lw=3)

        if show_irx:
            ax.errorbar(self.irx_wave, self.irx_flux * self.irx_wave,
                        xerr=self.irx_bandpass, yerr=self.irx_flux_er,
                        fmt=',', ecolor=self.irx_error_color, zorder=4,
                        marker=None)
            ax.scatter(self.irx_wave, self.irx_flux * self.irx_wave,
                       edgecolors='black', marker=self.marker,
                       c=self.marker_colors_irx, s=self.scatter_size, zorder=5,
                       alpha=self.scatter_alpha)
            # Model diamonds at the irx bands are DEFERRED until the winning
            # excess model is known below: they must show photosphere + the
            # winning model -- a bare-photosphere diamond sitting under an
            # excess point reads as a bad fit when the total is exactly what
            # the model predicts. (Scatter happens after the overlay block.)
            #
            # Build the photosphere part from the DISPLAYED photosphere CURVE
            # (ph_w, ph_f) sampled at the bands, not the stored per-band model:
            # the diamonds must sit on the same model line the eye follows, and
            # the band grid can carry artifacts the smooth spectrum does not
            # (a koester WD's W2 column is ~4x high and its IRAC-4.5 column is
            # 0). Fall back to the stored per-band model where the curve does
            # not reach.
            irx_model_display = np.asarray(self.irx_model, float).copy()
            if ph_w is not None and ph_f is not None \
                    and len(np.atleast_1d(ph_w)) > 1:
                _pw = np.asarray(ph_w, float)
                _pf = np.asarray(ph_f, float)
                _okp = np.isfinite(_pw) & np.isfinite(_pf) & (_pf > 0)
                if _okp.sum() > 1:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        _ph_lf = 10 ** np.interp(
                            np.log10(self.irx_wave), np.log10(_pw[_okp]),
                            np.log10(_pf[_okp]), left=np.nan, right=np.nan)
                    _ph_fl = _ph_lf / self.irx_wave   # lam*F_lam -> F_lam
                    _usep = np.isfinite(_ph_fl)
                    irx_model_display[_usep] = _ph_fl[_usep]

            # Overlay the fitted IR-excess model(s) (IRX view only). The DISK
            # component is the LS single-T blackbody (dust really is one); the
            # COMPANION component is the actual grid photosphere at the BMA
            # (Teff2, R2) -- a blackbody badly over-predicts 5-10 um where L/T
            # photospheres are carved by CH4/H2O. Totals (photosphere + excess)
            # are drawn over the photosphere's full range so they ride the black
            # line where the excess is negligible and peel off where it is not.
            # A clean classification shows one model; an ambiguous one shows
            # BOTH, in different colors/styles, each labeled with its posterior
            # probability. The companion carries light-red posterior draws
            # (per-grid BMA samples), mirroring the gray photosphere draws.
            fit = getattr(self, 'irx_fit', None)
            bb = (fit or {}).get('disk_bb') if fit else None
            _cls = (fit or {}).get('excess_class') if fit else None
            _comp = (fit or {}).get('companion') if fit else None
            p_d = float((fit or {}).get('p_disk') or 0.0)
            p_c = float((fit or {}).get('p_companion') or 0.0)
            have_bb = bool(bb) and np.isfinite(bb.get('t_dust', np.nan))
            show_comp = bool(_cls in ('companion', 'ambiguous') and _comp
                             and np.isfinite(_comp.get('teff2', np.nan))
                             and not self.norm)
            show_disk = have_bb and (not show_comp or _cls == 'ambiguous')
            both = show_comp and show_disk
            if (show_disk or show_comp) and ph_w is not None \
                    and ph_f is not None:
                from scipy.constants import h as _h, c as _c, k as _kB

                def _dustbb(lam_um, T, omega):
                    # Modified blackbody (beta=1 beyond 210 um), matching the
                    # fitter's disk convention.
                    lam_um = np.asarray(lam_um, float)
                    lam = lam_um * 1e-6
                    with np.errstate(over='ignore'):
                        planck = (2 * _h * _c ** 2 / lam ** 5) \
                            / np.expm1(_h * _c / (lam * _kB * T)) * 1e-3
                    fac = np.where(lam_um > 210.0, 210.0 / lam_um, 1.0)
                    return omega * planck * fac * lam_um

                # Draw the 2-T disk only when it DECISIVELY beats single-T
                # (same 0.75 convention as the classification): at even odds
                # the second component is prior-volume wash, not a detection.
                # Display uses the MAP pair -- marginal means of a weakly
                # constrained component are not a real solution.
                _d2 = (fit or {}).get('disk2')
                p_d2 = float((fit or {}).get('p_disk2') or 0.0)
                _use2 = (bool(_d2) and 't_warm_map' in (_d2 or {})
                         and (p_d + p_d2) > 0
                         and p_d2 / (p_d + p_d2) > 0.75)

                def _diskbb(lam_um):
                    if _use2:
                        return (_dustbb(lam_um, _d2['t_warm_map'],
                                        _d2['omega_warm_map'])
                                + _dustbb(lam_um, _d2['t_cold_map'],
                                          _d2['omega_cold_map']))
                    return _dustbb(lam_um, bb['t_dust'], bb['omega_sr'])

                def _comp_spectrum(gname, t2):
                    # Full-resolution companion spectrum from the cool-spectra
                    # cache (Datafiles/cool_spectra.h5, R~300 plotting cache):
                    # nearest feh node to the primary's z, logg node nearest
                    # 5.0, log-flux interpolated between the bracketing teff
                    # nodes. Returns (wave_um, surface F_lambda) or None.
                    cc = _load_cool_cache(gname)
                    if cc is None:
                        return None
                    wave, te, lo, fe, fl = cc
                    zsel = fe[int(np.argmin(np.abs(fe - self.theta[2])))]
                    m1 = fe == zsel
                    lsel = lo[m1][int(np.argmin(np.abs(lo[m1] - 5.0)))]
                    m2 = m1 & (lo == lsel)
                    ts = np.unique(te[m2])
                    if not len(ts):
                        return None
                    t2c = float(np.clip(t2, ts[0], ts[-1]))
                    i = int(np.searchsorted(ts, t2c))
                    tlo = ts[max(i - 1, 0)]
                    thi = ts[min(i, len(ts) - 1)] if t2c > ts[0] else ts[0]
                    slo = fl[np.where(m2 & (te == tlo))[0][0]]
                    if thi == tlo:
                        s = slo
                    else:
                        shi = fl[np.where(m2 & (te == thi))[0][0]]
                        u_ = (t2c - tlo) / (thi - tlo)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            s = 10 ** ((1 - u_) * np.log10(slo)
                                       + u_ * np.log10(shi))
                    return wave, s

                def _comp_curve(t2, r2, gname, ig=None):
                    # Companion lambda*F_lambda curve at (t2, r2): the cached
                    # spectrum when available (smooth), else band fluxes through
                    # the interpolation grid. Redward of ~1 um only -- the cool
                    # body is negligible in the optical and the structure there
                    # just clutters the plot.
                    try:
                        sp = _comp_spectrum(gname, t2)
                        if sp is not None:
                            wave, s = sp
                            dist_sr = self.theta[3] * u.pc.to(u.solRad)
                            flx = apply(self.av_law(wave * 1e4, self.theta[5],
                                                    3.1),
                                        s * (r2 / dist_sr) ** 2)
                            good = (np.isfinite(flx) & (flx > 0)
                                    & (wave >= 0.95))
                            if good.sum() > 10:
                                return wave[good], (flx * wave)[good]
                        if ig is None:
                            return None
                        la, _ta, fa = ig.index_columns
                        zq = float(np.asarray(fa)[int(np.argmin(np.abs(
                            np.asarray(fa) - self.theta[2])))])
                        lq = float(np.asarray(la)[int(np.argmin(np.abs(
                            np.asarray(la) - 5.0)))])
                        bands = np.asarray(
                            [b for b in self.star.filter_names
                             if b in ig.column_index])
                        wv = np.array([get_effective_wavelength(b)
                                       for b in bands])
                        ok = np.isfinite(wv)
                        bands, wv = bands[ok], wv[ok]
                        th2 = np.array([t2, lq, zq, self.theta[3], r2,
                                        self.theta[5]])
                        cf = np.asarray(model_grid(
                            th2, bands, wv, ig, self.norm,
                            self.av_law)).ravel()
                        good = np.isfinite(cf) & (cf > 0) & (wv >= 0.95)
                        if good.sum() < 4:
                            return None
                        order = np.argsort(wv[good])
                        return wv[good][order], (cf[good] * wv[good])[order]
                    except Exception:
                        return None

                def _comp_band_flux(t2, r2, ig=None):
                    """Companion band fluxes (F_lambda) at the irx bands via
                    proper grid convolution -- same source as the mean curve.
                    ``ig`` is the companion's OWN (cool) grid; default to the
                    primary interpolator only as a last resort. NaN where the
                    grid lacks the band (the companion is negligible there
                    anyway, e.g. SPIRE)."""
                    out = np.full(len(self.irx_wave), np.nan)
                    try:
                        ig = ig if ig is not None else self.interpolator
                        la, _ta, fa = ig.index_columns
                        zq = float(np.asarray(fa)[int(np.argmin(np.abs(
                            np.asarray(fa) - self.theta[2])))])
                        lq = float(np.asarray(la)[int(np.argmin(np.abs(
                            np.asarray(la) - 5.0)))])
                        bands = self.star.filter_names[
                            self.star.irx_filter_mask]
                        keep = np.array([b in ig.column_index for b in bands])
                        if not keep.any():
                            return out
                        th2 = np.array([t2, lq, zq, self.theta[3], r2,
                                        self.theta[5]])
                        out[keep] = np.asarray(model_grid(
                            th2, bands[keep], self.irx_wave[keep], ig,
                            self.norm, self.av_law)).ravel()
                    except Exception:
                        pass
                    return out

                def _total(wb, cb):
                    # From ~1 um on: the total is indistinguishable from the
                    # photosphere there, so it emerges smoothly from the black
                    # line (no orphaned segment) without painting over the
                    # jagged optical spectrum.
                    m = (ph_w >= max(wb[0], 0.95)) & (ph_w <= wb[-1])
                    if not m.any():
                        return None
                    ci = 10 ** np.interp(np.log10(ph_w[m]),
                                         np.log10(wb), np.log10(cb))
                    return ph_w[m], ph_f[m] + ci

                if show_comp:
                    comp_color = 'tab:red'

                    # Start the companion curve a short lead blueward of the
                    # first IR-excess band, so it leads INTO the excess rather
                    # than running deep into the photosphere. Falls back to the
                    # cross-over fraction if the star has no IRX bands.
                    def _trim_blue(wb, cb, lead=0.8):
                        wb = np.asarray(wb, float)
                        cb = np.asarray(cb, float)
                        iw = np.asarray(self.irx_wave, float)
                        iw = iw[np.isfinite(iw) & (iw > 0)]
                        if len(iw):
                            w0 = lead * float(iw.min())
                        else:
                            ph_at = 10 ** np.interp(
                                np.log10(wb), np.log10(ph_w),
                                np.log10(np.maximum(ph_f, 1e-300)))
                            sig = np.where(cb >= 0.25 * ph_at)[0]
                            w0 = wb[sig[0]] if len(sig) else wb[0]
                        m = wb >= w0
                        return wb[m], cb[m]

                    # Smooth the companion spectrum down a notch: the R~300 cool
                    # cache carries deep CH4/H2O molecular bands that read as
                    # noisy/jagged at plot scale. A boxcar in log-flux keeps the
                    # overall shape while calming the wiggles. (The model DIAMONDS
                    # still use the proper band-convolved fluxes, untouched.)
                    def _smooth(f, n=15):
                        f = np.asarray(f, float)
                        if len(f) < n:
                            return f
                        try:
                            from scipy.ndimage import uniform_filter1d
                            lf = np.log10(np.maximum(f, 1e-300))
                            return 10 ** uniform_filter1d(lf, size=n,
                                                          mode='nearest')
                        except Exception:
                            return f

                    # Light-red posterior draws of the cool body: sample
                    # (Teff2, R2) per grid from the companion BMA and draw the
                    # grid curve, mirroring the gray photosphere draws.
                    pg = _comp.get('per_grid') or {}
                    if pg:
                        rng = np.random.default_rng()
                        gnames = list(pg)
                        gw = np.array([pg[g]['weight'] for g in gnames], float)
                        gw = gw / gw.sum() if gw.sum() > 0 else None
                        gcache = {}

                        def _gi(name):
                            if name == self.grid:
                                return self.interpolator
                            if name not in gcache:
                                try:
                                    gcache[name] = load_grid(name)
                                except Exception:
                                    gcache[name] = None
                            return gcache[name]

                        for _ in range(25):
                            gn = rng.choice(gnames, p=gw)
                            e = pg[gn]
                            # Correlated (T2, R2) posterior samples from the
                            # quadrature (T2-R2 are strongly anticorrelated;
                            # independent marginal draws over-disperse).
                            smp = e.get('samples')
                            if smp:
                                t2, r2 = smp[int(rng.integers(len(smp)))]
                            else:
                                t2 = rng.normal(e['teff2'],
                                                e.get('teff2_err', 0.0) or 0.0)
                                r2 = rng.normal(e['rad2'],
                                                e.get('rad2_err', 0.0) or 0.0)
                            if r2 <= 0:
                                continue
                            cur = _comp_curve(t2, r2, gn, _gi(gn))
                            if cur is not None:
                                _dw, _dc = _trim_blue(cur[0], _smooth(cur[1]))
                                ax.plot(_dw, _dc, color='lightcoral',
                                        lw=0.8, alpha=0.12, zorder=1)
                    # Companion display grid: the highest-weight per-grid member
                    # that has a cool spectra cache (_COOL_GROUPS), so the curve
                    # is a smooth R~300 model spectrum reaching the mid-IR rather
                    # than a jagged line through the ~40 band points (which is
                    # what self.grid=btcond falls back to -- btcond has no cool
                    # cache group). Same grid feeds the model diamonds below.
                    _pg = _comp.get('per_grid') or {}
                    _cgc = sorted([(_pg[g].get('weight', 0.0), g)
                                   for g in _pg if g in _COOL_GROUPS
                                   and _pg[g].get('weight', 0.0) > 0],
                                  reverse=True)
                    _cg = _cgc[0][1] if _cgc else self.grid
                    try:
                        _cig = (self.interpolator if _cg == self.grid
                                else load_grid(_cg))
                    except Exception:
                        _cg, _cig = self.grid, self.interpolator
                    cur = _comp_curve(_comp['teff2'], _comp['rad2'], _cg, _cig)
                    if cur is not None:
                        wb, cb = cur
                        cb = _smooth(cb)
                        _cw, _cc = _trim_blue(wb, cb)
                        ax.plot(_cw, _cc, ls='--', lw=1.3, color=comp_color,
                                alpha=0.85, zorder=2,
                                label=(r'companion (P=%.2f): T$_2$=%.0f K, '
                                       r'R$_2$=%.2f R$_\odot$'
                                       % (p_c, _comp['teff2'],
                                          _comp.get('rad2', np.nan))))
                        tot = _total(wb, cb)
                        if tot is not None:
                            # Trim the COMBINED (star+companion) curve to the
                            # same cross-over: blueward of it the total is
                            # indistinguishable from the photosphere (black
                            # line), so drawing it there paints a thick red mass
                            # over the photosphere. Show it only where the
                            # companion lifts the total off the star.
                            tw = np.asarray(tot[0], float)
                            tf = np.asarray(tot[1], float)
                            if len(_cw):
                                _tm = tw >= _cw[0]
                                tw, tf = tw[_tm], tf[_tm]
                            ax.plot(tw, tf, lw=1.75, color=comp_color,
                                    ls='-.', zorder=2,
                                    label='star + companion model')
                    # The winner contributes to the model diamonds, via the same
                    # cool grid as the displayed curve so they stay consistent.
                    if not show_disk or p_c >= p_d:
                        cf = _comp_band_flux(_comp['teff2'], _comp['rad2'], _cig)
                        ok = np.isfinite(cf)
                        irx_model_display[ok] += cf[ok]
                if show_disk:
                    # The winner contributes to the model diamonds.
                    if not show_comp or p_d > p_c:
                        with np.errstate(invalid='ignore'):
                            irx_model_display += (_diskbb(self.irx_wave)
                                                  / self.irx_wave)
                    # When both hypotheses are shown the disk switches to green
                    # with its own line styles so the two are unmistakable; a
                    # clean disk keeps the established red styling.
                    disk_color = 'seagreen' if both else 'tab:red'
                    d_ls, t_ls = (':', '--') if both else ('--', '-.')
                    md = ph_w >= 0.95
                    ax.plot(ph_w[md], ph_f[md] + _diskbb(ph_w[md]), lw=1.75,
                            color=disk_color, ls=t_ls, zorder=2,
                            label='star + disk model')
                    lam_um = np.geomspace(0.95, 300.0, 400)
                    if _use2:
                        # Two-temperature disk: warm + cold components shown
                        # individually, the summed curve labeled with both.
                        for _tk, _ok in (('t_warm_map', 'omega_warm_map'),
                                         ('t_cold_map', 'omega_cold_map')):
                            ax.plot(lam_um,
                                    _dustbb(lam_um, _d2[_tk], _d2[_ok]),
                                    ls=':', lw=1.0, color=disk_color,
                                    alpha=0.6, zorder=2)
                        ax.plot(lam_um, _diskbb(lam_um), ls=d_ls, lw=1.3,
                                color=disk_color, alpha=0.85, zorder=2,
                                label=(r'2-T disk (P=%.2f): T$_w$=%.0f K, '
                                       r'T$_c$=%.0f K'
                                       % (p_d + p_d2, _d2['t_warm_map'],
                                          _d2['t_cold_map'])))
                    else:
                        ax.plot(lam_um, _diskbb(lam_um), ls=d_ls, lw=1.3,
                                color=disk_color, alpha=0.85, zorder=2,
                                label=(r'disk BB (P=%.2f): T$_d$=%.0f K, '
                                       r'L$_d$/L$_*$=%.1g'
                                       % (p_d + p_d2, bb['t_dust'],
                                          bb['l_dust_over_lstar'])))
                ax.legend(loc='upper right', fontsize=9, frameon=False)

            # Deferred model diamonds: photosphere + winning excess model. Open,
            # same size as the photosphere diamonds, drawn BEHIND the data so the
            # diamond corners frame each observed point. Every band now carries a
            # finite model (the per-band BMA models read at init), so a diamond
            # is drawn for every band -- the earlier "missing" diamonds were NaN
            # models from the forced btsettl grid, not occlusion.
            ax.scatter(self.irx_wave, irx_model_display * self.irx_wave,
                       marker=self.marker_model,
                       edgecolors=self.marker_colors_model_irx,
                       s=self.scatter_size, facecolor='none', zorder=3, lw=3)

        # Residual plot
        ax_r.axhline(y=0, lw=2, ls='--', c='k', alpha=.7)

        ax_r.errorbar(self.wave, norm_res, zorder=3,
                      xerr=self.bandpass, yerr=np.array(self.flux_er) / errors,
                      fmt=',', ecolor=self.error_color, marker=None)
        ax_r.scatter(self.wave, norm_res, zorder=3,
                     edgecolors='black', marker=self.marker,
                     c=self.marker_colors, s=self.scatter_size,
                     alpha=self.scatter_alpha)
        # ax_r.scatter(self.wave, norm_res,
        #              marker=self.marker_model,
        #              edgecolors=self.marker_colors_model, s=self.scatter_size,
        #              facecolor='none', lw=3, zorder=10)

        if show_irx:
            # Residuals of the DISPLAYED total model (photosphere + winning
            # excess = the model diamonds above), so the panel AGREES with the
            # top panel. Using the bare-photosphere excess significance
            # (self.irx_sigma) made a band whose total model sits on the data
            # read as a ~50 sigma "residual", contradicting the diamonds.
            _err = getattr(self, 'irx_band_err', None)
            if _err is None or not np.isfinite(np.asarray(_err, float)).any():
                _err = self.irx_flux_er
            with np.errstate(invalid='ignore', divide='ignore'):
                irx_res = ((self.irx_flux - irx_model_display)
                           / np.asarray(_err, float))
            # Plot-only bands (outside the ensemble's common quantification
            # reach, e.g. while a grid's new columns await reconvolution)
            # carry no reported sigma -- but the PLOT grid does predict them
            # (the diamonds above), so show their residuals as open markers:
            # visibly provisional, never confused with quantified evidence.
            irx_res = np.asarray(irx_res, float)
            _po = ~np.isfinite(irx_res) & np.isfinite(self.irx_model) \
                & np.isfinite(self.irx_flux) & (self.irx_flux_er > 0)
            if _po.any():
                _po_res = ((self.irx_flux[_po] - self.irx_model[_po])
                           / self.irx_flux_er[_po])
                ax_r.scatter(self.irx_wave[_po], _po_res, zorder=3,
                             marker=self.marker, facecolor='none',
                             edgecolors='darkorange', s=self.scatter_size,
                             lw=1.5)
            ax_r.errorbar(self.irx_wave, irx_res, zorder=3,
                          xerr=self.irx_bandpass, yerr=np.ones_like(irx_res),
                          fmt=',', ecolor=self.irx_error_color, marker=None)
            ax_r.scatter(self.irx_wave, irx_res, zorder=3,
                         edgecolors='black', marker=self.marker,
                         c=self.marker_colors_irx, s=self.scatter_size,
                         alpha=self.scatter_alpha)

        # Formatting
        # A NaN residual (e.g. a plot-only band whose model flux is NaN at a
        # sparse cool-splice node) must not poison the residual-panel scale.
        _nr = np.asarray(norm_res, float)
        _nr = _nr[np.isfinite(_nr)]
        res_std = float(_nr.std()) if _nr.size else 1.0
        # The IRX view gets extra headroom so the upper-right legend sits in
        # clean space above the SED peak.
        # IRX view: enough headroom for the upper-right legend without the
        # top third of the panel going empty (x3.0 left the SED floating in
        # blank space for warm-excess stars whose curves end early).
        ax.set_ylim([ymin * 0.6, ymax * (2.0 if show_irx else 1.5)])
        # ax_r.set_ylim([-5, 5])
        if show_irx:
            # Symmetric scale that snugly fits every residual (fit + irx). Using
            # 5*std instead lets one outlier band inflate the range and leave the
            # panel mostly empty; this keeps all points visible without that.
            _r = np.concatenate([norm_res, np.asarray(irx_res, dtype=float)])
            _r = _r[np.isfinite(_r)]
            _m = max(5.0, float(np.nanmax(np.abs(_r)))) * 1.1
            ax_r.set_ylim([-_m, _m])
        else:
            ax_r.set_ylim([-5 * res_std, 5 * res_std])
        ax.set_xscale('log', nonpositive='clip')
        ax.set_yscale('log', nonpositive='clip')
        ax_r.set_xscale('log', nonpositive='clip')
        ax_r.set_xlabel(r'$\lambda (\mu m)$',
                        fontsize=self.fontsize,
                        fontname=self.fontname
                        )
        ax.set_ylabel(r'$\lambda$F$_\lambda$ (erg cm$^{-2}$s$^{-1}$)',
                      fontsize=self.fontsize,
                      fontname=self.fontname
                      )
        ax_r.set_ylabel('Residuals\n$(\\sigma)$',
                        fontsize=self.fontsize,
                        fontname=self.fontname
                        )

        ax.tick_params(
            axis='both', which='major',
            labelsize=self.tick_labelsize
        )
        ax.tick_params(
            axis='both', which='minor',
            labelsize=self.tick_labelsize
        )
        ax_r.tick_params(
            axis='both', which='major',
            labelsize=self.tick_labelsize
        )
        xticks = np.linspace(1, 10, 10)
        if show_irx:
            # Red edge follows the data: reddest irx band plus margin, instead
            # of a fixed 250 um that leaves a long empty tail for stars whose
            # coverage stops at W4.
            _xmax = float(np.nanmax(self.irx_wave)) * 1.5
            xticks = [t for t in (1, 3, 5, 10, 20, 50, 100, 250)
                      if t <= _xmax]
        ax_r.set_xticks(xticks)
        ax_r.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(ticker.NullFormatter())
        ylocmin = ticker.LinearLocator(numticks=4)

        ax_r.yaxis.set_minor_locator(ylocmin)
        ax_r.yaxis.set_minor_formatter(ticker.NullFormatter())

        xlims1 = [0.125, 6]
        xlims2 = [0.25, 6]
        if show_irx:
            xlims1 = [0.125, _xmax]
            xlims2 = [0.25, _xmax]

        if 'GALEX_FUV' in self.star.filter_names[self.star.filter_mask] or \
                'GALEX_NUV' in self.star.filter_names[self.star.filter_mask]:
            ax.set_xlim(xlims1)
            ax_r.set_xlim(xlims1)
        else:
            ax.set_xlim(xlims2)
            ax_r.set_xlim(xlims2)

        labels = [item.get_text() for item in ax.get_xticklabels()]

        empty_string_labels = [''] * len(labels)
        ax.set_xticklabels(empty_string_labels)

        for tick in ax.get_yticklabels():
            tick.set_fontname(self.fontname)
        for tick in ax_r.get_yticklabels():
            tick.set_fontname(self.fontname)
        for tick in ax_r.get_xticklabels():
            tick.set_fontname(self.fontname)

        if self.pdf:
            plt.savefig(f'{self.out_folder}/{basename}.pdf',
                        bbox_inches='tight')
        if self.png:
            plt.savefig(f'{self.out_folder}/{basename}.png',
                        bbox_inches='tight')
        if self.save:
            data = np.vstack((self.wave, self.model * self.wave)).T
            np.savetxt(f'{self.out_folder}/synthetic.dat', data, fmt='%s',
                       header='wavelength(mu m) wave*flux(erg cm-2 s-2)')
        pass

    def SED(self, ax):
        """Plot the SED model."""
        # Grids with a dedicated branch below draw from the cache or the raw
        # model files. Anything else (sphinx, tlusty, bosz, stagger, btdusty,
        # the rebuilt newera, ...) is drawn straight from its spectra_cache
        # group when one exists -- nearest-node spectrum, extincted and scaled
        # like every other curve. Only a grid with NO cache group degrades to
        # photometry + synthetic-flux markers.
        _spectra_grids = ('phoenix', 'btsettl', 'btnextgen', 'btcond',
                          'ck04', 'kurucz', 'coelho')
        _generic = None
        if self.grid.lower() not in _spectra_grids:
            _generic = self._load_from_cache(self.grid.lower())
            if _generic is None:
                logger.warning(
                    f"Grid '{self.grid}' has no spectra in the cache; plotting "
                    f"the SED without the continuous model spectrum "
                    f"(photometry and synthetic model fluxes are still shown).")
                return
        Rv = 3.1  # For extinction.
        if not self.norm:
            rad = self.theta[4]
            dist = self.theta[3] * u.pc.to(u.solRad)
            norm = (rad / dist) ** 2
            Av = self.theta[5]
        else:
            norm = self.theta[3]
            Av = self.theta[4]

        if _generic is not None and not self.binary:
            wave, flux = _generic
            good = np.isfinite(flux) & (flux > 0)
            wave, flux = wave[good], flux[good]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flx = apply(ext, flux) * norm * wave
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)
            return

        # SED plot.
        if self.binary:
            # Two-component continuous spectrum: sum the components (sharing
            # dist, z and Av) and overlay each. Reuses the per-grid spectrum
            # fetcher; if the grid has no cached spectra it draws nothing (the
            # band-flux model points are still shown).
            norm2 = (self.theta[8] / dist) ** 2     # (rad2/dist)^2, dist solRad
            gk = self.grid.lower()
            s1 = self._get_raw_spectrum(gk, self.theta[0], self.theta[1],
                                        self.theta[2])
            s2 = self._get_raw_spectrum(gk, self.theta[6], self.theta[7],
                                        self.theta[2])
            if s1 is not None and s2 is not None:
                w, f1 = s1
                _, f2 = s2
                ext = self.av_law(w * 1e4, Av, Rv)
                comp1 = apply(ext, f1 * norm) * w
                comp2 = apply(ext, f2 * norm2) * w
                ax.plot(w, comp1, lw=0.8, ls='--', color=self.model_color,
                        alpha=0.55, zorder=0)
                ax.plot(w, comp2, lw=0.8, ls=':', color='tab:red',
                        alpha=0.6, zorder=0)
                ax.plot(w, comp1 + comp2, lw=1.25, color=self.model_color,
                        zorder=0)
        elif self.grid == 'phoenix':
            cached = self._load_from_cache('phoenix')
            if cached is not None and not self.irx:
                new_w, brf = cached
            else:
                wave = fits.open(self.moddir + self.__wav_file)[0].data
                wave *= u.angstrom.to(u.um)
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                flux = self.fetch_Phoenix()
                new_w = wave[lower_lim * upper_lim]
                new_ww = np.linspace(new_w[0], new_w[-1], len(new_w))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_ww, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
                brf = brf[lower_lim * upper_lim]
            ext = self.av_law(new_w * 1e4, Av, Rv)
            brf = apply(ext, brf)
            flx = brf * norm * new_w
            ax.plot(new_w, flx, lw=1.25, color=self.model_color,
                    zorder=0)

        elif self.grid == 'btsettl':
            cached = self._load_from_cache('btsettl')
            if cached is not None:
                wave, brf = cached
                if self.irx:
                    wave, brf = self._extend_photosphere(
                        wave, brf, self.theta[0], self.theta[1], self.theta[2])
            else:
                wave, flux = self.fetch_btsettl()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
            ext = self.av_law(wave * 1e4, Av, Rv)
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'btnextgen':
            cached = self._load_from_cache('btnextgen')
            if cached is not None:
                wave, brf = cached
                if self.irx:
                    wave, brf = self._extend_photosphere(
                        wave, brf, self.theta[0], self.theta[1], self.theta[2])
            else:
                wave, flux = self.fetch_btnextgen()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
            ext = self.av_law(wave * 1e4, Av, Rv)
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'btcond':
            cached = self._load_from_cache('btcond')
            if cached is not None:
                wave, brf = cached
                if self.irx:
                    wave, brf = self._extend_photosphere(
                        wave, brf, self.theta[0], self.theta[1], self.theta[2])
            else:
                wave, flux = self.fetch_btcond()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
            ext = self.av_law(wave * 1e4, Av, Rv)
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'newera':
            # NewEra has no external-drive fetch path; its plotting spectra live
            # only in the cache. Drawn whenever the BMA gives the star to NewEra
            # so the black best-fit line matches the (NewEra-weighted) gray draws.
            cached = self._load_from_cache('newera')
            if cached is not None:
                wave, flux = cached
                if self.irx:
                    wave, flux = self._extend_photosphere(
                        wave, flux, self.theta[0], self.theta[1], self.theta[2])
                ext = self.av_law(wave * 1e4, Av, Rv)
                flux = apply(ext, flux) * wave * norm
                ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)
            else:
                logger.warning('newera spectra not in cache; photosphere '
                               'curve skipped')
                wave, flux = np.array([]), np.array([])

        elif self.grid == 'ck04':
            cached = self._load_from_cache('ck04')
            if cached is not None and not self.irx:
                wave, flux = cached
            else:
                wave, flux = self.fetch_ck04()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'kurucz':
            cached = self._load_from_cache('kurucz')
            if cached is not None and not self.irx:
                wave, flux = cached
            else:
                wave, flux = self.fetch_kurucz()
                lower_lim = 0.15 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'coelho':
            cached = self._load_from_cache('coelho')
            if cached is not None:
                wave, flux = cached
            else:
                wave, flux = self.fetch_coelho()
                lower_lim = 0.15 < wave
                upper_lim = wave < 4.629296073126975
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)
        elif self.grid == 'koester':
            # WD photosphere from the koester spectra cache. Koester DA models
            # span 0.09-3 um; redward (the IRX bands) the WD continuum is pure
            # Rayleigh-Jeans, so _extend_rj is exact. No external models needed.
            cached = self._load_from_cache('koester')
            if cached is not None:
                wave, flux = cached
                if self.irx:
                    wave, flux = self._extend_photosphere(
                        wave, flux, self.theta[0], self.theta[1], self.theta[2])
                ext = self.av_law(wave * 1e4, Av, Rv)
                flux = apply(ext, flux) * wave * norm
                ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)
            else:
                logger.warning('koester spectra not in cache; WD photosphere '
                               'curve skipped')
                wave, flux = np.array([]), np.array([])
        else:
            # Generic fallback for any other BMA-winnable grid (atmo, bosz,
            # btdusty, btsettl_cool, sonora, sphinx, stagger, tlusty, ...):
            # draw straight from its spectra-cache group with the IRX mid-IR
            # extension. Keeps EVERY grid plottable without a bespoke branch.
            cached = self._load_from_cache(self.grid)
            if cached is not None:
                wave, flux = cached
                if self.irx:
                    wave, flux = self._extend_photosphere(
                        wave, flux, self.theta[0], self.theta[1], self.theta[2])
                ext = self.av_law(wave * 1e4, Av, Rv)
                flux = apply(ext, flux) * wave * norm
                ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)
            else:
                logger.warning('no spectra cache for grid %r; photosphere '
                               'curve skipped', self.grid)
                wave, flux = np.array([]), np.array([])
        if self.save:
            data = np.vstack((wave, flux)).T
            np.savetxt(f'{self.out_folder}/SED.dat', data, fmt='%s',
                       header='wavelength(mu m) wave*flux(erg cm-2 s-2)')
        pass

    def plot_trace(self):
        """Plot SED chains."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                f, ax = plt.subplots(figsize=(12, 4))
                ax.step(range(len(samples[param])), samples[param],
                        color='k', alpha=0.8)
                ax.set_ylabel(param,
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.set_xlabel('Steps',
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                best = self.out['best_fit'][param]
                # ax.axhline(np.median(samples[param]), color='red', lw=2)
                ax.axhline(best, color='red', lw=2)
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig(self.chain_out + '/' + param +
                            '.png', bbox_inches='tight')
        plt.close('all')
        pass

    def plot_like(self):
        """Plot Likelihoods."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                f, ax = plt.subplots(figsize=(12, 4))
                ax.scatter(samples[param], samples['loglike'], alpha=0.5, s=40)
                best = self.out['best_fit'][param]
                # ax.axvline(np.median(samples[param]), color='red', lw=1.5)
                ax.axvline(best, color='red', lw=1.5)
                ax.set_ylabel('log likelihood',
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.set_xlabel(param,
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig(self.like_out + '/' + param + '.png',
                            bbox_inches='tight')
        plt.close('all')
        if self.engine == 'dynesty':
            fig, axes = dyplot.traceplot(
                self.out['dynesty'],
                truths=self.theta,
                show_titles=True, trace_cmap='plasma',
            )
            plt.savefig(self.like_out + '/dynesty_trace.png')
        pass

    def plot_post(self):
        """Plot posteriors."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                f, ax = plt.subplots(figsize=(12, 4))
                ax.scatter(samples[param], samples['posteriors'], alpha=0.5,
                           s=40)
                best = self.out['best_fit'][param]
                # ax.axvline(np.median(samples[param]), color='red', lw=1.5)
                ax.axvline(best, color='red', lw=1.5)
                ax.set_ylabel('log posterior',
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.set_xlabel(param,
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig(self.post_out + '/' + param + '.png',
                            bbox_inches='tight')
        plt.close('all')
        pass

    def plot_hist(self):
        pass

    def plot_bma_hist(self):
        """Plot histograms."""
        print('Plotting BMA histograms.')
        models = [key for key in self.out['originals'].keys()]
        # One color per ensemble member; tab:cyan and tab:pink are reserved
        # for the weighted-sampling / weighted-average curves below.
        palette = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:gray', 'tab:olive', 'gold', 'teal',
            'limegreen', 'crimson', 'navy', 'darkviolet', 'magenta'
        ]
        colors = [palette[j % len(palette)] for j in range(len(models))]
        for i, param in enumerate(self.order):
            if 'noise' in param:
                continue
            if not self.coordinator[i]:
                f1, ax1 = plt.subplots(figsize=(12, 6))
                f2, ax2 = plt.subplots(figsize=(12, 6))
                for j, m in enumerate(models):
                    # Get samples
                    samp = self.out['originals'][m][param]
                    # Plot sample histogram
                    label = m + f" prob: {self.out['weights'][m]:.3f}"
                    # Normal
                    n, bins1, patches = ax1.hist(samp, alpha=.3, bins=20,
                                                 label=label, density=True,
                                                 color=colors[j])
                    # Weighted
                    n, bins2, patches = ax2.hist(
                        samp, alpha=.3, bins=20, label=label,
                        weights=[self.out['weights'][m]] * len(samp)
                    )
                    # Fit a KDE to data
                    kde = gaussian_kde(samp)
                    # Estimate amplitude of the weighted distributions
                    mu, sig = norm.fit(samp)
                    try:
                        bc = bins2[:-1] + np.diff(bins2)
                        popt, pcov = curve_fit(norm_fit, xdata=bc, ydata=n,
                                               p0=[mu, sig, n.max()],
                                               maxfev=50000)
                    except RuntimeError:
                        popt = (mu, sig, n.max())
                    xx1 = np.linspace(bins1[0], bins1[-1], 1000)
                    xx2 = np.linspace(bins2[0], bins2[-1], 1000)
                    # Plot best fit
                    ax1.plot(xx1, kde(xx2), lw=2, alpha=1, color=colors[j])
                    ax2.plot(xx1, kde(xx2) * popt[2], lw=2, alpha=1,
                             color=colors[j])
                # The same but for the weighted samples
                n, bins, patches = ax1.hist(
                    self.out['weighted_samples'][param], alpha=.3,
                    bins=20, label='Weighted sampling', density=True,
                    color='tab:cyan'
                )
                kde = gaussian_kde(self.out['weighted_samples'][param])
                xx = np.linspace(bins[0], bins[-1], 300)
                ax1.plot(xx, kde(xx), color='tab:cyan', lw=2, alpha=1, ls='--')

                # Now ditto for the weighted average
                n, bins, patches = ax1.hist(
                    self.out['weighted_average'][param], alpha=.3,
                    bins=20, label='Weighted average', density=True,
                    color='tab:pink'
                )
                kde = gaussian_kde(self.out['weighted_average'][param])
                xx = np.linspace(bins[0], bins[-1], 300)
                ax1.plot(xx, kde(xx), color='tab:pink', lw=2, alpha=1, ls='-.')
                lab = 'fix this'
                if param == 'teff':
                    lab = 'Teff (K)'
                elif param == 'rad':
                    lab = r'R$_*$ (R$_\odot$)'
                elif param == 'dist':
                    lab = 'D (pc)'
                elif param == 'logg':
                    lab = 'Log g'
                elif param == 'Av':
                    lab = r'A$_{\rm V}$ (mag)'
                elif param == 'z':
                    lab = '[Fe/H]'
                elif param == 'Age':
                    lab = 'Age (Gyr)'
                elif param == 'Mass':
                    lab = r'Mass (M$_\odot$)'
                # Normal
                ax1.set_ylabel('PDF',
                               fontsize=self.fontsize,
                               fontname=self.fontname
                               )
                # Weighted
                ax2.set_ylabel('N',
                               fontsize=self.fontsize,
                               fontname=self.fontname
                               )
                axes = [ax1, ax2]
                # Zero-weight grids pegged at a grid edge run away to absurd
                # parameter values and stretch the axes; frame the plot on the
                # members that actually carry weight (plus the weighted
                # posteriors).
                wmodels = [m for m in models
                           if self.out['weights'][m] > 1e-3]
                if wmodels:
                    lo = min(np.percentile(self.out['originals'][m][param],
                                           0.1) for m in wmodels)
                    hi = max(np.percentile(self.out['originals'][m][param],
                                           99.9) for m in wmodels)
                    for ws in ('weighted_samples', 'weighted_average'):
                        lo = min(lo, np.percentile(self.out[ws][param], 0.1))
                        hi = max(hi, np.percentile(self.out[ws][param], 99.9))
                    if hi > lo:
                        pad = 0.05 * (hi - lo)
                        for ax in axes:
                            ax.set_xlim(lo - pad, hi + pad)
                for ax in axes:
                    ax.set_xlabel(lab,
                                  fontsize=self.fontsize,
                                  fontname=self.fontname
                                  )
                    for tick in ax.get_yticklabels():
                        tick.set_fontname(self.fontname)
                    for tick in ax.get_xticklabels():
                        tick.set_fontname(self.fontname)

                    ax.tick_params(
                        axis='both', which='major',
                        labelsize=self.tick_labelsize
                    )
                    ax.legend(loc=0, prop={'size': 16})

                if param == 'z':
                    param = 'Fe_H'
                if self.png:
                    f1.savefig(self.hist_out + '/' + param + '.png',
                               bbox_inches='tight')
                    f2.savefig(self.hist_out + '/weighted_' + param + '.png',
                               bbox_inches='tight')
                if self.pdf:
                    f1.savefig(self.hist_out + '/' + param + '.pdf',
                               bbox_inches='tight')
                    f2.savefig(self.hist_out + '/weighted_' + param + '.pdf',
                               bbox_inches='tight')
                plt.close(f1)
                plt.close(f2)

        if self.bma:
            mist_samp = self.out.get('mist_samples', {})
            # Age hist
            if 'age' in mist_samp:
                f, ax = plt.subplots(figsize=(12, 4))
                samp = mist_samp['age']
                n, bins, patches = ax.hist(
                    samp, alpha=.3, bins=20, label='MIST', density=True
                )
                kde = gaussian_kde(samp)
                xx = np.linspace(bins[0], bins[-1], 10000)
                ax.plot(xx, kde(xx), color='k', lw=2, alpha=.7)
                ax.set_ylabel('PDF',
                              fontsize=self.fontsize,
                              fontname=self.fontname)
                ax.set_xlabel('Age',
                              fontsize=self.fontsize,
                              fontname=self.fontname)

                for tick in ax.get_yticklabels():
                    tick.set_fontname(self.fontname)
                for tick in ax.get_xticklabels():
                    tick.set_fontname(self.fontname)

                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.legend(loc=0)
                if self.png:
                    plt.savefig(self.hist_out + '/age.png',
                                bbox_inches='tight')
                if self.pdf:
                    plt.savefig(self.hist_out + '/age.pdf',
                                bbox_inches='tight')
                plt.close(f)
            # Mass hist
            if 'iso_mass' in mist_samp:
                f, ax = plt.subplots(figsize=(12, 4))
                samp = mist_samp['iso_mass']
                n, bins, patches = ax.hist(
                    samp, alpha=.3, bins=20, label='MIST', density=True
                )
                kde = gaussian_kde(samp)
                xx = np.linspace(bins[0], bins[-1], 10000)
                ax.plot(xx, kde(xx), color='k', lw=2, alpha=.7)
                ax.set_ylabel('PDF',
                              fontsize=self.fontsize,
                              fontname=self.fontname)
                ax.set_xlabel('Mass',
                              fontsize=self.fontsize,
                              fontname=self.fontname)

                for tick in ax.get_yticklabels():
                    tick.set_fontname(self.fontname)
                for tick in ax.get_xticklabels():
                    tick.set_fontname(self.fontname)

                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.legend(loc=0)
                if self.png:
                    plt.savefig(self.hist_out + '/iso_mass.png',
                                bbox_inches='tight')
                if self.pdf:
                    plt.savefig(self.hist_out + '/iso_mass.pdf',
                                bbox_inches='tight')
                plt.close(f)
            # EEP hist
            if 'eep' in mist_samp:
                f, ax = plt.subplots(figsize=(12, 4))
                samp = mist_samp['eep']
                n, bins, patches = ax.hist(
                    samp, alpha=.3, bins=20, label='MIST', density=True
                )
                kde = gaussian_kde(samp)
                xx = np.linspace(bins[0], bins[-1], 10000)
                ax.plot(xx, kde(xx), color='k', lw=2, alpha=.7)
                ax.set_ylabel('PDF',
                              fontsize=self.fontsize,
                              fontname=self.fontname)
                ax.set_xlabel('EEP',
                              fontsize=self.fontsize,
                              fontname=self.fontname)

                for tick in ax.get_yticklabels():
                    tick.set_fontname(self.fontname)
                for tick in ax.get_xticklabels():
                    tick.set_fontname(self.fontname)

                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.legend(loc=0)
                if self.png:
                    plt.savefig(self.hist_out + '/EEP.png',
                                bbox_inches='tight')
                if self.pdf:
                    plt.savefig(self.hist_out + '/EEP.pdf',
                                bbox_inches='tight')
                plt.close(f)

    def plot_bma_HR(self, nsamp):
        """Plot HR diagram for the star."""
        bf_key = f'best_fit_{self.method}'
        if 'age' not in self.out.get(bf_key, {}):
            return
        print('Plotting HR diagram')
        # Get necessary info from the star.
        age = self.out[bf_key]['age']
        feh = self.out[f'best_fit_{self.method}']['z']
        teff = np.log10(self.out[f'best_fit_{self.method}']['teff'])
        lum = np.log10(self.out[f'best_fit_{self.method}']['lum'])
        teff_lo, teff_hi = self.out[f'uncertainties_{self.method}']['teff']
        lum_lo, lum_hi = self.out[f'uncertainties_{self.method}']['lum']
        teff_lo = teff_lo / (10 ** teff * np.log(10))
        teff_hi = teff_hi / (10 ** teff * np.log(10))
        lum_lo = lum_lo / (10 ** lum * np.log(10))
        lum_hi = lum_hi / (10 ** lum * np.log(10))
        ages = self.out['mist_samples']['age']
        m = self.method if self.method == 'samples' else 'average'
        fehs = self.out[f'weighted_{m}']['z']

        if feh > 0.5:
            feh = 0.5

        iso_bf = get_isochrone(np.log10(age) + 9, feh)

        logteff = iso_bf['logTeff'].values
        loglum = iso_bf['logL'].values
        mass = iso_bf['mass'].values

        fig, ax = plt.subplots(figsize=self.hr_figsize)

        points = np.array([logteff, loglum]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(mass.min(), mass.max())
        lc = LineCollection(segments, cmap=self.hr_cmap, norm=norm,
                            linewidths=5)

        lc.set_array(mass)
        line = ax.add_collection(lc)
        line.zorder = 1000
        cbar = fig.colorbar(line, ax=ax, pad=0.01)
        cbar.set_label(r'$M_\odot$',
                       rotation=270,
                       fontsize=self.fontsize,
                       fontname=self.fontname,
                       labelpad=20)

        for i in range(nsamp):
            a = np.log10(choice(ages)) + 9
            z = choice(fehs)
            if z > 0.5:
                z = 0.5
            iso = get_isochrone(a, z)

            logt = iso['logTeff'].values
            logl = iso['logL'].values
            ax.plot(logt, logl, color='gray')

        ax.errorbar(teff, lum, xerr=[[teff_lo], [teff_hi]],
                    yerr=[[lum_lo], [lum_hi]], color=self.hr_color,
                    zorder=1001)
        ax.scatter(teff, lum, s=350, color=self.hr_color, zorder=1002,
                   edgecolors='k', marker=self.hr_marker)

        ax.invert_xaxis()
        ax.set_xlabel('logTeff',
                      fontsize=self.fontsize,
                      fontname=self.fontname)
        ax.set_ylabel('logL',
                      fontsize=self.fontsize,
                      fontname=self.fontname)
        ax.tick_params(
            axis='both', which='major',
            labelsize=self.tick_labelsize
        )
        for ll in cbar.ax.yaxis.get_ticklabels():
            ll.set_fontsize(self.tick_labelsize)
        for tick in ax.get_yticklabels():
            tick.set_fontname(self.fontname)
        for tick in ax.get_yticklabels():
            tick.set_fontname(self.fontname)

        if self.png:
            plt.savefig(self.out_folder + '/HR_diagram.png',
                        bbox_inches='tight')
        if self.pdf:
            plt.savefig(self.out_folder + '/HR_diagram.pdf',
                        bbox_inches='tight')

    def plot_corner(self):
        """Make corner plot."""
        print('Plotting corner.')
        m = self.method if self.method == 'samples' else 'average'
        samples = self.out[f'weighted_{m}']
        all_samps = []
        theta_lo = []
        theta_up = []

        for i, o in enumerate(self.order):
            if 'noise' in o:
                self.coordinator[i] = 1

        theta = self.theta[self.coordinator == 0]
        used_params = self.order[self.coordinator == 0]

        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                if 'noise' in param:
                    continue
                _, lo, up = credibility_interval(samples[param])
                theta_lo.append(lo)
                theta_up.append(up)
                all_samps.append(samples[param])

        corner_samp = np.vstack(all_samps)

        titles = self.__create_titles(used_params, theta, theta_up, theta_lo)
        labels = self.__create_labels(used_params)

        fig = corner.corner(
            corner_samp.T,
            plot_contours=True,
            fill_contours=False,
            plot_datapoints=True,
            no_fill_contours=True,
            max_n_ticks=4
        )

        axes = np.array(fig.axes).reshape((theta.shape[0], theta.shape[0]))

        for i in range(theta.shape[0]):
            ax = axes[i, i]
            ax.axvline(theta[i], color=self.corner_med_c,
                       linestyle=self.corner_med_style)
            ax.axvline(theta_lo[i], color=self.corner_v_c,
                       linestyle=self.corner_v_style)
            ax.axvline(theta_up[i], color=self.corner_v_c,
                       linestyle=self.corner_v_style)
            t = titles[i]

            ax.set_title(t, fontsize=self.corner_fontsize,
                         fontname=self.fontname)

        for yi in range(theta.shape[0]):
            for xi in range(yi):
                ax = axes[yi, xi]
                if xi == 0:
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label1.set_fontsize(self.corner_tick_fontsize)
                        tick.label1.set_fontname(self.fontname)
                        ax.set_ylabel(
                            labels[yi],
                            labelpad=self.corner_labelpad,
                            fontsize=self.corner_fontsize,
                            fontname=self.fontname
                        )
                if yi == theta.shape[0] - 1:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label1.set_fontsize(self.corner_tick_fontsize)
                        tick.label1.set_fontname(self.fontname)
                        ax.set_xlabel(
                            labels[xi],
                            labelpad=self.corner_labelpad,
                            fontsize=self.corner_fontsize,
                            fontname=self.fontname
                        )
                ax.axvline(theta[xi], color=self.corner_med_c,
                           linestyle=self.corner_med_style)
                ax.axhline(theta[yi], color=self.corner_med_c,
                           linestyle=self.corner_med_style)
                ax.plot(theta[xi], theta[yi], self.corner_marker)
            axes[-1, -1].set_xlabel(
                labels[-1],
                labelpad=self.corner_labelpad,
                fontsize=self.corner_fontsize,
                fontname=self.fontname
            )
            for tick in axes[-1, -1].xaxis.get_major_ticks():
                tick.label1.set_fontsize(self.corner_tick_fontsize)
                tick.label1.set_fontname(self.fontname)

            if self.pdf:
                plt.savefig(f'{self.out_folder}/CORNER.pdf',
                            bbox_inches='tight')
            if self.png:
                plt.savefig(f'{self.out_folder}/CORNER.png',
                            bbox_inches='tight')
        pass

    def clean(self):
        """Close opened figures."""
        plt.close('all')

    def fetch_Phoenix(self):
        """Fetch correct Phoenixv2 SED file.

        The directory containing the Phoenix spectra must be called PHOENIXv2
        Within PHOENIXv2 there should be the wavelength file called
        WAVE_PHOENIX-ACES-AGSS-COND-2011.fits and several folders called
        Z[-/+]X.X where X.X are the metallicities (e.g. Z-0.0, Z+1.0, etc)
        """
        # Change hdd to a class variable depending on an env param.
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        selected_SED = self.moddir + 'PHOENIXv2/Z'
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED += metal_add
        selected_SED += '/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 5 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + '0'
        selected_SED += metal_add
        selected_SED += '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        flux = fits.open(selected_SED)[0].data
        flux *= (u.erg / u.s / u.cm ** 2 / u.cm).to(
            u.erg / u.s / u.cm ** 2 / u.um)
        return flux

    def fetch_btsettl(self):
        """Fetch correct BT-Settl SED file.

        The directory containing the BT-Settl spectra must be called BTSettl
        Within BTSettl there should be yet another directory
        called AGSS2009, within BTSettl/AGSS2009 there should be the SED fits
        files with the following naming convention:

        lteTTT-G.G[-/+]Z.Za+0.0.BT-Settl.AGSS2009.fits

        where TTT are the first 3 digits of the effective temperature if it's a
        number over 10000, else it's the first 2 digit prepended by a 0.
        G.G is the log g and Z.Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff]) // 100
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.moddir + 'BTSettl/AGSS2009/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+*'
        gl = glob.glob(selected_SED)
        selected_SED = gl[0]
        tab = Table(fits.open(selected_SED)[1].data)
        flux = np.array(tab['FLUX'].tolist()) * conversion
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        return wave, flux

    def fetch_btnextgen(self):
        """Fetch correct BT-NextGen SED file.

        The directory containing the BT-NextGen spectra must be called
        BTNextGen. Within BTNextGen there should be yet another directory
        called AGSS2009, within BTNextGen/AGSS2009 there should be the SED fits
        files with the following naming convention:

        lteTTT-G.G[-/+]Z.Za+0.0..BT-NextGen.AGSS2009.fits

        where TTT are the first 3 digits of the effective temperature if it's a
        number over 10000, else it's the first 2 digit prepended by a 0.
        G.G is the log g and Z.Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff]) // 100
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.moddir + 'BTNextGen/AGSS2009/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+*'
        gl = glob.glob(selected_SED)
        selected_SED = gl[0]
        tab = Table(fits.open(selected_SED)[1].data)
        flux = np.array(tab['FLUX'].tolist()) * conversion
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        return wave, flux

    def fetch_btcond(self):
        """Fetch correct BT-COND SED file.

        The directory containing the BT-COND spectra must be called
        BTCOND. Within BTCOND there should be yet another directory
        called CIFIST2011, within BTCOND/CIFIST2011 there should be the SED
        fits files with the following naming convention:

        lteTTT-G.G[-/+]Z.Za+0.0..BT-Cond.CIFIST2011.fits

        where TTT are the first 3 digits of the effective temperature if it's a
        number over 10000, else it's the first 2 digit prepended by a 0.
        G.G is the log g and Z.Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff]) // 100
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.moddir + 'BTCond/CIFIST2011/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+*'
        gl = glob.glob(selected_SED)
        selected_SED = gl[0]
        tab = Table(fits.open(selected_SED)[1].data)
        flux = np.array(tab['FLUX'].tolist()) * conversion
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        return wave, flux

    def fetch_ck04(self):
        """Fetch correct Castelli-Kurucz 2004 SED file.

        The directory containing the Castelli-Kurucz spectra must be called
        Castelli_Kurucz. Within Castelli_Kurucz there should be a group of
        directories called ck[pm]ZZ where ZZ is the metalicity without the dot.
        Within each directory there are fits files named:

        ck[pm]ZZ_TTTT.fits

        where ZZ is metalicity as previous and TTTT is the effective
        temperature.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = 'm' + str(-sel_z).replace('.', '')
        if sel_z == 0:
            metal_add = 'p00'
        if sel_z > 0:
            metal_add = 'p' + str(sel_z).replace('.', '')
        name = 'ck' + metal_add
        lgg = f'g{sel_logg * 10:02.0f}'
        selected_SED = self.moddir + 'Castelli_Kurucz/' + name + '/' + name
        selected_SED += '_' + str(sel_teff) + '.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        flux = np.array(tab[lgg].tolist()) * conversion
        return wave, flux

    def fetch_kurucz(self):
        """Fetch correct Kurucz 1993 SED file.

        The directory containing the Kurucz spectra must be called
        Kurucz. Within Kurucz there should be a group of
        directories called k[pm]ZZ where ZZ is the metalicity without the dot.
        Within each directory there are fits files named:

        k[pm]ZZ_TTTT.fits

        where ZZ is metalicity as previous and TTTT is the effective
        temperature.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = 'm' + str(-sel_z).replace('.', '')
        if sel_z == 0:
            metal_add = 'p00'
        if sel_z > 0:
            metal_add = 'p' + str(sel_z).replace('.', '')
        name = 'k' + metal_add
        lgg = f'g{sel_logg * 10:02.0f}'
        selected_SED = self.moddir + 'Kurucz/' + name + '/' + name
        selected_SED += '_' + str(sel_teff) + '.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        flux = np.array(tab[lgg].tolist()) * conversion
        return wave, flux

    def fetch_coelho(self):
        """Fetch correct Coelho 2014 SED file.

        The directory containing the Coelho spectra must be called
        Coelho14. Within Coelho14 there should be a group of
        files called t[0X]XXXX_g[+-]Y.Y_[mp]ZZp0[14]_sed.fits
        where X is the temperature, Y is the logg and Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        sel_teff = str(sel_teff) if sel_teff >= 1e5 else f'0{sel_teff}'
        selected_SED = self.moddir + 'Coelho14/t' + sel_teff + '_g'
        sel_logg = f'+{sel_logg:.1f}' if sel_logg > 0 else '-0.5'
        selected_SED += sel_logg
        if sel_z < 0:
            selected_SED += f'_m{-sel_z * 10:02.0f}'
        else:
            selected_SED += f'_p{sel_z * 10:02.0f}'

        selected_SED = glob.glob(selected_SED + 'p0[04]_sed.fits')
        selected_SED = selected_SED[0]
        hdul = fits.open(selected_SED)
        head = hdul[0].header
        data = hdul[0].data
        flux = data * conversion
        CRVAL1 = head['CRVAL1']
        CDEL1 = head['CDELT1']
        wave = 10 ** np.array(
            [CRVAL1 + CDEL1 * i for i in range(data.shape[0])])
        wave *= u.angstrom.to(u.um)
        return wave, flux

    def __create_titles(self, titles, theta, theta_up, theta_lo):
        new_titles = np.empty(titles.shape[0], dtype=object)
        for i, param in enumerate(titles):
            if param == 'teff':
                new_titles[i] = r'Teff ='
            if param == 'logg':
                new_titles[i] = r'    Log g ='
            if param == 'z':
                new_titles[i] = r'        [Fe/H] ='
            if param == 'dist':
                new_titles[i] = r'    D ='
            if param == 'rad':
                new_titles[i] = r'R ='
            if param == 'norm':
                new_titles[i] = r'    (R/D)$^2$ ='
            if param == 'Av':
                new_titles[i] = r'Av ='
            if param == 'inflation':
                new_titles[i] = r'$\sigma$ ='
            if param == 'rad' or param == 'dist':
                new_titles[i] += f'{theta[i]:.3f}'
                new_titles[i] += r'$^{+' + \
                                 f'{theta_up[i] - theta[i]:.3f}'
                new_titles[i] += r'}_{-' + \
                                 f'{theta[i] - theta_lo[i]:.3f}'
                new_titles[i] += r'}$'
            else:
                new_titles[i] += f'{theta[i]:.2f}'
                new_titles[i] += r'$^{+' + \
                                 f'{theta_up[i] - theta[i]:.2f}'
                new_titles[i] += r'}_{-' + \
                                 f'{theta[i] - theta_lo[i]:.2f}'
                new_titles[i] += r'}$'
        return new_titles

    def __create_labels(self, labels):
        new_labels = np.empty(labels.shape[0], dtype=object)
        for i, param in enumerate(labels):
            if param == 'teff':
                new_labels[i] = r'Teff (K)'
            if param == 'logg':
                new_labels[i] = r'Log g'
            if param == 'z':
                new_labels[i] = r'[Fe/H]'
            if param == 'dist':
                new_labels[i] = r'D (pc)'
            if param == 'rad':
                new_labels[i] = r'R $($R$_\odot)$'
            if param == 'norm':
                new_labels[i] = r'(R/D)'
            if param == 'Av':
                new_labels[i] = r'Av'
            if param == 'inflation':
                new_labels[i] = r'$\sigma$'
        return new_labels

    def __read_config(self):
        """Read plotter configuration file."""
        # ARIADNE renders math with mathtext, never LaTeX. Some third-party
        # imports flip text.usetex globally mid-process, which crashes every
        # savefig on machines without a TeX install -- pin it off.
        import matplotlib
        matplotlib.rcParams['text.usetex'] = False
        if self.settings_dir is None:
            settings = open(filesdir + '/plot_settings.dat', 'r')
        else:
            settings = open(self.settings_dir, 'r')
        for line in settings.readlines():
            if line[0] == '#' or line[0] == '\n':
                continue
            splt = line.split(' ')
            attr = splt[0]
            if attr == 'figsize':
                vals = splt[1].split('\n')[0].split(',')
                val = (int(vals[0]), int(vals[1]))
            elif attr == 'hr_figsize':
                vals = splt[1].split('\n')[0].split(',')
                val = (int(vals[0]), int(vals[1]))
            elif 'alpha' in attr:
                val = float(splt[1].split('\n')[0])
            else:
                try:
                    val = int(splt[1].split('\n')[0])
                except ValueError:
                    val = splt[1].split('\n')[0]
            setattr(self, attr, val)
