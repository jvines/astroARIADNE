"""plot_utils module for plotting SEDs."""

__all__ = ['SEDPlotter']

import glob
import pickle
from random import choice

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from extinction import apply
from isochrones.interp import DFInterpolator
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from PyAstronomy import pyasl
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm

import corner
from dynesty import plotting as dyplot

from .config import filesdir, gridsdir, modelsdir
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
        if self.irx:
            model = 'btsettl'

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
            for filt, flx, flx_e in zip(filters, flxs, errs):
                p_ = get_noise_name(filt) + '_noise'
                self.order = np.append(self.order, p_)

            if self.grid.lower() == 'phoenix':
                with open(gridsdir + '/Phoenixv2_DF.pkl', 'rb') as intp:
                    self.interpolator = DFInterpolator(pd.read_pickle(intp))
            if self.grid.lower() == 'btsettl':
                with open(gridsdir + '/BTSettl_DF.pkl', 'rb') as intp:
                    self.interpolator = DFInterpolator(pd.read_pickle(intp))
            if self.grid.lower() == 'btnextgen':
                with open(gridsdir + '/BTNextGen_DF.pkl', 'rb') as intp:
                    self.interpolator = DFInterpolator(pd.read_pickle(intp))
            if self.grid.lower() == 'btcond':
                with open(gridsdir + '/BTCond_DF.pkl', 'rb') as intp:
                    self.interpolator = DFInterpolator(pd.read_pickle(intp))
            if self.grid.lower() == 'ck04':
                with open(gridsdir + '/CK04_DF.pkl', 'rb') as intp:
                    self.interpolator = DFInterpolator(pd.read_pickle(intp))
            if self.grid.lower() == 'kurucz':
                with open(gridsdir + '/Kurucz_DF.pkl', 'rb') as intp:
                    self.interpolator = DFInterpolator(pd.read_pickle(intp))
            if self.grid.lower() == 'coelho':
                with open(gridsdir + '/Coelho_DF.pkl', 'rb') as intp:
                    self.interpolator = DFInterpolator(pd.read_pickle(intp))

            # Get best fit parameters.
            theta_samples = np.zeros(self.order.shape[0])
            theta_average = np.zeros(self.order.shape[0])
            for i, param in enumerate(self.order):
                if param != 'inflation':
                    theta_samples[i] = out['best_fit_samples'][param]
                    theta_average[i] = out['best_fit_averaged'][param]

            # Calculate best fit model.
            model_samples = model_grid(theta_samples, filters, wave,
                                       self.interpolator, self.norm,
                                       self.av_law)
            model_average = model_grid(theta_average, filters, wave,
                                       self.interpolator, self.norm,
                                       self.av_law)
            if self.irx:
                irx_model_sam = model_grid(theta_samples, irx_filt, irx_wave,
                                           self.interpolator, self.norm,
                                           self.av_law)
                irx_model_avg = model_grid(theta_average, irx_filt, irx_wave,
                                           self.interpolator, self.norm,
                                           self.av_law)
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
        colors = np.array([
            'tomato', 'indianred', 'tab:red',
            'salmon', 'coral',
            'mediumorchid', 'mediumslateblue', 'tab:blue',
            'darkslateblue', 'darkblue',
            'olivedrab', 'yellowgreen', 'greenyellow', 'yellow',
            'orangered', 'chocolate', 'khaki',
            'limegreen', 'darkgreen', 'lime', 'seagreen', 'lawngreen', 'green',
            'aquamarine', 'turquoise', 'lightseagreen', 'teal', 'cadetblue',
            'tab:pink', 'tab:purple', 'tab:cyan', 'aqua', 'azure', 'dodgerblue',
            'firebrick', 'darkred',
            'blueviolet', 'darkviolet',
            'midnightblue', 'blue',
            'deeppink', 'fuchsia', 'mediumslateblue'
        ])

        for c, w, fl, fe, bp, fi in zip(
                colors[self.star.filter_mask],
                self.wave, self.flux, self.flux_er,
                self.bandpass, used_f):
            ax.errorbar(w, fl * w,
                        xerr=bp, yerr=fe * w,
                        fmt='',
                        ecolor=c,
                        marker=None)

            ax.scatter(w, fl * w,
                       edgecolors='black',
                       marker=self.marker,
                       c=c,
                       s=self.scatter_size,
                       alpha=self.scatter_alpha, label=fi)

        ax.set_ylim([ymin * .8, ymax * 1.25])
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
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
        ax.set_xticks(np.linspace(1, 10, 10))
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlim([0.1, 6])

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
        """Create the plot of the SED."""
        if self.moddir is None:
            print('Models directory not provided, skipping SED plot.')
            return
        print('Plotting SED')
        # Get plot ylims.
        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        if self.irx:
            ymin = (self.irx_model * self.irx_wave).min()

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
            self.av_law)

        norm_res = residuals / errors

        # Create plot layout

        f = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 0.5], hspace=0.05)

        ax = f.add_subplot(gs[0])
        ax_r = f.add_subplot(gs[1])

        self.SED(ax)

        # Model plot
        ax.errorbar(self.wave, self.flux * self.wave,
                    xerr=self.bandpass, yerr=errors,
                    fmt=',', ecolor=self.error_color, zorder=0,
                    marker=None)
        ax.scatter(self.wave, self.flux * self.wave,
                   edgecolors='black', marker=self.marker, c=self.marker_colors,
                   s=self.scatter_size, zorder=1, alpha=self.scatter_alpha)
        ax.scatter(self.wave, self.model * self.wave,
                   marker=self.marker_model,
                   edgecolors=self.marker_colors_model, s=self.scatter_size,
                   facecolor='none', zorder=3, lw=3)

        if self.irx:
            ax.errorbar(self.irx_wave, self.irx_flux * self.irx_wave,
                        xerr=self.irx_bandpass, yerr=self.irx_flux_er,
                        fmt=',', ecolor=self.irx_error_color, zorder=0,
                        marker=None)
            ax.scatter(self.irx_wave, self.irx_flux * self.irx_wave,
                       edgecolors='black', marker=self.marker,
                       c=self.marker_colors_irx, s=self.scatter_size, zorder=1,
                       alpha=self.scatter_alpha)
            ax.scatter(self.irx_wave, self.irx_model * self.irx_wave,
                       marker=self.marker_model,
                       edgecolors=self.marker_colors_model_irx,
                       s=self.scatter_size, facecolor='none', zorder=3, lw=3)

        # Residual plot
        ax_r.axhline(y=0, lw=2, ls='--', c='k', alpha=.7)

        ax_r.errorbar(self.wave, np.zeros(self.wave.shape[0]),
                      xerr=self.bandpass, yerr=self.flux_er,
                      fmt=',', ecolor=self.error_color, marker=None)
        ax_r.scatter(self.wave, np.zeros(self.wave.shape[0]),
                     edgecolors='black', marker=self.marker,
                     c=self.marker_colors, s=self.scatter_size,
                     alpha=self.scatter_alpha)
        ax_r.scatter(self.wave, norm_res,
                     marker=self.marker_model,
                     edgecolors=self.marker_colors_model, s=self.scatter_size,
                     facecolor='none', lw=3, zorder=10)

        if self.irx:
            irx_res = (self.irx_flux - self.irx_model) / self.irx_flux_er
            ax_r.errorbar(self.irx_wave, np.zeros(self.irx_wave.shape[0]),
                          xerr=self.irx_bandpass, yerr=self.irx_flux_er,
                          fmt=',', ecolor=self.irx_error_color, marker=None)
            ax_r.scatter(self.irx_wave, np.zeros(self.irx_wave.shape[0]),
                         edgecolors='black', marker=self.marker,
                         c=self.marker_colors_irx, s=self.scatter_size,
                         alpha=self.scatter_alpha)
            ax_r.scatter(self.irx_wave, irx_res,
                         marker=self.marker_model,
                         edgecolors=self.marker_colors_model_irx,
                         s=self.scatter_size, facecolor='none', lw=3, zorder=10)

        # Formatting
        res_std = norm_res.std()
        ax.set_ylim([ymin * 0.6, ymax * 1.5])
        # ax_r.set_ylim([-5, 5])
        ax_r.set_ylim([-5 * res_std, 5 * res_std])
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax_r.set_xscale('log', nonposx='clip')
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
        if self.irx:
            xticks = [1, 3, 5, 10, 20, 50, 100, 250]
        ax_r.set_xticks(xticks)
        ax_r.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(ticker.NullFormatter())
        ylocmin = ticker.LinearLocator(numticks=4)

        ax_r.yaxis.set_minor_locator(ylocmin)
        ax_r.yaxis.set_minor_formatter(ticker.NullFormatter())

        xlims1 = [0.125, 6]
        xlims2 = [0.25, 6]
        if self.irx:
            xlims1 = [0.125, 250]
            xlims2 = [0.25, 250]

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
            plt.savefig(f'{self.out_folder}/SED.pdf', bbox_inches='tight')
        if self.png:
            plt.savefig(f'{self.out_folder}/SED.png', bbox_inches='tight')
        if self.save:
            data = np.vstack((self.wave, self.model * self.wave)).T
            np.savetxt(f'{self.out_folder}/synthetic.dat', data, fmt='%s',
                       header='wavelength(mu m) wave*flux(erg cm-2 s-2)')
        pass

    def SED(self, ax):
        """Plot the SED model."""
        Rv = 3.1  # For extinction.
        if not self.norm:
            rad = self.theta[4]
            dist = self.theta[3] * u.pc.to(u.solRad)
            norm = (rad / dist) ** 2
            Av = self.theta[5]
        else:
            norm = self.theta[3]
            Av = self.theta[4]

        # SED plot.
        if self.grid == 'phoenix':
            wave = fits.open(self.moddir + self.__wav_file)[0].data
            wave *= u.angstrom.to(u.um)

            lower_lim = 0.125 < wave
            upper_lim = wave < 4.629296073126975
            if self.irx:
                upper_lim = wave < wave[-1]

            flux = self.fetch_Phoenix()

            new_w = wave[lower_lim * upper_lim]

            new_ww = np.linspace(new_w[0], new_w[-1], len(new_w))

            ext = self.av_law(new_w * 1e4, Av, Rv)

            brf, _ = pyasl.instrBroadGaussFast(
                new_ww, flux, 1500,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            brf = brf[lower_lim * upper_lim]
            brf = apply(ext, brf)
            flx = brf * norm * new_w
            ax.plot(new_w[:-1000], flx[:-1000], lw=1.25, color=self.model_color,
                    zorder=0)

        elif self.grid == 'btsettl':
            wave, flux = self.fetch_btsettl()

            lower_lim = 0.125 < wave
            upper_lim = wave < 4.629296073126975
            if self.irx:
                upper_lim = wave < wave[-1]

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)

            new_w = np.linspace(wave[0], wave[-1], len(wave))

            brf, _ = pyasl.instrBroadGaussFast(
                new_w, flux, 1500,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'btnextgen':
            wave, flux = self.fetch_btnextgen()

            lower_lim = 0.125 < wave
            upper_lim = wave < 4.629296073126975
            if self.irx:
                upper_lim = wave < wave[-1]

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)

            new_w = np.linspace(wave[0], wave[-1], len(wave))

            brf, _ = pyasl.instrBroadGaussFast(
                new_w, flux, 1500,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'btcond':
            wave, flux = self.fetch_btcond()

            lower_lim = 0.125 < wave
            upper_lim = wave < 4.629296073126975
            if self.irx:
                upper_lim = wave < wave[-1]

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)

            new_w = np.linspace(wave[0], wave[-1], len(wave))

            brf, _ = pyasl.instrBroadGaussFast(
                new_w, flux, 1500,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'ck04':
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
            wave, flux = self.fetch_coelho()

            lower_lim = 0.15 < wave
            upper_lim = wave < 4.629296073126975

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)
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
        colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown'
        ]
        models = [key for key in self.out['originals'].keys()]
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
                    label = m + ' prob: {:.3f}'.format(self.out['weights'][m])
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
            # Age hist
            f, ax = plt.subplots(figsize=(12, 4))
            samp = self.out['mist_samples']['age']
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
            f, ax = plt.subplots(figsize=(12, 4))
            samp = self.out['mist_samples']['iso_mass']
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
            f, ax = plt.subplots(figsize=(12, 4))
            samp = self.out['mist_samples']['eep']
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
        print('Plotting HR diagram')
        # Get necessary info from the star.
        age = self.out[f'best_fit_{self.method}']['age']
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
                        tick.label.set_fontsize(self.corner_tick_fontsize)
                        tick.label.set_fontname(self.fontname)
                        ax.set_ylabel(
                            labels[yi],
                            labelpad=self.corner_labelpad,
                            fontsize=self.corner_fontsize,
                            fontname=self.fontname
                        )
                if yi == theta.shape[0] - 1:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(self.corner_tick_fontsize)
                        tick.label.set_fontname(self.fontname)
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
                tick.label.set_fontsize(self.corner_tick_fontsize)
                tick.label.set_fontname(self.fontname)

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
        lgg = 'g{:.0f}'.format(sel_logg * 10)
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
        lgg = 'g{:.0f}'.format(sel_logg * 10)
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
        sel_teff = str(sel_teff) if sel_teff >= 1e5 else '0{}'.format(sel_teff)
        selected_SED = self.moddir + 'Coelho14/t' + sel_teff + '_g'
        sel_logg = '+{:.1f}'.format(sel_logg) if sel_logg > 0 else '-0.5'
        selected_SED += sel_logg
        if sel_z < 0:
            selected_SED += '_m{:02.0f}'.format(-sel_z * 10)
        else:
            selected_SED += '_p{:02.0f}'.format(sel_z * 10)

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
                new_titles[i] += '{:.3f}'.format(theta[i])
                new_titles[i] += r'$^{+' + \
                                 '{:.3f}'.format(theta_up[i] - theta[i])
                new_titles[i] += r'}_{-' + \
                                 '{:.3f}'.format(theta[i] - theta_lo[i])
                new_titles[i] += r'}$'
            else:
                new_titles[i] += '{:.2f}'.format(theta[i])
                new_titles[i] += r'$^{+' + \
                                 '{:.2f}'.format(theta_up[i] - theta[i])
                new_titles[i] += r'}_{-' + \
                                 '{:.2f}'.format(theta[i] - theta_lo[i])
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
