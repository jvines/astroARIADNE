# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*elif/ /^\s*def/
"""plot_utils module for plotting SEDs."""
# TODO: create settings file

import copy
import glob
import os
from contextlib import closing

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy as sp
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from extinction import apply
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from PyAstronomy import pyasl

import corner
from dynesty import plotting as dyplot
from phot_utils import *
from sed_library import *
from Star import *
from utils import *


class SEDPlotter:
    """Short summary.

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
    hdd : type
        Description of attribute `hdd`.
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

    __wav_file = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

    def __init__(self, input_files, out_folder, pdf=False, png=True,
                 model=None):
        """See class docstring."""
        print('\nInitializing plotter.\n')
        # General setup
        self.pdf = pdf
        self.png = png
        self.out_folder = out_folder

        chains = out_folder + '/chains'
        likelihoods = out_folder + '/likelihoods'
        posteriors = out_folder + '/posteriors'
        self.chain_out = chains
        self.like_out = likelihoods
        self.post_out = posteriors
        self.hdd = '/Volumes/JVines_ext/StellarAtmosphereModels/'

        # Read output files.
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
                zs = sp.array([out['lnZ'][key] for key in out['lnZ'].keys()])
                keys = sp.array([key for key in out['lnZ'].keys()])
                grid = keys[sp.argmax(zs)]
                self.grid = grid
        else:
            self.grid = model
        self.av_law = out['av_law']

        # Create target folders
        create_dir(out_folder)
        if self.engine != 'Bayesian Model Averaging':
            create_dir(chains)
            create_dir(likelihoods)
            create_dir(posteriors)

        self.star.load_grid(self.grid)

        if not self.norm:
            self.order = sp.array(
                [
                    'teff', 'logg', 'z',
                    'dist', 'rad', 'Av',
                    'inflation'
                ]
            )
        else:
            self.order = sp.array(
                ['teff', 'logg', 'z', 'norm', 'Av', 'inflation'])

        directory = '../Datafiles/model_grids/'
        if self.grid.lower() == 'phoenix':
            with open(directory + 'interpolations_Phoenix.pkl', 'rb') as intp:
                self.interpolator = pickle.load(intp)
        if self.grid.lower() == 'btsettl':
            with open(directory + 'interpolations_BTSettl.pkl', 'rb') as intp:
                self.interpolator = pickle.load(intp)
        if self.grid.lower() == 'ck04':
            with open(directory + 'interpolations_CK04.pkl', 'rb') as intp:
                self.interpolator = pickle.load(intp)
        if self.grid.lower() == 'kurucz':
            with open(directory + 'interpolations_Kurucz.pkl', 'rb') as intp:
                self.interpolator = pickle.load(intp)
        if self.grid.lower() == 'nextgen':
            with open(directory + 'interpolations_NextGen.pkl', 'rb') as intp:
                self.interpolator = pickle.load(intp)

        # Get best fit parameters.
        theta = sp.zeros(self.order.shape[0])
        for i, param in enumerate(self.order):
            if param != 'likelihood':
                theta[i] = out['best_fit'][param]
        self.theta = theta
        # self.theta = build_params(theta, self.coordinator, self.fixed)

        # Calculate best fit model.
        self.model = model_grid(self.theta, self.star,
                                self.interpolator, self.norm, self.av_law)

        # Get archival fluxes.
        self.__extract_info()

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

        self.flux = sp.array(self.flux)
        self.flux_er = sp.array(self.flux_er)
        self.wave = sp.array(self.wave)
        self.bandpass = sp.array(self.bandpass).T

    def plot_SED(self):
        """Create the plot of the SED."""
        # Get plot ylims.
        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        # Get models residuals
        residuals, errors = get_residuals(
            self.theta, self.star, self.interpolator, self.norm, self.av_law)

        # resdiuals = residuals / errors
        norm_res = residuals / errors

        # Create plot layout

        f = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 0.5], hspace=0.05)

        ax = f.add_subplot(gs[0])
        ax_r = f.add_subplot(gs[1])

        self.SED(ax)

        # Model plot
        ax.errorbar(self.wave, self.flux * self.wave,
                    xerr=self.bandpass, yerr=self.flux_er,
                    fmt=',',
                    ecolor=self.error_color,
                    # color='turquoise',
                    marker=None)

        ax.scatter(self.wave, self.flux * self.wave,
                   edgecolors=self.edgecolors,
                   marker=self.marker,
                   c=self.marker_colors,
                   s=self.scatter_size,
                   alpha=self.scatter_alpha)

        ax.scatter(self.wave, self.model * self.wave,
                   marker=self.marker_model,
                   edgecolors=self.marker_colors_model,
                   s=self.scatter_size,
                   facecolor='',
                   lw=3)

        # Residual plot
        ax_r.axhline(y=0, lw=2, ls='--', c='k', alpha=.7)

        ax_r.errorbar(self.wave, sp.zeros(self.wave.shape[0]),
                      xerr=self.bandpass, yerr=self.flux_er,
                      fmt=',',
                      ecolor=self.error_color,
                      # color='turquoise',
                      marker=None)
        ax_r.scatter(self.wave, sp.zeros(self.wave.shape[0]),
                     edgecolors=self.edgecolors,
                     marker=self.marker,
                     c=self.marker_colors,
                     s=self.scatter_size,
                     alpha=self.scatter_alpha)
        ax_r.scatter(self.wave, norm_res,
                     marker=self.marker_model,
                     edgecolors=self.marker_colors_model,
                     s=self.scatter_size,
                     facecolor='',
                     lw=3,
                     zorder=10)

        # Formatting
        res_std = norm_res.std()
        ax.set_ylim([ymin * .8, ymax * 1.25])
        # ax_r.set_ylim([-5, 5])
        ax_r.set_ylim([-4 * res_std, 4 * res_std])
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
        ax_r.tick_params(
            axis='both', which='major',
            labelsize=self.tick_labelsize
        )
        ax_r.set_xticks(sp.linspace(1, 10, 10))
        ax_r.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(sp.linspace(1, 10, 10))
        ax.get_xaxis().set_major_formatter(ticker.NullFormatter())
        ylocmin = ticker.LinearLocator(numticks=4)

        ax_r.yaxis.set_minor_locator(ylocmin)
        ax_r.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xlim([0.25, 6])
        ax_r.set_xlim([0.25, 6])

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
            plt.savefig(self.out_folder + '/SED.pdf', bbox_inches='tight')
        if self.png:
            plt.savefig(self.out_folder + '/SED.png', bbox_inches='tight')
        pass

    def SED(self, ax):
        """Plot the SED model."""
        Rv = 3.1  # For extinction.
        rad = self.theta[4]
        dist = self.theta[3] * u.pc.to(u.solRad)
        Av = self.theta[5]

        # SED plot.
        if self.grid == 'phoenix':
            wave = fits.open(self.hdd + self.__wav_file)[0].data
            wave *= u.angstrom.to(u.um)

            lower_lim = 0.25 < wave
            upper_lim = wave < 4.629296073126975

            flux = self.fetch_Phoenix()

            new_w = wave[lower_lim * upper_lim]

            new_ww = sp.linspace(new_w[0], new_w[-1], len(new_w))

            ext = self.av_law(new_w * 1e4, Av, Rv)

            brf, _ = pyasl.instrBroadGaussFast(
                new_ww, flux, 650,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            brf = brf[lower_lim * upper_lim]
            brf = apply(ext, brf)
            flx = brf * (rad / dist) ** 2 * new_w
            ax.plot(new_w[:-1000], flx[:-1000], lw=1.25, color='k', zorder=0)

        elif self.grid == 'btsettl':
            wave, flux = self.fetch_btsettl()

            lower_lim = 0.25 < wave
            upper_lim = wave < 4.629296073126975

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)

            new_w = sp.linspace(wave[0], wave[-1], len(wave))

            brf, _ = pyasl.instrBroadGaussFast(
                new_w, flux, 650,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            flx = apply(ext, brf)
            flx *= wave * (rad / dist) ** 2
            ax.plot(wave, flx, lw=1.25, color='k', zorder=0)

        elif self.grid == 'nextgen':
            wave, flux = self.fetch_nextgen()

            lower_lim = 0.25 < wave
            upper_lim = wave < 4.629296073126975

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)

            new_w = sp.linspace(wave[0], wave[-1], len(wave))

            brf, _ = pyasl.instrBroadGaussFast(
                new_w, flux, 650,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            flx = apply(ext, brf)
            flx *= wave * (rad / dist) ** 2
            ax.plot(wave, flx, lw=1.25, color='k', zorder=0)

        elif self.grid == 'ck04':
            wave, flux = self.fetch_ck04()

            lower_lim = 0.25 < wave
            upper_lim = wave < 4.629296073126975

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * (rad / dist) ** 2
            ax.plot(wave, flux, lw=1.25, color='k', zorder=0)

        elif self.grid == 'kurucz':
            wave, flux = self.fetch_kurucz()

            lower_lim = 0.25 < wave
            upper_lim = wave < 4.629296073126975

            wave = wave[lower_lim * upper_lim]
            flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * (rad / dist) ** 2
            ax.plot(wave, flux, lw=1.25, color='k', zorder=0)
        pass

    def plot_chains(self):
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
                # ax.axhline(sp.median(samples[param]), color='red', lw=2)
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
                # ax.axvline(sp.median(samples[param]), color='red', lw=1.5)
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
                # ax.axvline(sp.median(samples[param]), color='red', lw=1.5)
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

    def plot_corner(self):
        """Make corner plot."""
        samples = self.out['posterior_samples']
        all_samps = []
        theta_lo = []
        theta_up = []

        theta = self.theta[self.coordinator == 0]
        used_params = self.order[self.coordinator == 0]

        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                _, lo, up = credibility_interval(
                    samples[param])
                theta_lo.append(lo)
                theta_up.append(up)
                all_samps.append(samples[param])

        corner_samp = sp.vstack(all_samps)

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

        axes = sp.array(fig.axes).reshape((theta.shape[0], theta.shape[0]))

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
                plt.savefig(self.out_folder + '/CORNER.pdf',
                            bbox_inches='tight')
            if self.png:
                plt.savefig(self.out_folder + '/CORNER.png',
                            bbox_inches='tight')
        pass

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
        select_teff = sp.argmin((abs(teff - sp.unique(self.star.teff))))
        select_logg = sp.argmin((abs(logg - sp.unique(self.star.logg))))
        select_z = sp.argmin((abs(z - sp.unique(self.star.z))))
        sel_teff = int(sp.unique(self.star.teff)[select_teff])
        sel_logg = sp.unique(self.star.logg)[select_logg]
        sel_z = sp.unique(self.star.z)[select_z]
        selected_SED = self.hdd + 'PHOENIXv2/Z'
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
        flux *= (u.erg / u.s / u.cm**2 / u.cm).to(u.erg / u.s / u.cm**2 / u.um)
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
        conversion = (u.erg / u.s / u.cm**2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm**2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = sp.argmin((abs(teff - sp.unique(self.star.teff))))
        select_logg = sp.argmin((abs(logg - sp.unique(self.star.logg))))
        select_z = sp.argmin((abs(z - sp.unique(self.star.z))))
        sel_teff = int(sp.unique(self.star.teff)[select_teff]) // 100
        sel_logg = sp.unique(self.star.logg)[select_logg]
        sel_z = sp.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.hdd + 'BTSettl/AGSS2009/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+0.0'
        selected_SED += '.BT-Settl.AGSS2009.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        flux = sp.array(tab['FLUX'].tolist()) * conversion
        wave = sp.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        return wave, flux

    def fetch_nextgen(self):
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
        conversion = (u.erg / u.s / u.cm**2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm**2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = sp.argmin((abs(teff - sp.unique(self.star.teff))))
        select_logg = sp.argmin((abs(logg - sp.unique(self.star.logg))))
        select_z = sp.argmin((abs(z - sp.unique(self.star.z))))
        sel_teff = int(sp.unique(self.star.teff)[select_teff]) // 100
        sel_logg = sp.unique(self.star.logg)[select_logg]
        sel_z = sp.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.hdd + 'BTNextGen/AGSS2009/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+0.0'
        selected_SED += '.BT-NextGen.AGSS2009.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        flux = sp.array(tab['FLUX'].tolist()) * conversion
        wave = sp.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
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
        conversion = (u.erg / u.s / u.cm**2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm**2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = sp.argmin((abs(teff - sp.unique(self.star.teff))))
        select_logg = sp.argmin((abs(logg - sp.unique(self.star.logg))))
        select_z = sp.argmin((abs(z - sp.unique(self.star.z))))
        sel_teff = int(sp.unique(self.star.teff)[select_teff])
        sel_logg = sp.unique(self.star.logg)[select_logg]
        sel_z = sp.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = 'm' + str(-sel_z).replace('.', '')
        if sel_z == 0:
            metal_add = 'p00'
        if sel_z > 0:
            metal_add = 'p' + str(sel_z).replace('.', '')
        name = 'ck' + metal_add
        lgg = 'g{:.0f}'.format(sel_logg * 10)
        selected_SED = self.hdd + 'Castelli_Kurucz/' + name + '/' + name
        selected_SED += '_' + str(sel_teff) + '.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        wave = sp.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        flux = sp.array(tab[lgg].tolist()) * conversion
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
        conversion = (u.erg / u.s / u.cm**2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm**2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = sp.argmin((abs(teff - sp.unique(self.star.teff))))
        select_logg = sp.argmin((abs(logg - sp.unique(self.star.logg))))
        select_z = sp.argmin((abs(z - sp.unique(self.star.z))))
        sel_teff = int(sp.unique(self.star.teff)[select_teff])
        sel_logg = sp.unique(self.star.logg)[select_logg]
        sel_z = sp.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = 'm' + str(-sel_z).replace('.', '')
        if sel_z == 0:
            metal_add = 'p00'
        if sel_z > 0:
            metal_add = 'p' + str(sel_z).replace('.', '')
        name = 'k' + metal_add
        lgg = 'g{:.0f}'.format(sel_logg * 10)
        selected_SED = self.hdd + 'Kurucz/' + name + '/' + name
        selected_SED += '_' + str(sel_teff) + '.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        wave = sp.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        flux = sp.array(tab[lgg].tolist()) * conversion
        return wave, flux

    def __create_titles(self, titles, theta, theta_up, theta_lo):
        new_titles = sp.empty(titles.shape[0], dtype=object)
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
            new_titles[i] += '{:.2f}'.format(theta[i])
            new_titles[i] += r'$^{+' + '{:.2f}'.format(theta_up[i] - theta[i])
            new_titles[i] += r'}_{-' + '{:.2f}'.format(theta[i] - theta_lo[i])
            new_titles[i] += r'}$'
        return new_titles

    def __create_labels(self, labels):
        new_labels = sp.empty(labels.shape[0], dtype=object)
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
        settings = open('../Datafiles/plot_settings.dat', 'r')
        for line in settings.readlines():
            if line[0] == '#' or line[0] == '\n':
                continue
            splt = line.split(' ')
            attr = splt[0]
            if attr == 'figsize':
                vals = splt[1].split('\n')[0].split(',')
                val = (int(vals[0]), int(vals[1]))
                setattr(self, attr, val)
            else:
                try:
                    val = int(splt[1].split('\n')[0])
                except ValueError:
                    val = splt[1].split('\n')[0]
                setattr(self, attr, val)
