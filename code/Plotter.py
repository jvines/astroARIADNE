"""plot_utils module for plotting SEDs."""
# TODO: Add a log file
# TODO: create settings file
from __future__ import division, print_function

import copy
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy as sp
from astropy import units as u
from astropy.io import fits
from extinction import apply, fitzpatrick99
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from PyAstronomy import pyasl

import corner
from phot_utils import *
from sed_library import *
from Star import *

order = sp.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation'])
with open('interpolations.pkl', 'rb') as intp:
    interpolators = pickle.load(intp)


class SEDPlotter:

    wav_file = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

    def __init__(self, input_files, out_folder, pdf=False, png=True):
        # TODO: read settings file.
        # General setup
        self.pdf = pdf
        self.png = png

        # Create target folders
        try:
            chains = out_folder + '/chains'
            likelihoods = out_folder + '/likelihoods'
            os.mkdir(out_folder)
            os.mkdir(chains)
            os.mkdir(likelihoods)
        except OSError:
            print('Creation of one of the directories failed.', end=' ')
            print('Maybe they already exist?')
        else:
            print('Created the directories succesfully.')

        self.out_folder = out_folder
        self.chain_out = chains
        self.like_out = likelihoods
        self.hdd = '/Volumes/JVines_ext/PHOENIXv2/'

        # Read output files.
        out = pickle.load(open(input_files, 'rb'))
        self.out = out
        self.engine = out['engine']
        self.star = out['star']
        self.coordinator = out['coordinator']
        self.fixed = out['fixed']

        # Get best fit parameters.
        theta = sp.zeros(order.shape[0])
        for i, param in enumerate(order):
            if param != 'likelihood':
                theta[i] = out['best_fit'][param]
        self.theta = theta
        # self.theta = build_params(theta, self.coordinator, self.fixed)

        # Calculate best fit model.
        self.model = model_grid(self.theta, self.star, interpolators)

        # Get archival fluxes.
        self.__extract_info()

        # Setup plots.
        self.read_config()

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
        # TODO: Finish the function

        # Get plot ylims.
        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        # Get models residuals
        residuals, errors = get_residuals(self.theta, self.star, interpolators)

        # resdiuals = residuals / errors
        norm_res = residuals / errors

        # Create plot layout

        f = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 0.5], hspace=0.05)

        ax = f.add_subplot(gs[0])
        ax_r = f.add_subplot(gs[1])

        # SED plot.
        if True:
            Rv = 3.1  # For extinction.
            rad = self.theta[4]
            dist = self.theta[3] * u.pc.to(u.solRad)
            Av = self.theta[5]

            wave = fits.open(self.hdd + self.wav_file)[0].data
            wave *= u.angstrom.to(u.um)

            flux = self.fetch_SED()

            lower_lim = 0.25 < wave
            upper_lim = wave < 4.629296073126975
            new_w = wave[lower_lim * upper_lim]

            new_ww = sp.linspace(new_w[0], new_w[-1], len(new_w))

            ext = fitzpatrick99(new_w * 1e4, Av, Rv)

            brf, _ = pyasl.instrBroadGaussFast(
                new_ww, flux, 650,
                edgeHandling="firstlast",
                fullout=True, maxsig=8
            )
            brf = brf[lower_lim * upper_lim]
            brf = apply(ext, brf)
            flx = brf * (rad / dist) ** 2 * new_w
            ax.plot(new_w[:-1000], flx[:-1000], lw=1.25, color='k', zorder=0)
        # Model plot
        ax.errorbar(self.wave, self.flux * self.wave,
                    xerr=self.bandpass, yerr=self.flux_er,
                    fmt=',',
                    # ecolor=self.error_color,
                    # color='turquoise',
                    marker=None)

        ax.scatter(self.wave, self.flux * self.wave,
                   edgecolors=self.edgecolors,
                   c=self.marker_colors,
                   s=self.scatter_size,
                   alpha=self.scatter_alpha)

        ax.scatter(self.wave, self.model * self.wave,
                   marker='D',
                   edgecolors=self.marker_colors_model,
                   s=self.scatter_size,
                   facecolor='',
                   lw=3)

        # Residual plot
        ax_r.axhline(y=0, lw=2, ls='--', c='k', alpha=.7)

        ax_r.errorbar(self.wave, sp.zeros(self.wave.shape[0]),
                      xerr=self.bandpass, yerr=self.flux_er,
                      fmt=',',
                      # ecolor=self.error_color,
                      # color='turquoise',
                      marker=None)
        ax_r.scatter(self.wave, sp.zeros(self.wave.shape[0]),
                     edgecolors=self.edgecolors,
                     c=self.marker_colors,
                     s=self.scatter_size,
                     alpha=self.scatter_alpha)
        ax_r.scatter(self.wave, norm_res,
                     marker='D',
                     edgecolors=self.marker_colors_model,
                     s=self.scatter_size,
                     facecolor='',
                     lw=3,
                     zorder=10)

        # Formatting
        res_std = norm_res.std()
        ax.set_ylim([ymin * .8, ymax * 1.1])
        # ax_r.set_ylim([-5, 5])
        ax.set_xlim([0.25, 10])
        ax_r.set_xlim([0.25, 10])
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

    def plot_chains(self):
        """Plot SED chains."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                mx_prob = sp.argmax(samples['loglike'])
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
                ax.axhline(samples[param][mx_prob], color='red', lw=2)
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig(self.chain_out + '/' + param +
                            '.png', bbox_inches='tight')
        pass

    def plot_like(self):
        """Plot Likelihoods."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                mx_prob = sp.argmax(samples['loglike'])
                f, ax = plt.subplots(figsize=(12, 4))
                ax.scatter(samples[param], samples['loglike'], alpha=0.5, s=40)
                ax.axvline(samples[param][mx_prob], color='red', lw=1.5)
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
        pass

    def plot_corner(self):
        """Make corner plot."""
        samples = self.out['posterior_samples']
        all_samps = []
        theta_lo = []
        theta_up = []

        theta = self.theta[self.coordinator == 0]
        used_params = order[self.coordinator == 0]

        for i, param in enumerate(order):
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

    def fetch_SED(self):
        """Fetch correct SED file."""
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
        selected_SED = self.hdd + 'Z'
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED += metal_add
        selected_SED += '/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)
                                             ) == 5 else '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + '0'
        selected_SED += metal_add
        selected_SED += '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        flux = fits.open(selected_SED)[0].data
        flux *= (u.erg / u.s / u.cm**2 / u.cm).to(u.erg / u.s / u.cm**2 / u.um)
        return flux

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
                new_titles[i] = r'  D ='
            if param == 'rad':
                new_titles[i] = r'R ='
            if param == 'Av':
                new_titles[i] = r'Av ='
            if param == 'inflation':
                new_titles[i] = r'$\sigma$ ='
            new_titles[i] += '{:.2f}'.format(theta[i])
            new_titles[i] += r'$^{+' + '{:.2f}'.format(theta_up[i])
            new_titles[i] += r'}_{-' + '{:.2f}'.format(theta_lo[i])
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
            if param == 'Av':
                new_labels[i] = r'Av'
            if param == 'inflation':
                new_labels[i] = r'$\sigma$'
        return new_labels

    def read_config(self):
        self.figsize = (12, 8)

        # SCATTER
        self.scatter_size = 60
        self.edgecolors = 'k'
        self.marker_colors = 'deepskyblue'
        self.marker_colors_model = 'mediumvioletred'
        self.scatter_alpha = .85
        self.scatter_linewidths = 1

        # ERRORBARS
        self.error_color = 'k'
        self.error_alpha = 1

        # TEXT FORMAT
        self.fontsize = 22
        self.fontname = 'serif'
        self.tick_labelsize = 18

        # CORNER
        self.corner_med_c = 'firebrick'
        self.corner_v_c = 'lightcoral'
        self.corner_v_style = '-.'
        self.corner_med_style = '--'
        self.corner_fontsize = 20
        self.corner_tick_fontsize = 15
        self.corner_labelpad = 15
        self.corner_marker = 'sr'
