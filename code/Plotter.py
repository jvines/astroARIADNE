"""plot_utils module for plotting SEDs."""
# TODO: Add a log file
# TODO: create settings file
from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy as sp
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

from phot_utils import *
from sed_library import *
from Star import *

order = sp.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation'])
with open('interpolations.pkl', 'rb') as intp:
    interpolators = pickle.load(intp)


class SEDPlotter:

    def __init__(self, fitter):
        # TODO: read settings file.
        self.fitter = fitter

        self.__extract_info()
        # self.__extract_params()
        self.read_config()

        if self.fitter.engine == 'multinest':
            out = pickle.load(open('multinest_out.pkl', 'rb'))
            theta = out['best_fit']['parameters']
            self.theta = build_params(
                theta, self.fitter.coordinator, self.fitter.fixed
            )

        self.model = model_grid(
            self.theta, self.fitter.star, interpolators)

    def __extract_params(self):
        # teff, logg, z, dist, rad, Av = theta
        theta = self.fitter.sampler.get_chain(flat=True)[sp.argmax(
            self.fitter.sampler.get_log_prob(flat=True))]
        self.theta = theta

    def __extract_info(self):
        self.flux = []
        self.flux_er = []
        self.wave = []
        self.bandpass = []

        for i, f in zip(self.fitter.star.used_filters, self.fitter.star.flux):
            if i:
                self.flux.append(f)
        for i, e in zip(self.fitter.star.used_filters,
                        self.fitter.star.flux_er):
            if i:
                self.flux_er.append(e)
        for i, w in zip(self.fitter.star.used_filters, self.fitter.star.wave):
            if i:
                self.wave.append(w)
        for i, bp in zip(self.fitter.star.used_filters,
                         self.fitter.star.bandpass):
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
        residuals, errors = get_residuals(
            self.theta, self.fitter.star, interpolators)

        # resdiuals = residuals / errors
        norm_res = residuals / errors

        # Create plot layout

        f = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 0.5], hspace=0.05)

        ax = f.add_subplot(gs[0])
        ax_r = f.add_subplot(gs[1])

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
        # ax.set_xlim([0.18, 10])
        # ax_r.set_xlim([0.18, 10])
        ax_r.set_ylim([-4 * res_std, 4 * res_std])
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax_r.set_xscale('log', nonposx='clip')
        ax.set_xlabel(r'$\lambda (\mu m)$',
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

        plt.savefig('plots/test.png', bbox_inches='tight')
        pass

    def plot_chains(self):
        """Plot SED chains."""
        if self.fitter.engine == 'multinest':
            out = pickle.load(open('multinest_out.pkl', 'rb'))
            samples = out['posterior_samples']
        for i, param in enumerate(order):
            if not self.fitter.coordinator[i]:
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
                ax.axhline(samples[param][mx_prob], color='gray', lw=2)
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig('plots/sample_' + param + '.png',
                            bbox_inches='tight')
        pass

    def plot_like(self):
        """Plot Likelihoods."""
        if self.fitter.engine == 'multinest':
            out = pickle.load(open('multinest_out.pkl', 'rb'))
            samples = out['posterior_samples']
        for i, param in enumerate(order):
            if not self.fitter.coordinator[i]:
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
                plt.savefig('plots/likelihood_' + param + '.png',
                            bbox_inches='tight')
        pass

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
