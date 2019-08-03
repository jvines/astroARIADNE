"""plot_utils module for plotting SEDs."""
# TODO: Add a log file
# TODO: create settings file
from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy as sp
from matplotlib import rcParams

from phot_utils import *
from sed_library import *
from Star import *


class SEDPlotter:

    def __init__(self, star, fitter):
        # TODO: read settings file.
        self.star = star
        self.fitter = fitter

        self.__extract_info()
        self.__extract_params()
        self.read_config()

        self.model = model_grid(self.theta, self.star)

    def __extract_params(self):
        # teff, logg, z, dist, rad, Av = theta
        theta = []
        for i in range(5):
            theta.append(sp.median(self.fitter.chain[:, i]))
        self.theta = theta

    def __extract_info(self):
        self.flux = []
        self.flux_er = []
        self.wave = []
        self.bandpass = []

        for _, f in self.star.flux.items():
            self.flux.append(f)
        for _, e in self.star.flux_er.items():
            self.flux_er.append(e)
        for _, w in self.star.wave.items():
            self.wave.append(w)
        for _, bp in self.star.bandpass.items():
            self.bandpass.append(bp)

        self.flux = sp.array(self.flux)
        self.flux_er = sp.array(self.flux_er)
        self.wave = sp.array(self.wave)
        self.bandpass = sp.array(self.bandpass).T

    def plot_SED(self):
        """Create the plot of the SED."""
        # TODO: Finish the function

        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        mod_f = []
        for k in self.model.keys():
            mod_f.append(self.model[k])
        mod_f = sp.array(mod_f)

        f, ax = plt.subplots(figsize=self.figsize)

        ax.errorbar(self.wave, self.flux * self.wave,
                    xerr=self.bandpass, yerr=self.flux_er,
                    fmt=',',
                    # ecolor=self.error_color,
                    # color='turquoise',
                    marker=None,
                    alpha=self.error_alpha)

        ax.scatter(self.wave, self.flux * self.wave,
                   edgecolors=self.edgecolors,
                   c=self.marker_colors,
                   s=self.scatter_size,
                   alpha=self.scatter_linewidths)

        ax.scatter(self.wave, mod_f * self.wave,
                   marker='D',
                   edgecolors=self.edgecolors,
                   c=self.marker_colors_model,
                   s=self.scatter_size,
                   alpha=self.scatter_linewidths)

        ax.set_ylim([ymin * .8, ymax * 1.1])
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlabel(r'$\lambda (\mu m)$',
                      fontsize=self.fontsize,
                      fontname=self.fontname
                      )
        ax.set_ylabel(r'$\lambda$F$_\lambda$ (erg cm$^{-2}$s$^{-1}$)',
                      fontsize=self.fontsize,
                      fontname=self.fontname
                      )
        ax.set_xticks(range(1, 11))
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(
            axis='both', which='major',
            labelsize=self.tick_labelsize
        )
        for tick in ax.get_yticklabels():
            tick.set_fontname(self.fontname)
        for tick in ax.get_xticklabels():
            tick.set_fontname(self.fontname)

        pass

    def read_config(self):
        self.figsize = (12, 8)

        # SCATTER
        self.scatter_size = 80
        self.edgecolors = 'k'
        self.marker_colors = 'cyan'
        self.marker_colors_model = 'magenta'
        self.scatter_alpha = .85
        self.scatter_linewidths = 1

        # ERRORBARS
        self.error_color = 'k'
        self.error_alpha = 1

        # TEXT FORMAT
        self.fontsize = 22
        self.fontname = 'serif'
        self.tick_labelsize = 18
