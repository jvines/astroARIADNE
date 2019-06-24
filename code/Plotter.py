"""plot_utils module for plotting SEDs."""
# TODO: Add a log file
# TODO: create settings file
from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy as sp

from phot_utils import *
from Star import *


class SEDPlotter:

    def __init__(self, star):
        # TODO: read settings file.
        self.star = star

    def plot_SED(self):
        """Create the plot of the SED."""
        # TODO: Finish the function

        ymin, ymax = self.star.flux[:, 0].min(), self.star.flux[:, 0].max()

        f, ax = plt.subplots()

        ax.scatter(self.star.wave, self.star.flux[:, 0], edgecolors='k',
                   c='cyan', s=50, alpha=.85, linewidths=.5)
        ax.errorbar(self.star.wave, self.star.flux[:, 0],
                    xerr=self.star.bandpass, yerr=self.star.flux[:, 1],
                    fmt='o', ecolor='k', color='turquoise', marker=None,
                    alpha=.6)
        ax.set_ylim([ymin * .8, ymax * 1.1])
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlabel(r'$\lambda (\mu m)$')
        ax.set_ylabel(r'$\lambda$F (erg cm$^{-2}$s$^{-1}$)')
        ax.set_xticks(range(1, 11))
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        pass
