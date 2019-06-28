"""Main driver of the fitting routine."""
from __future__ import division, print_function

from multiprocessing import cpu_count

import astropy.units as u
import scipy as sp
import scipy.stats as st
from scipy.interpolate import griddata

import emcee
from phot_utils import *
from sed_library import *


class Fitter:

    def __init__(self, star, nwalkers, nsteps, priorfile=None, burnout=False):
        self.star = star

        # emcee settings
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnout = nsteps // 2 if not burnout else burnout
        self.cores = cpu_count()

        # Setup priors.
        self.create_priors(priorfile)

    def create_artist(self):
        """Instantiate the plotter here."""
        pass

    def create_priors(self, priorfile):
        """Read the prior file.

        Returns a dictionary with the pdfs of each parameter/prior
        """
        if priorfile:
            param, prior, bounds = sp.loadtxt(
                priorfile, usecols=[0, 1, 2], unpack=True)
        # Dict with priors.
        prior_dict = dict()
        if priorfile:
            for par, pri, bo in zip(param, prior, bounds):
                if pri.lower() == 'uniform':
                    a, b = bounds.split(',')
                    a, b = float(a), float(b)
                    prior_dict[par] = st.uniform(loc=a, scale=b - a)
                elif pri.lower() == 'normal':
                    mu, sig = bounds.split(',')
                    mu, sig = float(mu), float(sig)
                    prior_dict[par] = st.norm(loc=mu, scale=sig)
                elif pri.lower() == 'truncatednormal':
                    mu, sig, up, low = bounds.split(',')
                    mu, sig, up, low = float(mu), float(
                        sig), float(up), float(low)
                    up, low = (up - mu) / sig, (low - mu) / sig
                    priot_dict[par] = st.truncnorm(
                        a=low, b=up, loc=mu, scale=sig)
                elif pri.lower() == 'fixed':
                    continue
        else:
            prior_dict['teff'] = st.uniform(loc=2300, scale=12000 - 2300)
            prior_dict['logg'] = st.uniform(loc=0, scale=6)
            if not self.star.fixed_z:
                prior_dict['z'] = st.uniform(loc=-4, scale=1 + 4)

            prior_dict['dist'] = st.norm(
                loc=self.star.dist, scale=2 * self.star.dist_e)

            if self.star.get_rad:
                prior_dict['radius'] = st.norm(
                    loc=self.star.rad, scale=2 * self.star.rad_e)
            else:
                prior_dict['radius'] = st.uniform(loc=.1, scale=15 - .1)
        self.priors = prior_dict

    def pos(self, ndim):
        p0 = sp.empty(self.nwalkers)

        for k in self.priors.keys():
            p0 = sp.vstack((p0, self.priors[k].rvs(size=self.nwalkers)))
        self.p0 = p0.T[:, 1:]

    def fit(self):
        """Run emcee here."""
        # The parameters are teff, logg, z, dist, rad - unless z is fixed.
        ndim = 5 if not self.star.fixed_z else 4
        # Randomize starting position for the walkers.
        self.pos(ndim)
        sampler = emcee.EnsembleSampler(
            self.nwalkers, ndim, log_probability,
            args=[self.star, self.priors], threads=self.cores
        )
        sampler.run_mcmc(self.p0, self.nsteps, progress=True)
        flat_samples = sampler.get_chain(
            discard=self.burnout, thin=1, flat=True)
        for i in range(ndim):
            print(flat_samples[:, i].mean())
