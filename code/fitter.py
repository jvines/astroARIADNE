"""Main driver of the fitting routine."""
from __future__ import division, print_function

import pickle
import time
from multiprocessing import Pool, cpu_count

import astropy.units as u
import scipy as sp
import scipy.stats as st
from scipy.interpolate import griddata

import emcee
from phot_utils import *
from sed_library import *


class Fitter:

    order = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']

    def __init__(self, star, nwalkers, nsteps, priorfile=None, burnout=False):
        self.star = star
        # Interpolations
        with open('interpolations.pkl', 'rb') as intp:
            self.interpolators = pickle.load(intp)
        # Parameter coordination.
        # Order for the parameters are:
        # tef, logg, z, dist, rad, Av
        self.coordinator = dict()  # 1 for fixed params

        # Setup priors.
        self.default_priors = self._default_priors()
        self.priorfile = priorfile
        self.create_priors(priorfile)

        # emcee settings
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnout = nsteps // 2 if not burnout else burnout
        self.cores = cpu_count()
        self.ndim = self.get_ndim()

    def create_artist(self):
        """Instantiate the plotter here."""
        pass

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = 6 - len(self.coordinator.keys())
        return ndim

    def _default_priors(self):
        defaults = dict()
        with open('z_logg_kde.pkl', 'rb') as pkl:
            logg_prior = pickle.load(pkl)
        defaults['teff'] = st.norm(
            loc=self.star.temp, scale=2 * self.star.temp_e)
        defaults['logg'] = logg_prior['logg']
        defaults['z'] = st.norm(loc=-0.125, scale=0.234)
        defaults['dist'] = st.norm(
            loc=self.star.dist, scale=2 * self.star.dist_e)
        defaults['rad'] = st.norm(
            loc=self.star.rad, scale=2 * self.star.rad_e)
        defaults['Av'] = st.uniform(loc=0, scale=.032)
        # defaults['inflation'] = st.norm(loc=0, scale=.05)
        return defaults

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
                    a, b = bo.split(',')
                    a, b = float(a), float(b)
                    prior_dict[par] = st.uniform(loc=a, scale=b - a)
                elif pri.lower() == 'normal':
                    mu, sig = bo.split(',')
                    mu, sig = float(mu), float(sig)
                    prior_dict[par] = st.norm(loc=mu, scale=sig)
                elif pri.lower() == 'truncatednormal':
                    mu, sig, up, low = bo.split(',')
                    mu, sig, up, low = float(mu), float(
                        sig), float(up), float(low)
                    up, low = (up - mu) / sig, (low - mu) / sig
                    priot_dict[par] = st.truncnorm(
                        a=low, b=up, loc=mu, scale=sig)
                elif pri.lower() == 'default':
                    prior_dict[par] = self.default_priors[par]
                elif pri.lower() == 'fixed':
                    self.coordinator[par] = bo
            self.priors = prior_dict
        else:
            print('No priorfile detected. Using default priors', end=' ')
            print('for exoplanet hosts.')
            self.priors = self.default_priors

    def pos(self):
        """Set initial position of the walkers."""
        p0 = sp.empty(self.nwalkers)

        for k in self.order:
            try:
                if self.coordinator[k]:
                    flag = True
            except KeyError:
                flag = False

            if flag:
                continue
            if k == 'logg' and not self.priorfile:
                p0 = sp.vstack((p0, self.priors[k].resample(self.nwalkers)))
            else:
                p0 = sp.vstack((p0, self.priors[k].rvs(size=self.nwalkers)))
        self.p0 = p0.T[:, 1:]

    def fit(self):
        """Run emcee."""
        # Randomize starting position for the walkers.
        self.pos()

        self.sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, log_probability,
            args=[
                self.star,
                self.priors,
                self.coordinator,
                self.interpolators
            ],
            threads=cpu_count()
        )
        self.sampler.run_mcmc(self.p0, self.nsteps + self.burnout,
                              progress=True)
        flat_samples = self.sampler.get_chain(
            discard=self.burnout, thin=1, flat=True)
        self.chain = flat_samples
