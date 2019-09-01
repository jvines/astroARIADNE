"""Main driver of the fitting routine."""
from __future__ import division, print_function

import os
import pickle
import random
import time
from multiprocessing import Pool, cpu_count

import astropy.units as u
import dill
import scipy as sp
import scipy.stats as st
from isochrones import SingleStarModel, get_ichrone
from termcolor import colored

import pymultinest
from isochrone import estimate
from phot_utils import *
from sed_library import *

# GLOBAL VARIABLES

order = sp.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation'])
with open('interpolations.pkl', 'rb') as intp:
    interpolators = pickle.load(intp)


class Fitter:

    def __init__(self, star, setup, priorfile=None, estimate_logg=False,
                 engine='multinest', out_folder=None,
                 verbose=True):
        global prior_dict, coordinator, fixed
        # Global settings
        self.start = time.time()
        self.star = star
        self.engine = engine
        self.verbose = verbose

        if out_folder is None:
            self.out_folder = self.star.starname + '/'
        else:
            self.out_folder = out_folder

        try:
            os.mkdir(self.out_folder)
        except OSError:
            print("Creation of the directory {:s} failed".format(
                self.out_folder))
        else:
            print("Created the directory {:s} ".format(self.out_folder))

        if engine == 'multinest' or engine == 'dynesty':
            self.live_points = setup[0]
            self.dlogz = setup[1]

        # Parameter coordination.
        # Order for the parameters are:
        # tef, logg, z, dist, rad, Av
        self.coordinator = sp.zeros(7)  # 1 for fixed params
        self.fixed = sp.zeros(7)
        coordinator = self.coordinator
        fixed = self.fixed

        # Setup priors.
        self.default_priors = self._default_priors(estimate_logg)
        self.priorfile = priorfile
        self.create_priors(priorfile)
        prior_dict = self.priors

        # Get dimensions.
        self.ndim = self.get_ndim()

        self.display()

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = 7 - self.coordinator.sum()
        return int(ndim)

    def _default_priors(self, estimate_logg):
        defaults = dict()
        # Logg prior setup.
        if not estimate_logg:
            with open('logg_ppf.pkl', 'rb') as jar:
                defaults['logg'] = pickle.load(jar)
        else:
            params = dict()  # params for isochrones.
            if self.star.get_temp:
                params['Teff'] = (self.star.temp, self.star.temp_e)
            if self.star.get_lum:
                params['LogL'] = (self.star.lum, self.star.lum_e)
            if self.star.get_rad:
                params['radius'] = (self.star.rad, self.star.rad_e)
            if self.star.get_plx:
                params['parallax'] = (self.star.plx, self.star.plx_e)
            mask = sp.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
            mags = self.star.mags[mask == 1]
            mags_e = self.star.mag_errs[mask == 1]
            bands = ['H', 'J', 'K', 'G', 'RP', 'BP', 'W1', 'W2']
            used_bands = []
            for m, e, b in zip(mags, mags_e, bands):
                if m != 0:
                    params[b] = (m, e)
                    used_bands.append(b)
            if self.verbose and estimate_logg:
                print('*** ESTIMATING LOGG USING MIST ISOCHRONES ***')
            logg_est = estimate(used_bands, params)
            if logg_est is not None:
                defaults['logg'] = st.norm(loc=logg_est[0], scale=logg_est[1])
        # Teff prior setup.
        if self.star.get_temp:
            defaults['teff'] = st.norm(
                loc=self.star.temp, scale=self.star.temp_e)
        else:
            with open('teff_ppf.pkl', 'rb') as jar:
                defaults['teff'] = pickle.load(jar)
            defaults['teff'] = teff_prior['teff']
        defaults['z'] = st.norm(loc=-0.125, scale=0.234)
        defaults['dist'] = st.norm(
            loc=self.star.dist, scale=self.star.dist_e)
        defaults['rad'] = st.norm(
            loc=self.star.rad, scale=self.star.rad_e)
        defaults['Av'] = st.uniform(loc=0, scale=self.star.Av)
        up, low = (5 - 0.5) / 0.5, (0 - 0.5) / 0.5
        defaults['inflation'] = st.truncnorm(a=low, b=up, loc=0.5, scale=0.5)
        return defaults

    def create_priors(self, priorfile):
        """Read the prior file.

        Returns a dictionary with each parameter's prior
        """
        if priorfile:
            param, prior, bounds = sp.loadtxt(
                priorfile, usecols=[0, 1, 2], unpack=True, dtype=object)
            # Dict with priors.
            prior_dict = dict()
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
                    idx = sp.where(par == order)[0]
                    self.coordinator[idx] = 1
                    self.fixed[idx] = float(bo)
            self.priors = prior_dict
        else:
            print('No priorfile detected. Using default priors.')
            self.priors = self.default_priors
        pass

    def save_multinest(self):
        """Analyze and save multinest output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        out : the whole multinest output, raw.
        lnZ : The global evidence
        lnZerr : The global evidence error
        posterior_samples : A dictionary containing the sampls of each
                            parameter (even if it's fixed) and the
                            log likelihood for each set of sampled parameters.
        fixed : An array with the fixed parameter values
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit : The best fit according to multinest. This corresponds to
                   finding the set of parameters that corresponds to the
                   highest log likelihood
        star : The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)
        engine : The fitting engine used (i.e. MultiNest or Dynesty)

        """
        path = self.out_folder + 'multinest/'
        out = dict()
        output = pymultinest.Analyzer(outputfiles_basename=path + 'chains',
                                      n_params=self.ndim)
        posterior_samples = output.get_equal_weighted_posterior()[:, :-1]
        out['out'] = output
        out['lnZ'] = output.get_stats()['global evidence']
        out['lnZerr'] = output.get_stats()['global evidence error']
        out['posterior_samples'] = dict()
        j = 0
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                out['posterior_samples'][param] = posterior_samples[:, j]
                j += 1
            else:
                out['posterior_samples'][param] = self.fixed[i]
        out['posterior_samples']['loglike'] = sp.zeros(
            posterior_samples.shape[0])
        for i in range(posterior_samples.shape[0]):
            theta = build_params(posterior_samples[i, :], coordinator, fixed)
            out['posterior_samples']['loglike'][i] = log_likelihood(
                theta, self.star, interpolators)
        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['best_fit'] = dict()
        best_theta = sp.zeros(order.shape[0])
        j = 0
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                out['best_fit'][param] = sp.median(posterior_samples[:, j])
                j += 1
            else:
                out['best_fit'][param] = self.fixed[i]
            best_theta[i] = out['best_fit'][param]
        out['best_fit']['likelihood'] = log_likelihood(
            best_theta, self.star, interpolators)
        out['star'] = self.star
        out['engine'] = self.engine
        pickle.dump(out, open(self.out_folder + '/multinest_out.pkl', 'wb'))
        pass

    def fit_multinest(self):
        """Run MuiltiNest."""
        global star
        star = self.star
        path = self.out_folder + 'multinest/'
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory {:s} failed".format(path))
        else:
            print("Created the directory {:s} ".format(path))
        pymultinest.run(
            log_like, pt_multinest, self.ndim,
            n_params=self.ndim,
            sampling_efficiency=0.8,
            evidence_tolerance=self.dlogz,
            n_live_points=self.live_points,
            outputfiles_basename=path + 'chains',
            verbose=self.verbose,
            resume=False
        )
        self.save_multinest()
        elapsed_time = self.execution_time()
        self.end(elapsed_time)
        pass

    def fit_dynesty(self):
        """Run dynesty."""
        # TODO: implement
        pass

    def execution_time(self):
        """Calculate run execution time."""
        end = time.time() - self.start
        weeks, rest0 = end // 604800, end % 604800
        days, rest1 = rest0 // 86400, rest0 % 86400
        hours, rest2 = rest1 // 3600, rest1 % 3600
        minutes, seconds = rest2 // 60, rest2 % 60
        elapsed = ''
        if weeks == 0:
            if days == 0:
                if hours == 0:
                    if minutes == 0:
                        elapsed = '{:f} seconds'.format(seconds)
                    else:
                        elapsed = '{:f} minutes'.format(minutes)
                        elapsed += ' and {:f} seconds'.format(seconds)
                else:
                    elapsed = '{:f} hours'.format(hours)
                    elapsed += ', {:f} minutes'.format(minutes)
                    elapsed += ' and {:f} seconds'.format(seconds)
            else:
                elapsed = '{:f} days'.format(days)
                elapsed += ', {:f} hours'.format(hours)
                elapsed += ', {:f} minutes'.format(minutes)
                elapsed += ' and {:f} seconds'.format(seconds)
        else:
            elapsed = '{:f} weeks'.format(weeks)
            elapsed += ', {:f} days'.format(days)
            elapsed += ', {:f} hours'.format(hours)
            elapsed += ', {:f} minutes'.format(minutes)
            elapsed += ' and {:f} seconds'.format(seconds)
        return elapsed

    def display(self):
        """Display program information.

        What is displayed is:
        Program name
        Program author
        Star selected
        Algorithm used (i.e. Multinest or Dynesty)
        Setup used (i.e. Live points, dlogz tolerance)
        """
        colors = [
            'red', 'green', 'blue', 'yellow',
            'grey', 'magenta', 'cyan', 'white'
        ]
        c = random.choice(colors)
        if self.engine == 'multinest':
            engine = 'MultiNest'
        if self.engine == 'dynesty':
            engine = 'Dynesty'
        temp, temp_e = self.star.temp, self.star.temp_e
        rad, rad_e = self.star.rad, self.star.rad_e
        plx, plx_e = self.star.plx, self.star.plx_e
        lum, lum_e = self.star.lum, self.star.lum_e
        print(colored('\n\t\t####################################', c))
        print(colored('\t\t##          PLACEHOLDER           ##', c))
        print(colored('\t\t####################################', c))
        print(colored('\n\t\t\tAuthor: Jose Vines', c))
        print(colored('\t\t\tStar : ', c), end='')
        print(colored(self.star.starname, c))
        print(colored('\t\t\tEffective temperature : ', c), end='')
        print(colored('{:.3f} +/- {:.3f}'.format(temp, temp_e), c))
        print(colored('\t\t\tStellar radius : ', c), end='')
        print(colored('{:.3f} +/- {:.3f}'.format(rad, rad_e), c))
        print(colored('\t\t\tStellar Luminosity : ', c), end='')
        print(colored('{:.3f} +/- {:.3f}'.format(lum, lum_e), c))
        print(colored('\t\t\tParallax : ', c), end='')
        print(colored('{:.3f} +/- {:.3f}'.format(plx, plx_e), c))
        print(colored('\t\t\tEstimated Av : ', c), end='')
        print(colored('{:.3f}'.format(self.star.Av), c))
        print(colored('\t\t\tSelected engine : ', c), end='')
        print(colored(engine, c))
        print(colored('\t\t\tLive points : ', c), end='')
        print(colored(str(self.live_points), c))
        print(colored('\t\t\tlog Evidence tolerance : ', c), end='')
        print(colored(str(self.dlogz), c))
        print(colored('\t\t\tFree parameters : ', c), end='')
        print(colored(str(self.ndim), c))
        pass

    def end(self, elapsed_time):
        """Display end of run information.

        What is displayed is:
        best fit parameters
        elapsed time
        Spectral type (TODO)
        """
        out = pickle.load(open(self.out_folder + '/multinest_out.pkl', 'rb'))
        theta = sp.zeros(order.shape[0])
        for i, param in enumerate(order):
            if param != 'likelihood':
                theta[i] = out['best_fit'][param]
        # theta = build_params(theta, self.coordinator, self.fixed)
        uncert = []
        # lglk = out['best_fit']['log_likelihood']
        lglk = out['best_fit']['likelihood']
        z, z_err = out['lnZ'], out['lnZerr']
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                _, lo, up = credibility_interval(
                    out['posterior_samples'][param])
                uncert.append([abs(theta[i] - lo), abs(up - theta[i])])
            else:
                uncert.append('fixed')
        print('\t\tFitting finished.')
        print('\t\tBest fit parameters are:')
        for i, p in enumerate(order):
            if p == 'z':
                p = '[Fe/H]'
            print('\t\t' + p, end=' : ')
            print('{:.4f}'.format(theta[i]), end=' ')
            if not self.coordinator[i]:
                print('+ {:.4f}'.format(uncert[i][1]), end=' - ')
                print('{:.4f}'.format(uncert[i][0]))
            else:
                print('fixed')
        print('\t\tLog Likelihood of best fit : ', end='')
        print('{:.3f}'.format(lglk))
        print('\t\tlog Bayesian evidence : ', end='')
        print('{:.3f}'.format(z), end=' +/- ')
        print('{:.3f}'.format(z_err))
        print('\t\tElapsed time : ', end='')
        print(elapsed_time)
        pass
#####################


def log_like(cube, ndim, nparams):
    """Multinest log likelihood wrapper."""
    theta = [cube[i] for i in range(ndim)]
    theta = build_params(theta, coordinator, fixed)
    return log_likelihood(theta, star, interpolators)


def pt_multinest(cube, ndim, nparams):
    """Multinest prior transform."""
    prior_transform_multinest(cube, star, prior_dict, coordinator)


def log_prob(theta, star):
    """Wrap for sed_library.log_probability."""
    return log_probability(theta, star, prior_dict, coordinator, interpolators,
                           fixed)
