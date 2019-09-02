"""Various utilities used throughout the code.

Here go various utilities that don't belong directly in any class,
photometry utils module nor or SED model module.
"""
import os
import random
import time

import scipy as sp
from termcolor import colored


def credibility_interval(post, alpha=.68):
    """Calculate bayesian credibility interval.

    Parameters:
    -----------
    post : array_like
        The posterior sample over which to calculate the bayesian credibility
        interval.
    alpha : float, optional
        Confidence level.
    Returns:
    --------
    med : float
        Median of the posterior.
    low : float
        Lower part of the credibility interval.
    up : float
        Upper part of the credibility interval.

    """
    er_msg = 'Cannot calculate credibility interval of a single element.'
    assert len(post) > 1, er_msg
    lower_percentile = 100 * (1 - alpha) / 2
    upper_percentile = 100 * (1 + alpha) / 2
    low, med, up = sp.percentile(
        post,
        [lower_percentile, 50, upper_percentile]
    )
    return med, low, up


def display(engine, star, live_points, dlogz, ndim):
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
    if engine == 'multinest':
        engine = 'MultiNest'
    if engine == 'dynesty':
        engine = 'Dynesty'
    temp, temp_e = star.temp, star.temp_e
    rad, rad_e = star.rad, star.rad_e
    plx, plx_e = star.plx, star.plx_e
    lum, lum_e = star.lum, star.lum_e
    print(colored('\n\t\t####################################', c))
    print(colored('\t\t##          PLACEHOLDER           ##', c))
    print(colored('\t\t####################################', c))
    print(colored('\n\t\t\tAuthor: Jose Vines', c))
    print(colored('\t\t\tStar : ', c), end='')
    print(colored(star.starname, c))
    print(colored('\t\t\tEffective temperature : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(temp, temp_e), c))
    print(colored('\t\t\tStellar radius : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(rad, rad_e), c))
    print(colored('\t\t\tStellar Luminosity : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(lum, lum_e), c))
    print(colored('\t\t\tParallax : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(plx, plx_e), c))
    print(colored('\t\t\tEstimated Av : ', c), end='')
    print(colored('{:.3f}'.format(star.Av), c))
    print(colored('\t\t\tSelected engine : ', c), end='')
    print(colored(engine, c))
    print(colored('\t\t\tLive points : ', c), end='')
    print(colored(str(live_points), c))
    print(colored('\t\t\tlog Evidence tolerance : ', c), end='')
    print(colored(str(dlogz), c))
    print(colored('\t\t\tFree parameters : ', c), end='')
    print(colored(str(ndim), c))
    pass


def end(coordinator, elapsed_time, out_folder):
    """Display end of run information.

    What is displayed is:
    best fit parameters
    elapsed time
    Spectral type
    """
    order = sp.array(['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation'])
    out = pickle.load(open(out_folder + '/multinest_out.pkl', 'rb'))
    mamajek_spt = sp.loadtxt(
        '../Datafiles/mamajek_spt.dat', dtype=str, usecols=[0])
    mamajek_temp = sp.loadtxt('../Datafiles/mamajek_spt.dat', usecols=[1])
    theta = sp.zeros(order.shape[0])
    for i, param in enumerate(order):
        if param != 'likelihood':
            theta[i] = out['best_fit'][param]
    uncert = []
    lglk = out['best_fit']['likelihood']
    z, z_err = out['lnZ'], out['lnZerr']
    for i, param in enumerate(order):
        if not coordinator[i]:
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
        if not coordinator[i]:
            print('+ {:.4f}'.format(uncert[i][1]), end=' - ')
            print('{:.4f}'.format(uncert[i][0]))
        else:
            print('fixed')
    spt_idx = sp.argmin(abs(mamajek_temp - theta[0]))
    spt = mamajek_spt[spt_idx]
    print('\t\tMamajek Spectral Type : ', end='')
    print(spt)
    print('\t\tLog Likelihood of best fit : ', end='')
    print('{:.3f}'.format(lglk))
    print('\t\tlog Bayesian evidence : ', end='')
    print('{:.3f}'.format(z), end=' +/- ')
    print('{:.3f}'.format(z_err))
    print('\t\tElapsed time : ', end='')
    print(elapsed_time)
    pass


def create_dir(path):
    """Create a directory."""
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory {:s} failed".format(path))
    else:
        print("Created the directory {:s} ".format(path))
    pass


def execution_time(start):
    """Calculate run execution time."""
    end = time.time() - start
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
