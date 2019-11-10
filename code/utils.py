"""Various utilities used throughout the code.

Here go various utilities that don't belong directly in any class,
photometry utils module nor or SED model module.
"""
import os
import pickle
import random
import time
from contextlib import closing

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


def display(engine, star, live_points, dlogz, ndim, bound=None, sample=None,
            nthreads=None, dynamic=None):
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
    print(colored('\t\t\tContact : jose . vines at ug . uchile . cl', c))
    print(colored('\t\t\tStar : ', c), end='')
    print(colored(star.starname, c))
    print(colored('\t\t\tEffective temperature : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(temp, temp_e), c))
    if rad is not None:
        print(colored('\t\t\tStellar radius : ', c), end='')
        print(colored('{:.3f} +/- {:.3f}'.format(rad, rad_e), c))
    if lum is not None:
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
    if engine == 'Dynesty' or engine == 'Bayesian Model Averaging':
        print(colored('\t\t\tBounding : ', c), end='')
        print(colored(bound, c))
        print(colored('\t\t\tSampling : ', c), end='')
        print(colored(sample, c))
        print(colored('\t\t\tN threads : ', c), end='')
        print(colored(nthreads, c))
        if dynamic:
            print(colored('\t\t\tRunning the Dynamic Nested Sampler', c))
    print('')
    pass


def end(coordinator, elapsed_time, out_folder, engine, use_norm):
    """Display end of run information.

    What is displayed is:
    best fit parameters
    elapsed time
    Spectral type
    """
    colors = [
        'red', 'green', 'blue', 'yellow',
        'grey', 'magenta', 'cyan', 'white'
    ]
    c = random.choice(colors)
    if use_norm:
        order = sp.array(['teff', 'logg', 'z', 'norm', 'Av', 'inflation'])
    else:
        order = sp.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'Av', 'inflation']
        )
    if engine == 'Bayesian Model Averaging':
        res_dir = out_folder + '/BMA_out.pkl'
    else:
        res_dir = out_folder + '/' + engine + '_out.pkl'
    with closing(open(res_dir, 'rb')) as jar:
        out = pickle.load(jar)

    theta = sp.zeros(order.shape[0])
    for i, param in enumerate(order):
        if param != 'loglike':
            theta[i] = out['best_fit'][param]
    uncert = []
    if engine != 'Bayesian Model Averaging':
        lglk = out['best_fit']['loglike']
        z, z_err = out['global_lnZ'], out['global_lnZerr']
    for i, param in enumerate(order):
        if not coordinator[i]:
            _, lo, up = credibility_interval(
                out['posterior_samples'][param])
            uncert.append([abs(theta[i] - lo), abs(up - theta[i])])
        else:
            uncert.append('fixed')
    print('')
    print(colored('\t\t\tFitting finished.', c))
    print(colored('\t\t\tBest fit parameters are:', c))
    for i, p in enumerate(order):
        if p == 'norm':
            p = '(R/D)^2'
            print(colored('\t\t\t' + p + ' : ', c), end='')
            print(colored('{:.4e}'.format(theta[i]), c), end=' ')
            if not coordinator[i]:
                print(colored('+ {:.4e} -'.format(uncert[i][1]), c), end=' ')
                print(colored('{:.4e}'.format(uncert[i][0]), c))
            else:
                print(colored('fixed', c))
            samp = out['posterior_samples']['rad']
            rad = out['best_fit']['rad']
            _, lo, up = credibility_interval(samp)
            unlo = abs(rad - lo)
            unhi = abs(rad - up)
            print(colored('\t\t\trad : ', c), end='')
            print(colored('{:.4e}'.format(rad), c), end=' ')
            print(colored('+ {:.4e} -'.format(unhi), c), end=' ')
            print(colored('{:.4e} derived'.format(unlo), c))
        if p == 'z':
            p = '[Fe/H]'
        print(colored('\t\t\t' + p + ' : ', c), end='')
        print(colored('{:.4f}'.format(theta[i]), c), end=' ')
        if not coordinator[i]:
            print(colored('+ {:.4f} -'.format(uncert[i][1]), c), end=' ')
            print(colored('{:.4f}'.format(uncert[i][0]), c))
        else:
            print(colored('fixed', c))
    samp = out['posterior_samples']['mass']
    mass = out['best_fit']['mass']
    _, lo, up = credibility_interval(samp)
    unlo = abs(mass - lo)
    unhi = abs(mass - up)
    print(colored('\t\t\tmass : ', c), end='')
    print(colored('{:.2f}'.format(mass), c), end=' ')
    print(colored('+ {:.2f} -'.format(unhi), c), end=' ')
    print(colored('{:.2f} derived'.format(unlo), c))
    spt = out['spectral_type']
    print(colored('\t\t\tMamajek Spectral Type : ', c), end='')
    print(colored(spt, c))
    if engine != 'Bayesian Model Averaging':
        print(colored('\t\t\tLog Likelihood of best fit : ', c), end='')
        print(colored('{:.3f}'.format(lglk), c))
        print(colored('\t\t\tlog Bayesian evidence : ', c), end='')
        print(colored('{:.3f} +/-'.format(z), c), end=' ')
        print(colored('{:.3f}'.format(z_err), c))
    print(colored('\t\t\tElapsed time : ', c), end='')
    print(colored(elapsed_time, c))
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
                    elapsed = '{:.2f} seconds'.format(seconds)
                else:
                    elapsed = '{:.0f} minutes'.format(minutes)
                    elapsed += ' and {:.2f} seconds'.format(seconds)
            else:
                elapsed = '{:.0f} hours'.format(hours)
                elapsed += ', {:.0f} minutes'.format(minutes)
                elapsed += ' and {:.2f} seconds'.format(seconds)
        else:
            elapsed = '{:.0f} days'.format(days)
            elapsed += ', {:.0f} hours'.format(hours)
            elapsed += ', {:.0f} minutes'.format(minutes)
            elapsed += ' and {:.2f} seconds'.format(seconds)
    else:
        elapsed = '{:.0f} weeks'.format(weeks)
        elapsed += ', {:.0f} days'.format(days)
        elapsed += ', {:.0f} hours'.format(hours)
        elapsed += ', {:.0f} minutes'.format(minutes)
        elapsed += ' and {:.2f} seconds'.format(seconds)
    return elapsed
