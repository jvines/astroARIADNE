# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*elif/ /^\s*def/
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
from scipy.special import erf
from scipy.stats import gaussian_kde
from termcolor import colored


def norm_fit(x, mu, sigma, A):
    """Gaussian function."""
    return A * sp.stats.norm.pdf(x, loc=mu, scale=sigma)


def credibility_interval(post, alpha=1.):
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
    z = erf(alpha / sp.sqrt(2))

    lower_percentile = 100 * (1 - z) / 2
    upper_percentile = 100 * (1 + z) / 2
    low, med, up = sp.percentile(
        post, [lower_percentile, 50, upper_percentile]
    )
    return med, low, up


def display_star_fin(star, c):
    """Display stellar information."""
    temp, temp_e = star.temp, star.temp_e
    rad, rad_e = star.rad, star.rad_e
    plx, plx_e = star.plx, star.plx_e
    lum, lum_e = star.lum, star.lum_e
    print(colored('\t\t\tGaia DR2 ID : {}'.format(star.g_id), c))
    if star.tic:
        print(colored('\t\t\tTIC : {}'.format(star.tic), c))
    if star.kic:
        print(colored('\t\t\tKIC : {}'.format(star.kic), c))
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
    print(colored('\t\t\tMaximum Av : ', c), end='')
    print(colored('{:.3f}'.format(star.Av), c))
    print('')
    pass


def display_star_init(star, c):
    """Display stellar information."""
    print(colored('\n\t\t#####################################', c))
    print(colored('\t\t##             ARIADNE             ##', c))
    print(colored('\t\t#####################################', c))
    print(colored('   spectrAl eneRgy dIstribution', c), end=' ')
    print(colored('bAyesian moDel averagiNg fittEr', c))
    print(colored('\n\t\t\tAuthor : Jose Vines', c))
    print(colored('\t\t\tContact : jose . vines at ug . uchile . cl', c))
    print(colored('\t\t\tStar : ', c), end='')
    print(colored(star.starname, c))
    pass


def display_routine(engine, live_points, dlogz, ndim, bound=None, sample=None,
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
    print(colored('\n\t\t*** EXECUTING MAIN FITTING ROUTINE ***', c))
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
        order = sp.array(['teff', 'logg', 'z', 'norm', 'Av'])
    else:
        order = sp.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
        )
    if engine == 'Bayesian Model Averaging':
        res_dir = out_folder + '/BMA_out.pkl'
    else:
        res_dir = out_folder + '/' + engine + '_out.pkl'
    with closing(open(res_dir, 'rb')) as jar:
        out = pickle.load(jar)

    star = out['star']
    mask = star.filter_mask
    n = int(star.used_filters.sum())
    for filt in star.filter_names[mask]:
        p_ = get_noise_name(filt) + '_noise'
        order = sp.append(order, p_)

    theta = sp.zeros(order.shape[0] - 1 + n)
    for i, param in enumerate(order):
        if param != 'loglike':
            theta[i] = out['best_fit'][param]
        if param == 'inflation':
            for m, fi in enumerate(star.filter_names[mask]):
                _p = get_noise_name(fi) + '_noise'
                theta[i + m] = out['best_fit'][_p]
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
        p2 = p
        if 'noise' in p:
            continue
        if p == 'norm':
            p2 = '(R/D)^2'
            print(colored('\t\t\t' + p2 + ' : ', c), end='')
            print(colored('{:.4e}'.format(theta[i]), c), end=' ')
            if not coordinator[i]:
                print(colored('+ {:.4e} -'.format(uncert[i][1]), c), end=' ')
                print(colored('{:.4e}'.format(uncert[i][0]), c), end=' ')
                samp = out['posterior_samples']['norm']
                _, lo, up = credibility_interval(samp, 3)
                print(colored('[{:.4f}, {:.4f}]'.format(lo, up), c))
            else:
                print(colored('fixed', c))
            rad = out['best_fit']['rad']
            unlo, unhi = out['uncertainties']['rad']
            lo, up = out['confidence_interval']['rad']
            print(colored('\t\t\trad : ', c), end='')
            print(colored('{:.4e}'.format(rad), c), end=' ')
            print(colored('+ {:.4e} -'.format(unhi), c), end=' ')
            print(colored('[{:.4f}, {:.4f}]'.format(lo, up), c))
            print(colored('{:.4e} derived'.format(unlo), c))
        if p == 'z':
            p2 = '[Fe/H]'
        print(colored('\t\t\t' + p2 + ' : ', c), end='')
        print(colored('{:.4f}'.format(theta[i]), c), end=' ')
        if not coordinator[i]:
            print(colored('+ {:.4f} -'.format(uncert[i][1]), c), end=' ')
            print(colored('{:.4f}'.format(uncert[i][0]), c), end=' ')
            samp = out['posterior_samples'][p]
            _, lo, up = credibility_interval(samp, 3)
            print(colored('[{:.4f}, {:.4f}]'.format(lo, up), c))
        else:
            print(colored('fixed', c))

    if not use_norm:
        ad = out['best_fit']['AD']
        unlo, unhi = out['uncertainties']['AD']
        lo, up = out['confidence_interval']['AD']
        print(colored('\t\t\tAngular diameter : ', c), end='')
        print(colored('{:.4f}'.format(ad), c), end=' ')
        print(colored('+ {:.4f} -'.format(unhi), c), end=' ')
        print(colored('{:.4f}'.format(unlo), c), end=' ')
        print(colored('[{:.4f}, {:.4f}]'.format(lo, up), c))

    mass = out['best_fit']['grav_mass']
    unlo, unhi = out['uncertainties']['grav_mass']
    lo, up = out['confidence_interval']['grav_mass']
    print(colored('\t\t\tmass : ', c), end='')
    print(colored('{:.2f}'.format(mass), c), end=' ')
    print(colored('+ {:.2f} -'.format(unhi), c), end=' ')
    print(colored('{:.2f}'.format(unlo), c), end=' ')
    print(colored('[{:.2f}, {:.2f}]'.format(lo, up), c))

    lum = out['best_fit']['lum']
    unlo, unhi = out['uncertainties']['lum']
    lo, up = out['confidence_interval']['lum']
    print(colored('\t\t\tluminosity : ', c), end='')
    print(colored('{:.3f}'.format(lum), c), end=' ')
    print(colored('+ {:.3f} -'.format(unhi), c), end=' ')
    print(colored('{:.3f}'.format(unlo), c), end=' ')
    print(colored('[{:.3f}, {:.3f}]'.format(lo, up), c))

    if engine == 'Bayesian Model Averaging':
        age = out['best_fit']['age']
        unlo, unhi = out['uncertainties']['age']
        lo, up = out['confidence_interval']['age']
        print(colored('\t\t\tage : ', c), end='')
        print(colored('{:.4f}'.format(age), c), end=' ')
        print(colored('+ {:.4f} -'.format(unhi), c), end=' ')
        print(colored('{:.4f}'.format(unlo), c), end=' ')
        print(colored('[{:.4f}, {:.4f}]'.format(lo, up), c))

        miso = out['best_fit']['mass_iso']
        unlo, unhi = out['uncertainties']['mass_iso']
        lo, up = out['confidence_interval']['mass_iso']
        print(colored('\t\t\tmass_iso : ', c), end='')
        print(colored('{:.4f}'.format(miso), c), end=' ')
        print(colored('+ {:.4f} -'.format(unhi), c), end=' ')
        print(colored('{:.4f}'.format(unlo), c), end=' ')
        print(colored('[{:.4f}, {:.4f}]'.format(lo, up), c))

    for i, p in enumerate(order):
        if 'noise' not in p:
            continue
        p_ = 'Excess '
        if 'SDSS' not in p and 'PS1' not in p:
            p1, p2 = p.split('_')
        else:
            p1, p2, p3 = p.split('_')
            p1 += '_' + p2
            p2 = p3
        print(colored('\t\t\t' + p_ + p1 + ' ' + p2 + ' : ', c), end='')
        print(colored('{:.4e}'.format(theta[i]), c), end=' ')
        print(colored('+ {:.4e}'.format(uncert[i][1]), c), end=' ')
        print(colored('- {:.4e}'.format(uncert[i][0]), c), end=' ')
        samp = out['posterior_samples'][p]
        _, lo, up = credibility_interval(samp, 3)
        print(colored('[{:.4e}, {:.4e}]'.format(lo, up), c))

    spt = out['spectral_type']
    print(colored('\t\t\tMamajek Spectral Type : ', c), end='')
    print(colored(spt, c))
    if engine != 'Bayesian Model Averaging':
        # print(colored('\t\t\tLog Likelihood of best fit : ', c), end='')
        # print(colored('{:.3f}'.format(lglk), c))
        print(colored('\t\t\tlog Bayesian evidence : ', c), end='')
        print(colored('{:.3f} +/-'.format(z), c), end=' ')
        print(colored('{:.3f}'.format(z_err), c))
    else:
        probs = out['weights']
        for k in probs.keys():
            text = '\t\t\t{} probability : '.format(k)
            print(colored(text, c), end='')
            print(colored('{:.4f}'.format(probs[k]), c))
    print(colored('\t\t\tElapsed time : ', c), end='')
    print(colored(elapsed_time, c))
    pass


def create_dir(path):
    """Create a directory."""
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory {:s} failed".format(path))
        pass
    else:
        print("Created the directory {:s} ".format(path))
        pass
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


def get_noise_name(filt):
    """Retrieve parameter name for white noise."""
    if filt == 'TYCHO_B_MvB':
        return 'BT'
    if filt == 'TYCHO_V_MvB':
        return 'VT'
    if filt == 'SPITZER_IRAC_36':
        return 'IRAC_36'
    if filt == 'SPITZER_IRAC_45':
        return 'IRAC_45'
    if filt == 'NGTS_I':
        return 'NGTS'
    if filt == 'WISE_RSR_W1':
        return 'W1'
    if filt == 'WISE_RSR_W2':
        return 'W2'
    if 'SDSS' in filt or 'PS1' in filt:
        return filt
    return filt.split('_')[-1]


def out_filler(samp, logdat, param, name, out, fmt='f', fixed=False):
    """Fill up the output file."""
    best = get_max_from_kde(samp)
    if not fixed:
        out['best_fit'][param] = best
        logdat += '{}\t{:.4{f}}\t'.format(name, best, f=fmt)
        _, lo, up = credibility_interval(samp)
        out['uncertainties'][param] = (best - lo, up - best)
        logdat += '{:.4{f}}\t{:.4{f}}\t'.format(up - best, best - lo, f=fmt)
        _, lo, up = credibility_interval(samp, 3)
        out['confidence_interval'][param] = (lo, up)
        logdat += '[{:.4{f}}, {:.4{f}}]\n'.format(lo, up, f=fmt)
    else:
        out['best_fit'][param] = fixed
        out['uncertainties'][param] = sp.nan
        out['confidence_interval'][param] = sp.nan
        logdat += '{}\t{:.4{f}}\t'.format(name, fixed, f=fmt)
        logdat += '(FIXED)\n'
    return logdat


def get_max_from_kde(samp):
    """Get maximum of the given distribution."""
    kde = gaussian_kde(samp)
    xmin = samp.min()
    xmax = samp.max()
    xx = sp.linspace(xmin, xmax, 5000)
    kde = kde(xx)
    best = xx[kde.argmax()]
    return best
