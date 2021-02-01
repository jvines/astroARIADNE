"""Various utilities used throughout the code.

Here go various utilities that don't belong directly in any class,
photometry utils module nor or SED model module.
"""
import os
import pickle
import random
import time
from contextlib import closing

import numpy as np
from scipy.special import erf
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d
from termcolor import colored


def sample_from_distribution(distribution, size=100):
    """Take random samples from an empirical distribution.

    Parameters
    ----------
    distribution: array_like
        The empirical distribution from which we want to sample.
    size: int, optional
        The number of samples we wish to extract.

    Returns
    -------
    samples: array_like
        The array with the random samples.
    """
    # First we calculate the CDF of the distribution
    cdf = estimate_cdf(distribution)
    # Now we interpolate the inverted cdf
    xx = np.linspace(np.min(distribution), np.max(distribution), 500)
    icdf = interp1d(cdf, xx)
    points = np.random.random_sample(size=size)
    return icdf(points)


def estimate_pdf(distribution):
    """Estimates the PDF of a distribution using a gaussian KDE.

    Parameters
    ----------
    distribution: array_like
        The distribution.
    Returns
    -------
    xx: array_like
        The x values of the PDF.
    pdf: array_like
        The estimated PDF.
    """
    kde = gaussian_kde(distribution)
    xmin, xmax = distribution.min(), distribution.max()
    xx = np.linspace(xmin, xmax, 500)
    pdf = kde(xx)
    return xx, pdf


def estimate_cdf(distribution, hdr=False):
    """Estimate the CDF of a distribution."""
    h, hx = np.histogram(distribution, density=True, bins=499)
    cdf = np.zeros(500)  # ensure the first value of the CDF is 0
    if hdr:
        idx = np.argsort(h)[::-1]
        cdf[1:] = np.cumsum(h[idx]) * np.diff(hx)
    else:
        cdf[1:] = np.cumsum(h) * np.diff(hx)
    return cdf


def norm_fit(x, mu, sigma, A):
    """Gaussian function."""
    return A * norm.pdf(x, loc=mu, scale=sigma)


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
    z = erf(alpha / np.sqrt(2))

    lower_percentile = 100 * (1 - z) / 2
    upper_percentile = 100 * (1 + z) / 2
    low, med, up = np.percentile(
        post, [lower_percentile, 50, upper_percentile]
    )
    return med, low, up


def credibility_interval_hdr(xx, pdf, cdf, sigma=1.):
    """Calculate the highest density region for an empirical distribution.

    Reference: Hyndman, Rob J. 1996

    Parameters
    ----------
    xx: array_like
        The x values of the PDF (and the y values of the CDF).
    pdf: array_like
        The PDF of the distribution.
    cdf: array_like
        The CDF of the distribution.
    sigma: float
        The confidence level in sigma notation. (e.g. 1 sigma = 68%)

    Returns
    -------
    best: float
        The value corresponding to the peak of the posterior distribution.
    low: float
        The minimum value of the HDR.
    high: float
        The maximum value of the HDR.

    Note: The HDR is capable of calculating more robust credible regions
    for multimodal distributions. It is identical to the usual probability
    regions of symmetric about the mean distributions. Using this then should
    lead to more realistic errorbars and 3-sigma intervals for multimodal
    outputs.

    """
    # Get best fit value
    best = xx[np.argmax(pdf)]
    z = erf(sigma / np.sqrt(2))
    # Sort the pdf in reverse order
    idx = np.argsort(pdf)[::-1]
    # Find where the CDF reaches 100*z%
    idx_hdr = np.where(cdf >= z)[0][0]
    # Isolate the HDR
    hdr = pdf[idx][:idx_hdr]
    # Get the minimum density
    hdr_min = hdr.min()
    # Get CI
    low = xx[pdf > hdr_min].min()
    high = xx[pdf > hdr_min].max()
    return best, low, high


def display_star_fin(star, c):
    """Display stellar information."""
    temp, temp_e = star.temp, star.temp_e
    rad, rad_e = star.rad, star.rad_e
    plx, plx_e = star.plx, star.plx_e
    lum, lum_e = star.lum, star.lum_e
    dist, dist_e = star.dist, star.dist_e
    print(colored('\t\t\tGaia DR2 ID : {}'.format(star.g_id), c))
    if star.tic:
        print(colored('\t\t\tTIC : {}'.format(star.tic), c))
    if star.kic:
        print(colored('\t\t\tKIC : {}'.format(star.kic), c))
    print(colored('\t\t\tGaia Effective temperature : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(temp, temp_e), c))
    if rad is not None:
        print(colored('\t\t\tGaia Stellar radius : ', c), end='')
        print(colored('{:.3f} +/- {:.3f}'.format(rad, rad_e), c))
    if lum is not None:
        print(colored('\t\t\tGaia Stellar Luminosity : ', c), end='')
        print(colored('{:.3f} +/- {:.3f}'.format(lum, lum_e), c))
    print(colored('\t\t\tGaia Parallax : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(plx, plx_e), c))
    print(colored('\t\t\tBailer-Jones distance : ', c), end='')
    print(colored('{:.3f} +/- {:.3f}'.format(dist, dist_e), c))
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
        order = np.array(['teff', 'logg', 'z', 'norm', 'rad', 'Av'])
    else:
        order = np.array(
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
        order = np.append(order, p_)

    theta = np.zeros(order.shape[0] - 1 + n)
    for i, param in enumerate(order):
        if param != 'loglike':
            theta[i] = out['best_fit'][param]
        if param == 'inflation':
            for m, fi in enumerate(star.filter_names[mask]):
                _p = get_noise_name(fi) + '_noise'
                theta[i + m] = out['best_fit'][_p]

    if engine != 'Bayesian Model Averaging':
        z, z_err = out['global_lnZ'], out['global_lnZerr']

    print('')
    print(colored('\t\t\tFitting finished.', c))
    print(colored('\t\t\tBest fit parameters are:', c))
    fmt_str = ''
    for i, p in enumerate(order):
        p2 = p
        if 'noise' in p:
            continue
        fmt_str += '\t\t\t'
        fmt = 'f'
        if p == 'norm':
            p2 = '(R/D)^2'
            fmt = 'e'
        if p == 'z':
            p2 = '[Fe/H]'
        fmt_str += f'{p2} : {theta[i]:.4{fmt}} '
        if not coordinator[i]:
            unlo, unhi = out['uncertainties'][p]
            lo, up = out['confidence_interval'][p]
            fmt_str += f'+ {unhi:.4{fmt}} - {unlo:.4{fmt}} '
            fmt_str += f'[{lo:.4{fmt}}, {up:.4{fmt}}]\n'
        else:
            fmt_str += 'fixed\n'

    if not use_norm:
        ad = out['best_fit']['AD']
        unlo, unhi = out['uncertainties']['AD']
        lo, up = out['confidence_interval']['AD']
        fmt_str += f'\t\t\tAngular Diameter : {ad:.4f} '
        fmt_str += f'+ {unhi:.4f} - {unlo:.4f} [{lo:.4f}, {up:.4f}]\n'

    mass = out['best_fit']['grav_mass']
    unlo, unhi = out['uncertainties']['grav_mass']
    lo, up = out['confidence_interval']['grav_mass']
    fmt_str += f'\t\t\tGrav mass : {mass:.4f} '
    fmt_str += f'+ {unhi:.4f} - {unlo:.4f} [{lo:.4f}, {up:.4f}]\n'

    lum = out['best_fit']['lum']
    unlo, unhi = out['uncertainties']['lum']
    lo, up = out['confidence_interval']['lum']
    fmt_str += f'\t\t\tLuminosity : {lum:.4f} '
    fmt_str += f'+ {unhi:.4f} - {unlo:.4f} [{lo:.4f}, {up:.4f}]\n'

    if engine == 'Bayesian Model Averaging':
        miso = out['best_fit']['iso_mass']
        unlo, unhi = out['uncertainties']['iso_mass']
        lo, up = out['confidence_interval']['iso_mass']
        fmt_str += f'\t\t\tIso mass : {miso:.4f} '
        fmt_str += f'+ {unhi:.4f} - {unlo:.4f} [{lo:.4f}, {up:.4f}]\n'

        age = out['best_fit']['age']
        unlo, unhi = out['uncertainties']['age']
        lo, up = out['confidence_interval']['age']
        fmt_str += f'\t\t\tAge (Gyr) : {age:.4f} '
        fmt_str += f'+ {unhi:.4f} - {unlo:.4f} [{lo:.4f}, {up:.4f}]\n'

        eep = out['best_fit']['eep']
        unlo, unhi = out['uncertainties']['eep']
        lo, up = out['confidence_interval']['eep']
        fmt_str += f'\t\t\tEEP : {eep:.4f} '
        fmt_str += f'+ {unhi:.4f} - {unlo:.4f} [{lo:.4f}, {up:.4f}]\n'

    for i, p in enumerate(order):
        if 'noise' not in p:
            continue
        unlo, unhi = out['uncertainties'][p]
        lo, up = out['confidence_interval'][p]
        p_ = 'Excess '
        if 'SDSS' not in p and 'PS1' not in p:
            p1, p2 = p.split('_')
        else:
            p1, p2, p3 = p.split('_')
            p1 += '_' + p2
            p2 = p3
        fmt_str += f'\t\t\t{p_ + p1} {p2} : {theta[i]:.4f} '
        fmt_str += f'{unhi:.4f} - {unlo:.4f} [{lo:.4f}, {up:.4f}]\n'
    print(colored(fmt_str, c), end='')

    spt = out['spectral_type']
    print(colored('\t\t\tMamajek Spectral Type : ', c), end='')
    print(colored(spt, c))
    if engine != 'Bayesian Model Averaging':
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
        err_msg = f"Creation of the directory {path:s} failed. "
        err_msg += "It might already exist"
        print(err_msg)
        pass
    else:
        print(f"Created the directory {path:s} ")
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
        return 'IRAC 36'
    if filt == 'SPITZER_IRAC_45':
        return 'IRAC 45'
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
    if fixed is False:
        xx, pdf = estimate_pdf(samp)
        cdf = estimate_cdf(samp, hdr=True)
        best, lo, up = credibility_interval_hdr(xx, pdf, cdf, sigma=1)
        # best = get_max_from_kde(samp)
        out['best_fit'][param] = best
        logdat += '{}\t{:.4{f}}\t'.format(name, best, f=fmt)
        # _, lo, up = credibility_interval(samp)
        out['uncertainties'][param] = (best - lo, up - best)
        logdat += '{:.4{f}}\t{:.4{f}}\t'.format(up - best, best - lo, f=fmt)
        _, lo, up = credibility_interval_hdr(xx, pdf, cdf, sigma=3)
        out['confidence_interval'][param] = (lo, up)
        logdat += '{:.4{f}}\t{:.4{f}}\n'.format(lo, up, f=fmt)
    else:
        out['best_fit'][param] = fixed
        out['uncertainties'][param] = np.nan
        out['confidence_interval'][param] = np.nan
        logdat += '{}\t{:.4{f}}\t'.format(name, fixed, f=fmt)
        logdat += '(FIXED)\n'
    return logdat


def get_max_from_kde(samp):
    """Get maximum of the given distribution."""
    raise DeprecationWarning()
    kde = gaussian_kde(samp)
    xmin = samp.min()
    xmax = samp.max()
    xx = np.linspace(xmin, xmax, 1000)
    kde = kde(xx)
    best = xx[kde.argmax()]
    return best
