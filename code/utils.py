"""Various utilities used throughout the code.

Here go various utilities that don't belong directly in any class,
photometry utils module nor or SED model module.
"""
import scipy as sp


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
