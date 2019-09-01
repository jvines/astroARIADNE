"""Estimate logg using MIST isochrones."""

import os

from isochrones import SingleStarModel, get_ichrone

from sed_library import credibility_interval


def estimate(bands, params):
    """Estimate logg using MIST isochrones."""
    mist = get_ichrone('mist', bands=bands)
    model = SingleStarModel(mist, **params)
    model.fit(resume=False, verbose=False)
    logg_samples = model.derived_samples['logg']
    med, lo, up = credibility_interval(logg_samples)
    med_e = 2 * max([med - lo, up - med])
    return med, med_e
