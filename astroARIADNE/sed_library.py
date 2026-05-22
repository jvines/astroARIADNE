"""sed_library contain the model, prior and likelihood to be used."""

import numba as nb
import numpy as np
from extinction import apply
from scipy.special import ndtr, ndtri

from .utils import get_noise_name

# The parameter order and the extinction curve are constant for a given fit
# (they depend only on the filter set, the use_norm flag and the extinction
# law), yet the prior transform and the model are evaluated millions of times
# during nested sampling. Cache them so the per-call cost is just a dict lookup
# instead of rebuilding arrays and re-running the extinction law every time.
_ORDER_CACHE = {}
_EXT_CACHE = {}
_TRANSFORM_CACHE = {}
_GRID_CACHE = {}


@nb.njit(cache=True, nogil=True)
def _searchsorted(arr, x):
    """Binary search matching isochrones' searchsorted: returns (L, eq).

    ``eq`` is True when ``x`` exactly equals a grid node, in which case ``L``
    is that node's index; otherwise ``L`` is the insertion index (upper
    bracket). Ported verbatim so bracket selection is identical.
    """
    N = arr.shape[0]
    L = 0
    R = N - 1
    eq = False
    m = (L + R) // 2
    done = False
    while not done:
        xm = arr[m]
        if xm < x:
            L = m + 1
        elif xm > x:
            R = m - 1
        else:
            L = m
            eq = True
            done = True
        m = (L + R) // 2
        if L > R:
            done = True
    return L, eq


@nb.njit(cache=True, nogil=True)
def _trilinear3d(x0, x1, x2, ii0, ii1, ii2, cube):
    """GIL-free trilinear interpolation over a pre-sliced (n0,n1,n2,nf) cube.

    Replicates isochrones' ``interp_value_3d`` exactly: NaN input or any axis
    out of bounds returns NaN for every filter; otherwise the 8 corner values
    are combined with the same edge weights and accumulation order. Edge
    indices are clamped, which only ever touches the zero-weight corner of an
    exact-node hit (so the result is unchanged) while avoiding out-of-bounds
    reads.
    """
    nf = cube.shape[3]
    out = np.empty(nf, dtype=nb.float64)
    n0 = ii0.shape[0]
    n1 = ii1.shape[0]
    n2 = ii2.shape[0]
    if (x0 != x0 or x1 != x1 or x2 != x2
            or x0 < ii0[0] or x0 > ii0[n0 - 1]
            or x1 < ii1[0] or x1 > ii1[n1 - 1]
            or x2 < ii2[0] or x2 > ii2[n2 - 1]):
        for k in range(nf):
            out[k] = np.nan
        return out

    l0, eq0 = _searchsorted(ii0, x0)
    if eq0:
        i0 = l0
        y0 = 0.0
    else:
        i0 = l0 - 1
        y0 = (x0 - ii0[i0]) / (ii0[l0] - ii0[i0])
    l1, eq1 = _searchsorted(ii1, x1)
    if eq1:
        i1 = l1
        y1 = 0.0
    else:
        i1 = l1 - 1
        y1 = (x1 - ii1[i1]) / (ii1[l1] - ii1[i1])
    l2, eq2 = _searchsorted(ii2, x2)
    if eq2:
        i2 = l2
        y2 = 0.0
    else:
        i2 = l2 - 1
        y2 = (x2 - ii2[i2]) / (ii2[l2] - ii2[i2])

    for k in range(nf):
        out[k] = 0.0
    for e in range(8):
        d0 = (e >> 2) & 1
        d1 = (e >> 1) & 1
        d2 = e & 1
        w = (y0 if d0 else 1.0 - y0) \
            * (y1 if d1 else 1.0 - y1) \
            * (y2 if d2 else 1.0 - y2)
        j0 = i0 + d0 if i0 + d0 < n0 else n0 - 1
        j1 = i1 + d1 if i1 + d1 < n1 else n1 - 1
        j2 = i2 + d2 if i2 + d2 < n2 else n2 - 1
        for k in range(nf):
            out[k] += cube[j0, j1, j2, k] * w
    return out


def _grid_arrays(interp, filts):
    """Return (and cache) the axis arrays and the filter-sliced flux cube.

    The DFInterpolator's per-call cost (``column_index`` lookups, ``icols``
    construction, broadcasting checks) is hoisted here: the cube is sliced to
    the requested filters once per (interpolator, filter set), so the hot path
    is a single nogil numba call. Memoised on object identity to avoid building
    keys on each call; references are held so ids cannot be reused.
    """
    cached = _GRID_CACHE.get(id(interp))
    if (cached is not None and cached[0] is interp and cached[1] is filts):
        return cached[2]
    ii0, ii1, ii2 = interp.index_columns
    icols = np.array([interp.column_index[f] for f in filts])
    cube = np.ascontiguousarray(interp.grid[:, :, :, icols], dtype=np.float64)
    arrays = (
        np.ascontiguousarray(ii0, dtype=np.float64),
        np.ascontiguousarray(ii1, dtype=np.float64),
        np.ascontiguousarray(ii2, dtype=np.float64),
        cube,
    )
    _GRID_CACHE[id(interp)] = (interp, filts, arrays)
    return arrays


def _make_transform(prior):
    """Build a fast unit-cube -> parameter transform for a single prior.

    scipy frozen-distribution ``.ppf`` is extremely slow (tens of us per call,
    and it is called once per free parameter on every nested-sampling
    proposal). For the distributions ARIADNE actually uses we substitute the
    closed-form inverse CDF, which is exact and ~100-1000x faster. Anything
    else (e.g. the callable spline population priors) is used directly, and
    unknown frozen distributions fall back to ``.ppf``.
    """
    # scipy frozen distributions expose ``.dist.name`` and stash their
    # parameters in ``.kwds``.
    if hasattr(prior, 'dist') and hasattr(prior.dist, 'name'):
        name = prior.dist.name
        kwds = prior.kwds
        loc = kwds.get('loc', 0.0)
        scale = kwds.get('scale', 1.0)
        if name == 'uniform':
            return lambda u, loc=loc, scale=scale: loc + u * scale
        if name == 'norm':
            return lambda u, loc=loc, scale=scale: loc + scale * ndtri(u)
        if name == 'truncnorm':
            a = kwds['a']
            b = kwds['b']
            phi_a = ndtr(a)
            span = ndtr(b) - phi_a
            return lambda u, loc=loc, scale=scale, pa=phi_a, span=span: \
                loc + scale * ndtri(pa + u * span)
        return prior.ppf
    # Callable priors (e.g. the InterpolatedUnivariateSpline Teff prior).
    if callable(prior):
        return prior
    return prior.ppf


def _free_param_transforms(prior_dict, coordinator, order):
    """Return (and cache) the per-free-parameter transforms, in order.

    The inputs are the same global objects on every call within a fit, so we
    memoise on object identity and verify with ``is`` rather than building a
    hashable key (e.g. ``tuple(order)``) on each of the ~10^5 calls. Keeping a
    reference to ``prior_dict`` keeps its ``id`` from being reused.
    """
    cached = _TRANSFORM_CACHE.get(id(prior_dict))
    if (cached is not None and cached[0] is prior_dict
            and cached[1] is coordinator and cached[2] is order):
        return cached[3]
    transforms = [
        _make_transform(prior_dict[par])
        for fixed, par in zip(coordinator, order) if not fixed
    ]
    _TRANSFORM_CACHE[id(prior_dict)] = (prior_dict, coordinator, order,
                                        transforms)
    return transforms


def _param_order(filts, use_norm):
    """Return (and cache) the full parameter-name order for a filter set.

    Memoised on ``use_norm`` with an identity check on ``filts`` to avoid
    building ``tuple(filts)`` on every call.
    """
    cached = _ORDER_CACHE.get(use_norm)
    if cached is not None and cached[0] is filts:
        return cached[1]
    if use_norm:
        base = ['teff', 'logg', 'z', 'norm', 'Av']
    else:
        base = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
    order = np.array(base + [get_noise_name(f) + '_noise' for f in filts])
    _ORDER_CACHE[use_norm] = (filts, order)
    return order


def _ext_attenuation_base(wave, av_law):
    """Return (and cache) 10**(-0.4 * A(lambda)) evaluated at Av=1.

    All supported laws are exactly linear in Av for fixed Rv, so the
    attenuation at arbitrary Av is simply this array raised to the power Av.
    Memoised on the law with an identity check on ``wave`` to avoid building
    ``tuple(wave)`` on every call.
    """
    cached = _EXT_CACHE.get(av_law)
    if cached is not None and cached[0] is wave:
        return cached[1]
    Rv = 3.1
    unit_ext = av_law(wave * 1e4, 1.0, Rv)
    base = 10 ** (-0.4 * unit_ext)
    _EXT_CACHE[av_law] = (wave, base)
    return base


def build_params(theta, flux, flux_e, filts, coordinator, fixed, use_norm):
    """Build the parameter vector that goes into the model."""
    params = np.zeros(len(coordinator))
    i = 0
    for j in range(len(coordinator)):
        if coordinator[j]:
            params[j] = fixed[j]
        else:
            params[j] = theta[i]
            i += 1
    return params


def get_interpolated_flux(temp, logg, z, filts, interpolators):
    """Interpolate the grid of fluxes in a given teff, logg and z.

    Parameters
    ----------
    temp: float
        The effective temperature.
    logg: float
        The superficial gravity.
    z: float
        The metallicity.
    filts: str
        The desired filter.

    Returns
    -------
    flux : float
        The interpolated flux at temp, logg, z for filter filt.

    """
    ii0, ii1, ii2, cube = _grid_arrays(interpolators, filts)
    # Grid axes are (logg, teff, z), matching the original DFInterpolator call.
    return _trilinear3d(logg, temp, z, ii0, ii1, ii2, cube)


def model_grid(theta, filts, wave, interpolators, use_norm, av_law):
    """Return the model grid in the selected filters.

    Parameters:
    -----------
    theta : array_like
        The parameters of the fit: teff, logg, z, radius, distance
    star : Star
        The Star object containing all relevant information regarding the star.
    interpolators : dict
        A dictionary with the interpolated grid.
    use_norm : bool
        False for a full fit  (including radius and distance). True to fit
        for a normalization constant instead.

    Returns
    -------
    model : dict
        A dictionary whose keys are the filters and the values are the
        interpolated fluxes

    """
    if use_norm:
        teff, logg, z, norm, Av = theta[:5]
    else:
        teff, logg, z, dist, rad, Av = theta[:6]
        dist *= 4.435e+7  # Transform from pc to solRad

    flux = get_interpolated_flux(teff, logg, z, filts, interpolators)

    # Extinction is linear in Av (fixed Rv), so scale the cached unit curve.
    atten = _ext_attenuation_base(wave, av_law) ** Av
    if use_norm:
        model = flux * atten * norm
    else:
        model = flux * atten * (rad / dist) ** 2
    return model


def get_residuals(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                  av_law):
    """Calculate residuals of the model."""
    model = model_grid(theta, filts, wave, interpolators, use_norm, av_law)
    start = 5 if use_norm else 6
    inflation = theta[start:]
    residuals = flux - model
    errs = np.sqrt(flux_er ** 2 + inflation ** 2)
    return residuals, errs


def log_likelihood(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                   av_law):
    """Calculate log likelihood of the model."""
    res, ers = get_residuals(theta, flux, flux_er, wave,
                             filts, interpolators, use_norm, av_law)

    lnl = fast_loglik(res, ers)

    if not np.isfinite(lnl):
        return -1e300

    return -.5 * lnl


@nb.njit(cache=True)
def fast_loglik(res, ers):
    ers2 = ers ** 2
    c = np.log(2 * np.pi * ers2)
    lnl = (c + (res ** 2 / ers2)).sum()
    return lnl


def prior_transform_dynesty(u, flux, flux_er, filts, prior_dict, coordinator,
                            use_norm):
    """Transform the prior from the unit cube to the parameter space."""
    u2 = np.array(u)
    order = _param_order(filts, use_norm)
    transforms = _free_param_transforms(prior_dict, coordinator, order)
    for i, transform in enumerate(transforms):
        u2[i] = transform(u2[i])
    return u2


def prior_transform_multinest(u, flux, flux_er, filts, prior_dict, coordinator,
                              use_norm):
    """Transform the prior from the unit cube to the parameter space."""
    order = _param_order(filts, use_norm)
    transforms = _free_param_transforms(prior_dict, coordinator, order)
    for i, transform in enumerate(transforms):
        u[i] = transform(u[i])
