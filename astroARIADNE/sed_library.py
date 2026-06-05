"""sed_library contain the model, prior and likelihood to be used."""

import numba as nb
import numpy as np
from extinction import apply
from scipy.special import ndtr, ndtri

from .config import gridsdir, GRID_BASENAMES
from .utils import get_noise_name


class GridInterpolator:
    """Lightweight atmosphere-grid interpolator backed by an HDF5 cube.

    Replaces isochrones' ``DFInterpolator`` for the model grids: it loads the
    dense flux cube + axes + filter names from a version-stable HDF5 file (no
    pandas, no pickle) and exposes exactly the three attributes the numba
    interpolation path (``_grid_arrays`` / ``_trilinear3d``) reads —
    ``index_columns`` (logg, teff, feh axes), ``grid`` (n_logg, n_teff, n_feh,
    n_filter cube) and ``column_index`` (filter name -> column).

    Also exposes ``finite_bands``: the set of filters the grid actually
    constrains (at least one finite node). Bands beyond the grid's spectral
    reach are NaN-padded at build time (see notebooks/build_model_grids.py), so
    ``finite_bands`` is the grid's usable reach -- it drives IR-excess band
    gating, where only bands every in-play grid reaches are quantified and
    redder bands are shown plot-only.
    """

    def __init__(self, path):
        import h5py
        with h5py.File(path, 'r') as h:
            self.index_columns = (np.asarray(h['logg'][:], dtype=np.float64),
                                  np.asarray(h['teff'][:], dtype=np.float64),
                                  np.asarray(h['feh'][:], dtype=np.float64))
            self.grid = np.asarray(h['flux'][:], dtype=np.float64)
            filts = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in h['filters'][:]]
            self.column_index = {f: i for i, f in enumerate(filts)}
        finite = np.isfinite(self.grid).any(axis=(0, 1, 2))
        self.finite_bands = frozenset(f for f, i in self.column_index.items()
                                      if finite[i])


def load_grid(name):
    """Load an atmosphere grid by name into a ``GridInterpolator``."""
    base = GRID_BASENAMES[name.lower()]
    return GridInterpolator(f'{gridsdir}/{base}.h5')


def grid_irx_reach(grid_name, interp):
    """Bands a grid both physically reaches (finite cube column) AND reliably
    models (within its ``grid_wave_coverage`` cutoff, if any).

    ``finite_bands`` alone is physical coverage; some grids (e.g. Coelho) span
    further than they are trustworthy -- their red flux is extrapolated, capped
    by ``grid_wave_coverage``. IR-excess gating uses this reliability-aware
    reach (the same cutoff that drops those bands from the fit), so the
    quantified IRX set is the intersection of it across the in-play grids.
    """
    from .config import grid_wave_coverage
    from .phot_utils import get_effective_wavelength
    bands = set(interp.finite_bands)
    cov = grid_wave_coverage.get(grid_name.lower())
    if cov is not None:
        bands = {b for b in bands if get_effective_wavelength(b) <= cov}
    return bands

# The parameter order and the extinction curve are constant for a given fit
# (they depend only on the filter set, the use_norm flag and the extinction
# law), yet the prior transform and the model are evaluated millions of times
# during nested sampling. Cache them so the per-call cost is just a dict lookup
# instead of rebuilding arrays and re-running the extinction law every time.
_ORDER_CACHE = {}
_EXT_CACHE = {}
_TRANSFORM_CACHE = {}
_GRID_CACHE = {}
_BINARY_SWAP_CACHE = {}


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
    # scipy frozen distributions expose ``.dist.name``; normalise the shape
    # params, loc and scale via ``_parse_args`` so this works whether the prior
    # was built with keywords (``norm(loc=.., scale=..)``) or positionally
    # (``norm(5500, 120)`` / ``truncnorm(a, b, loc, scale)``). Reading ``.kwds``
    # alone silently dropped positional loc/scale and raised KeyError on a
    # positionally-built truncnorm.
    if hasattr(prior, 'dist') and hasattr(prior.dist, 'name'):
        name = prior.dist.name
        shapes, loc, scale = prior.dist._parse_args(*prior.args, **prior.kwds)
        if name == 'uniform':
            return lambda u, loc=loc, scale=scale: loc + u * scale
        if name == 'norm':
            return lambda u, loc=loc, scale=scale: loc + scale * ndtri(u)
        if name == 'truncnorm':
            a, b = shapes
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


def _param_order(filts, use_norm, binary=False, shared=False):
    """Return (and cache) the full parameter-name order for a filter set.

    Memoised on ``(use_norm, binary, shared)`` with an identity check on
    ``filts`` to avoid building ``tuple(filts)`` on every call.

    For an unresolved-binary fit (``binary=True``, requires ``use_norm=False``)
    a second stellar component is appended -- ``teff2, logg2, rad2`` -- sharing
    the primary's ``z``, ``dist`` and ``Av``. When ``shared=True`` the noise
    model is a single ``shared_noise`` term instead of one per band.
    """
    cached = _ORDER_CACHE.get((use_norm, binary, shared))
    if cached is not None and cached[0] is filts:
        return cached[1]
    if use_norm:
        base = ['teff', 'logg', 'z', 'norm', 'Av']
    else:
        base = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
    if binary:
        base = base + ['teff2', 'logg2', 'rad2']
    noise = ['shared_noise'] if shared else [get_noise_name(f) + '_noise'
                                             for f in filts]
    order = np.array(base + noise)
    _ORDER_CACHE[(use_norm, binary, shared)] = (filts, order)
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


def model_grid(theta, filts, wave, interpolators, use_norm, av_law,
               binary=False):
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
    binary : bool
        If True, sum a second stellar component (teff2, logg2, rad2 in
        ``theta[6:9]``) sharing the primary's z, dist and Av. Requires
        ``use_norm=False``.

    Returns
    -------
    model : dict
        A dictionary whose keys are the filters and the values are the
        interpolated fluxes

    """
    if use_norm:
        teff, logg, z, norm, Av = theta[:5]
    elif binary:
        teff, logg, z, dist, rad, Av, teff2, logg2, rad2 = theta[:9]
        dist *= 4.435e+7  # Transform from pc to solRad
    else:
        teff, logg, z, dist, rad, Av = theta[:6]
        dist *= 4.435e+7  # Transform from pc to solRad

    flux = get_interpolated_flux(teff, logg, z, filts, interpolators)

    # Extinction is linear in Av (fixed Rv), so scale the cached unit curve.
    atten = _ext_attenuation_base(wave, av_law) ** Av
    if use_norm:
        model = flux * atten * norm
    elif binary:
        # Two components share distance and extinction; the secondary uses the
        # primary's z (co-natal). Both luminosities scale as (R/d)^2.
        flux2 = get_interpolated_flux(teff2, logg2, z, filts, interpolators)
        model = (flux * (rad / dist) ** 2
                 + flux2 * (rad2 / dist) ** 2) * atten
    else:
        model = flux * atten * (rad / dist) ** 2
    return model


def get_residuals(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                  av_law, binary=False):
    """Calculate residuals of the model."""
    model = model_grid(theta, filts, wave, interpolators, use_norm, av_law,
                       binary=binary)
    start = 9 if binary else (5 if use_norm else 6)
    inflation = theta[start:]
    residuals = flux - model
    errs = np.sqrt(flux_er ** 2 + inflation ** 2)
    return residuals, errs


def contrast_loglike(theta, interpolators, contrast):
    """Gaussian log-likelihood of the binary component magnitude difference(s).

    ``contrast`` is ``(cfilts, dmags, sigmas)`` of equal length: ``cfilts`` a
    stable array of ARIADNE filter names where a delta-magnitude was measured
    (speckle/AO/Gaia), ``dmags`` the observed |m2 - m1| in each, ``sigmas`` the
    uncertainties. The model delta-mag in a band is
    ``2.5 |log10[(rad2/rad1)^2 * F2/F1]|`` -- distance and (shared) extinction
    cancel in the flux ratio. ``abs`` makes it label-independent: the constraint
    is on the brighter/fainter contrast, while teff ordering is handled in the
    prior transform.
    """
    cfilts, dmags, sigmas = contrast
    teff1, logg1, z = theta[0], theta[1], theta[2]
    rad1 = theta[4]
    teff2, logg2, rad2 = theta[6], theta[7], theta[8]
    f1 = get_interpolated_flux(teff1, logg1, z, cfilts, interpolators)
    f2 = get_interpolated_flux(teff2, logg2, z, cfilts, interpolators)
    ratio = (rad2 / rad1) ** 2 * (f2 / f1)
    dmag_model = 2.5 * np.abs(np.log10(ratio))
    return -0.5 * (((dmag_model - dmags) / sigmas) ** 2).sum()


def log_likelihood(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                   av_law, binary=False, contrast=None):
    """Calculate log likelihood of the model."""
    res, ers = get_residuals(theta, flux, flux_er, wave,
                             filts, interpolators, use_norm, av_law,
                             binary=binary)

    lnl = fast_loglik(res, ers)

    if not np.isfinite(lnl):
        return -1e300

    ll = -.5 * lnl
    if binary and contrast is not None:
        c = contrast_loglike(theta, interpolators, contrast)
        if not np.isfinite(c):
            return -1e300
        ll += c
    return ll


@nb.njit(cache=True)
def fast_loglik(res, ers):
    ers2 = ers ** 2
    c = np.log(2 * np.pi * ers2)
    lnl = (c + (res ** 2 / ers2)).sum()
    return lnl


def _binary_swap_indices(order, coordinator):
    """Free-parameter indices of (teff, teff2, logg, logg2, rad, rad2).

    Returns the positions in the free-parameter vector so the binary prior
    transform can enforce teff1 >= teff2 by swapping the two component blocks.
    ``None`` if any of the six is fixed (then ordering is not enforced).
    Cached on the (order, coordinator) identity.
    """
    cached = _BINARY_SWAP_CACHE.get(id(order))
    if (cached is not None and cached[0] is order
            and cached[1] is coordinator):
        return cached[2]
    free = [par for fixed, par in zip(coordinator, order) if not fixed]
    try:
        idx = (free.index('teff'), free.index('teff2'),
               free.index('logg'), free.index('logg2'),
               free.index('rad'), free.index('rad2'))
    except ValueError:
        idx = None
    _BINARY_SWAP_CACHE[id(order)] = (order, coordinator, idx)
    return idx


def prior_transform_dynesty(u, flux, flux_er, filts, prior_dict, coordinator,
                            use_norm, binary=False, shared=False):
    """Transform the prior from the unit cube to the parameter space."""
    u2 = np.array(u)
    # Derive the shared-jitter flag from prior_dict so it is robust to the
    # module global not propagating into spawned pool workers — prior_dict
    # always carries 'shared_noise' exactly when the noise model is shared.
    shared = shared or ('shared_noise' in prior_dict)
    order = _param_order(filts, use_norm, binary, shared)
    transforms = _free_param_transforms(prior_dict, coordinator, order)
    for i, transform in enumerate(transforms):
        u2[i] = transform(u2[i])
    if binary:
        # Enforce teff1 >= teff2 by folding the prior (swap the two component
        # blocks when out of order). This imposes the ordered prior at sampling
        # time so dynesty never sees the label-switching bimodality, keeping the
        # evidence (the BMA's currency) clean. Requires teff1/teff2, logg1/logg2
        # and rad1/rad2 to share priors, which _default_priors guarantees.
        idx = _binary_swap_indices(order, coordinator)
        if idx is not None and u2[idx[1]] > u2[idx[0]]:
            it, it2, ig, ig2, ir, ir2 = idx
            u2[it], u2[it2] = u2[it2], u2[it]
            u2[ig], u2[ig2] = u2[ig2], u2[ig]
            u2[ir], u2[ir2] = u2[ir2], u2[ir]
    return u2
