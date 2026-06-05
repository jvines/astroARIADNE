"""Main driver of the fitting routine."""

__all__ = ['Fitter', 'dynesty_log_like', 'dynesty_loglike_bma', 'pt_dynesty']

import copy
import logging
import os
import pickle
import time
import warnings

logger = logging.getLogger(__name__)
from multiprocessing import (Pool, set_start_method)
from tqdm import tqdm

import extinction
import astropy.units as u
import numpy as np
import scipy.stats as st
from numpy.random import choice
from astropy.constants import sigma_sb
from termcolor import colored

from .config import (filesdir, gridsdir, priorsdir, filter_names, colors,
                     iso_mask, iso_bands, grid_wave_coverage)
from .error import *
from .isochrone import estimate
from .phot_utils import *
from .sed_library import *
from .utils import *

try:
    import dynesty
    from dynesty.utils import resample_equal

    bma_flag = True
except ModuleNotFoundError:
    wrn = 'Dynesty package not found.\n'
    wrn += 'Install dynesty with `pip install dynesty`'
    warnings.warn(wrn)
    bma_flag = False

set_start_method('fork', force=True)

# Module-level globals consumed by the dynesty wrappers; set per fit in
# Fitter.initialize. Defaults keep single-star fits free of binary state.
binary = False
contrast = None
# True when the noise model is a single shared jitter term (jitter='shared'),
# so the prior transform builds one 'shared_noise' instead of per-band terms.
jitter_shared = False


def _fm07_rv(wave, a_v, r_v=3.1):
    """Fitzpatrick & Massa (2007) extinction with the ARIADNE call signature.

    ``extinction.fm07`` is fixed at Rv=3.1 and takes ``(wave, a_v, unit)`` rather
    than ``(wave, a_v, r_v)``. ARIADNE always calls laws as ``law(wave, Av, Rv)``,
    so this wrapper absorbs (and ignores) ``r_v`` and forwards to ``fm07``. It is
    a single module-level function on purpose: ``sed_library._ext_attenuation_base``
    memoises on the law object's identity, so a stable callable is required for
    the cache to hit.
    """
    return extinction.fm07(wave, a_v)


class Fitter:
    """The Fitter class handles the fitting routines and parameter estimation.

    Examples
    --------
    The fitter isn't instantiaded with any arguments, rather you instantiate a
    Fitter object, then you set up the configurations and finally you
    initialize the object by running the initialize method.
    >>> f = Fitter()
    >>> f.star = s  # s must be a valid Star object.
    >>> f.initialize()

    Attributes
    ----------
    star : Star
        The ``Star`` object to fit.
    setup : list
        Sampler configuration. The first element selects the engine
        (``'dynesty'``); the remaining elements are
        ``[nlive, dlogz, bound, sample, threads, dynamic, walks]``.
    out_folder : str
        Folder where results and plots are written. Defaults to the star name.
    verbose : bool
        Program verbosity. Default ``True``.
    norm : bool
        If ``True``, fit a single normalisation constant instead of radius and
        distance; the radius is recovered afterwards from the Gaia parallax.
    grid : str
        Atmosphere grid for a single-grid fit (e.g. ``'phoenix'``,
        ``'btsettl'``, ``'ck04'``, ``'kurucz'``, ``'coelho'``, ``'bosz'``,
        ``'sphinx'``, ``'tlusty'``, ``'newera'``, ``'koester'``). ``'koester'``
        is a DA white-dwarf grid (standalone; pins [Fe/H]=0 and uses uniform
        WD Teff/logg priors).
    bma : bool
        If ``True``, run Bayesian Model Averaging across several grids. This
        loads every model interpolator and fits all of them, so it is slower.
    models : list
        The grids to average over when ``bma`` is ``True``.
    sequential : bool
        If ``True``, fit the BMA grids one after another. See also
        ``n_grid_jobs`` to fit grids concurrently. Default ``True``.
    n_grid_jobs : int
        Number of BMA grids fit concurrently, one core each. Default ``1``
        (sequential). Each job loads its own grid, so size this to the
        machine's core count and memory.
    n_samples : int or None
        Number of samples drawn from the BMA-averaged posterior. ``None`` uses
        the combined chain length.
    walks : int
        Number of ``rwalk`` MCMC steps per dynesty proposal. Default ``25``;
        also settable as ``setup[7]``.
    av_law : str
        Extinction law. Default ``'fitzpatrick'``.
    estimate_logg : bool
        If ``True``, treat log g as a derived quantity via the isochrone step.
        Default ``False``.
    estimate_age : bool
        If ``True``, run the MIST isochrone fit for age, mass and EEP after the
        SED fit. Default ``True``.
    isochrone_dlogz : float
        Evidence tolerance for the MIST isochrone age/mass fit (the dominant
        cost of a BMA run). Lower is more precise but slower. Default ``0.01``.
    prior_setup : dict or None
        Manual prior specification consumed by ``create_priors_from_setup``.
    experimental : bool
        Enable experimental code paths. Default ``False``.

    """

    colors = colors

    def __init__(self):

        # Default values for attributes
        self._interpolators = []
        self._grids = []
        self.out_folder = None
        self.verbose = True
        self.star = None
        self.setup = ['dynesty']
        self.norm = False
        self.grid = 'phoenix'
        self.estimate_logg = False
        self.estimate_age = True
        # Excess-noise (jitter) model. ``True`` (default): one jitter term per
        # band — the most flexible buffer for underestimated/heterogeneous
        # catalogue errors and model inadequacy, but N nuisance dimensions.
        # ``'shared'``: a single jitter term broadcast across all bands — keeps
        # the buffer (so the likelihood stays smooth) while collapsing those N
        # dimensions to 1. ``False``: fix all jitter to 0 (catalogue errors
        # only) — NOT recommended: removing the buffer makes the likelihood
        # stiff (slower, not faster) and biases the stellar parameters.
        self.jitter = True
        self.av_law = 'fitzpatrick'
        self.n_samples = None
        self.bma = False
        # Default BMA ensemble (override via ``f.models``). Listing every grid
        # is safe: ``initialize`` Teff-gates each one to its valid regime, so a
        # grid inappropriate for the star is skipped automatically (e.g. SPHINX
        # only contributes below 4000 K, CK04/Kurucz only above 4000 K,
        # BT-NextGen/BT-Cond only below 4000 K). Coelho (optical-only) and
        # TLUSTY (hot-star, needs its own uniform prior) are deliberately left
        # out — they are standalone-only and must be requested explicitly.
        self.models = ['phoenix', 'btsettl', 'btnextgen', 'btcond',
                       'kurucz', 'ck04', 'bosz', 'sphinx']
        self.prior_setup = None
        self.sequential = True
        self.experimental = False
        # Unresolved-binary / SB2 mode (opt-in, off by default). When True the
        # forward model sums two stellar components sharing distance, extinction
        # and metallicity (teff2/logg2/rad2 added). REQUIRES a flux-ratio
        # constraint -- ``contrast`` is a list of (filter_name, delta_mag,
        # sigma) tuples (|m2 - m1| in an ARIADNE filter) -- or the secondary is
        # degenerate. Use ``add_contrast`` or set ``contrast`` directly; the
        # binary gate in initialize enforces no-norm + a parallax + >=1 contrast.
        self.binary = False
        self.contrast = None
        # Number of MCMC steps per proposal for dynesty's rwalk sampler.
        self.walks = 25
        # Number of BMA model grids to fit concurrently, each on a single
        # core. 1 (default) keeps the original sequential behaviour. Set this
        # to len(models) to run every grid at once. CAUTION: this spawns
        # ``n_grid_jobs`` processes, so choose a value your machine's core
        # count and memory can handle (each grid loads its own model grid).
        self.n_grid_jobs = 1
        # Evidence tolerance for the MIST isochrone age/mass nested sampling
        # (the dominant cost of a BMA run). Lower is more precise but slower;
        # the isochrone fit itself is parallelised across ``threads``.
        self.isochrone_dlogz = 0.01
        # Activate the typo guard now that all default attributes exist. After
        # this, assigning a *new* public attribute (e.g. ``f.bmaa = True`` for
        # ``f.bma``) raises instead of silently creating a dead attribute and
        # running the wrong kind of fit. See ``__setattr__``.
        self._guard_active = True

    # Public attributes set after __init__ (by initialize(), the fit methods
    # and the BMA worker). Allow-listed so the typo guard does not reject
    # ARIADNE's own internal writes.
    _ALLOWED_DYNAMIC = frozenset({
        'coordinator', 'default_priors', 'fixed', 'ndim', 'out',
        'prior_sources', 'priors', 'sampler', 'start', 'binary_comparison',
    })

    def __setattr__(self, name, value):
        """Guard against silently mistyped config attributes.

        Private (``_``-prefixed) names, declared ``@property`` setters, the
        public attributes created in ``__init__``, and the known post-init
        attributes (``_ALLOWED_DYNAMIC``) are all allowed. A *new* public name
        assigned after construction (e.g. ``f.bmaa`` for ``f.bma``) raises
        ``AttributeError`` rather than becoming a dead attribute. The guard is
        inactive until ``_guard_active`` is set at the end of ``__init__`` so
        construction, copying and unpickling (which start from a fresh
        ``__dict__``) are unaffected.
        """
        if (name.startswith('_')
                or not self.__dict__.get('_guard_active', False)
                or name in self.__dict__
                or name in self._ALLOWED_DYNAMIC
                or isinstance(getattr(type(self), name, None), property)):
            super().__setattr__(name, value)
            return
        raise AttributeError(
            f"{type(self).__name__!r} has no configuration attribute "
            f"{name!r}. Check for a typo (e.g. 'bma' not 'bmaa'). Valid "
            f"config options: av_law, binary, bma, contrast, estimate_age, "
            f"estimate_logg, experimental, grid, isochrone_dlogz, jitter, "
            f"models, n_grid_jobs, n_samples, norm, out_folder, prior_setup, "
            f"sequential, setup, star, verbose, walks."
        )

    @property
    def star(self):
        """Star to fit for."""
        return self._star

    @star.setter
    def star(self, star):
        self._star = star

    @property
    def setup(self):
        """Set up options."""
        return self._setup

    @setup.setter
    def setup(self, setup):
        err_msg = 'The setup has to contain at least the fitting engine'
        err_msg += f', which must be dynesty.\nThe setup was {setup}'
        if len(setup) < 1:
            InputError(err_msg).__raise__()
        self._setup = setup
        self._engine = setup[0]
        defaults = False
        if len(setup) == 1:
            defaults = True
        if self._engine == 'multinest':
            InputError(
                "The 'multinest' engine has been removed (deprecated since "
                "1.5.0). Use engine='dynesty' (the default), which also "
                "supports Bayesian Model Averaging."
            ).__raise__()
        if self._engine == 'dynesty':
            if defaults:
                self._nlive = 500
                self._dlogz = 0.5
                self._bound = 'multi'
                self._sample = 'rwalk'
                self._threads = 1
                self._dynamic = False
            else:
                self._nlive = setup[1]
                self._dlogz = setup[2]
                self._bound = setup[3]
                self._sample = setup[4]
                self._threads = setup[5]
                self._dynamic = setup[6]
                # Optional 8th element: number of rwalk MCMC steps per
                # proposal. Omitted -> keep the current value (default 25),
                # so existing 7-element setups are unaffected.
                if len(setup) > 7:
                    self.walks = setup[7]

    @property
    def norm(self):
        """Bool to decide if a normalization constant will be fitted.

        Set as True to not fit for radius and distance and fit for a
        normalization constant. After the fit a radius is calculated using
        the Gaia parallax.
        """
        return self._norm

    @norm.setter
    def norm(self, norm):
        if type(norm) is not bool:
            InputError('norm must be True or False.').__raise__()
        self._norm = norm

    @property
    def grid(self):
        """Model grid selected."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        assert type(grid) == str
        self._grid = grid
        self._interpolator = load_grid(grid)

    @property
    def bma(self):
        """Bayesian Model Averaging (BMA).

        Set to True if BMA is wanted. This loads every model grid interpolator
        and fits an SED to all of them, so the runtime will be slower!
        """
        return self._bma

    @bma.setter
    def bma(self, bma):
        self._bma = bma if bma_flag else False

    @property
    def models(self):
        """Models to be used in BMA.

        Defaults to the full Teff-gated ensemble
        ``['phoenix', 'btsettl', 'btnextgen', 'btcond', 'kurucz', 'ck04',
        'bosz', 'sphinx']``; ``initialize`` skips any grid outside its valid
        temperature range for the target. Assign a list to override.
        """
        return self._bma_models

    @models.setter
    def models(self, mods):
        self._bma_models = mods

    @property
    def sequential(self):
        """Set to True to make BMA sequentially instead of parallel."""
        return self._sequential

    @sequential.setter
    def sequential(self, sequential):
        self._sequential = sequential

    @property
    def n_samples(self):
        """Set number of samples for BMA."""
        return self._nsamp

    @n_samples.setter
    def n_samples(self, nsamp):
        self._nsamp = nsamp

    @property
    def verbose(self):
        """Program verbosity. Default is True."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if type(verbose) is not bool:
            InputError('Verbose must be True or False.').__raise__()
        self._verbose = verbose

    @property
    def out_folder(self):
        """Output folder.

        If none is provided the default will be the starname.
        """
        return self._out_folder

    @out_folder.setter
    def out_folder(self, out_folder):
        if type(out_folder) is not str and out_folder is not None:
            err_msg = 'Output folder must be an address or None.'
            InputError(err_msg).__raise__()
        self._out_folder = out_folder

    @property
    def av_law(self):
        """Select extinction law."""
        return self._av_law

    @av_law.setter
    def av_law(self, law):
        laws = [
            'cardelli',
            'odonnell',
            'calzetti',
            'fitzpatrick',
            'fm07'
        ]
        law = law.lower()
        if law not in laws:
            err_msg = f'Extinction law {law} not recognized.'
            err_msg += 'Available extinction  laws are: `cardelli`, `odonnell`'
            err_msg += ', `calzetti`, `fitzpatrick`, and `fm07`'
            InputError(err_msg).__raise__()
        law_f = None
        if law == laws[0]:
            law_f = extinction.ccm89
        if law == laws[1]:
            law_f = extinction.odonnell94
        if law == laws[2]:
            law_f = extinction.calzetti00
        if law == laws[3]:
            law_f = extinction.fitzpatrick99
        if law == laws[4]:
            law_f = _fm07_rv  # fixed Rv=3.1; see module-level wrapper
        self._av_law = law_f

    def _apply_grid_coverage(self):
        """Drop bands outside the current grid's native wavelength coverage.

        Some grids (e.g. Coelho 2014) only cover the optical; their redder
        photometry was extrapolated when the grid was built and is unreliable.
        For a standalone fit we therefore exclude those bands. The user's Star
        is left untouched: a shallow copy with a restricted ``used_filters`` /
        ``filter_mask`` is substituted, so every downstream consumer
        (dimension count, priors, likelihood, output) sees the reduced set.
        """
        cov = grid_wave_coverage.get(self._grid.lower())
        if cov is None:
            return
        used = self.star.used_filters == 1
        outside = used & (self.star.wave > cov)
        if not outside.any():
            return
        dropped = self.star.filter_names[outside]
        s = copy.copy(self.star)
        s.used_filters = self.star.used_filters.copy()
        s.used_filters[outside] = 0
        s.filter_mask = np.where(s.used_filters == 1)[0]
        self.star = s
        logger.warning(
            "Grid '%s' only covers <= %.2f um; excluding %d out-of-coverage "
            "band(s) from the fit: %s", self._grid, cov, len(dropped),
            ', '.join(dropped))

    def add_contrast(self, filt, delta_mag, sigma):
        """Add a flux-ratio (delta-magnitude) constraint for a binary fit.

        ``filt`` is an ARIADNE filter name where a companion contrast was
        measured (speckle/AO/Gaia), ``delta_mag`` = |m2 - m1| > 0 in that band,
        ``sigma`` its 1-sigma uncertainty. At least one such constraint is
        required to enable ``binary`` mode (the secondary is otherwise
        degenerate). Multiple bands of differing colour additionally constrain
        the secondary Teff.
        """
        if self.contrast is None:
            self.contrast = []
        self.contrast.append((str(filt), float(delta_mag), float(sigma)))

    def _prep_contrast(self):
        """Validate ``self.contrast`` and pack it into the (cfilts, dmags,
        sigmas) array triple the likelihood consumes. The filter array is built
        once so the grid-interpolation cache keys on its identity.
        """
        cols = set(self._interpolator.column_index.keys())
        cfilts, dmags, sigmas = [], [], []
        for filt, dmag, sig in self.contrast:
            filt = str(filt)
            if filt not in cols:
                raise ValueError(
                    f"contrast filter {filt!r} is not a column in the model "
                    f"grid; use an ARIADNE filter name present in the grid.")
            cfilts.append(filt)
            dmags.append(float(dmag))
            sigmas.append(float(sig))
        return (np.array(cfilts), np.array(dmags), np.array(sigmas))

    def _add_binary_dilution(self, out):
        """Per-band host-flux fraction f1/(f1+f2) from the binary posterior.

        This is the Furlan & Howell planet-radius dilution / third-light
        correction -- the quantity exoplanet radius-vetting needs. Distance and
        the shared extinction cancel in the ratio, so the dilution in a band is
        ``f1*rad1^2 / (f1*rad1^2 + f2*rad2^2)``. Stored per fitted band in
        ``out['binary_dilution']`` (posterior samples). Also flags the secondary
        as prior-dominated when its Teff scatter is large -- a single-band
        contrast typically leaves teff2 weakly constrained, in which case the
        dilution is still reliable but teff2 should not be quoted.
        """
        ps = out['posterior_samples']
        t1, g1, z, r1 = (np.asarray(ps['teff']), np.asarray(ps['logg']),
                         np.asarray(ps['z']), np.asarray(ps['rad']))
        t2, g2, r2 = (np.asarray(ps['teff2']), np.asarray(ps['logg2']),
                      np.asarray(ps['rad2']))
        bands = self.star.filter_names[self.star.filter_mask]
        n = len(t1)
        host = np.empty((n, len(bands)))
        for i in range(n):
            f1 = get_interpolated_flux(
                t1[i], g1[i], z[i], bands, interpolator) * r1[i] ** 2
            f2 = get_interpolated_flux(
                t2[i], g2[i], z[i], bands, interpolator) * r2[i] ** 2
            host[i] = f1 / (f1 + f2)
        out['binary_dilution'] = {str(b): host[:, j]
                                  for j, b in enumerate(bands)}
        # Also expose each band's dilution as a posterior trace so the BMA
        # averager (which iterates posterior_samples) folds it across grids.
        for j, b in enumerate(bands):
            out['posterior_samples'][f'dilution_{b}'] = host[:, j]
        frac = float(np.std(t2) / np.median(t2)) if np.median(t2) else np.inf
        out['binary_secondary_unconstrained'] = bool(frac > 0.15)
        if self.verbose:
            med = ', '.join(
                f'{b} {np.median(host[:, j]):.3f}'
                for j, b in enumerate(bands))
            print(colored('\t\t\tBinary host-flux fraction (median): '
                          + med, 'cyan'))
            if out['binary_secondary_unconstrained']:
                print(colored(
                    '\t\t\tWARNING: secondary Teff is prior-dominated '
                    '(sigma/Teff > 15%); the dilution is reliable but teff2 is '
                    'weakly constrained. Add a second contrast band of '
                    'differing colour to constrain it.', 'yellow'))

    def _evidence_fit(self, binary, grids, subfolder):
        """Run an auxiliary SED-only fit (no contrast) and return per-grid
        log-evidence. Used by the model comparison; reuses this fit's star,
        sampler settings and primary priors. For ``binary=True`` the degenerate
        (contrast-free) binary is allowed via ``_skip_contrast_gate``.
        """
        sub = Fitter()
        sub.star = self.star
        sub.setup = list(self._setup)
        sub.norm = False
        sub._av_law = self._av_law            # resolved law (bypass setter)
        sub.binary = binary
        sub._skip_contrast_gate = True        # SED-only, no contrast term
        sub.estimate_logg = self.estimate_logg
        sub.estimate_age = False              # not needed for the evidence
        sub.n_samples = self.n_samples
        sub.n_grid_jobs = self.n_grid_jobs
        sub.prior_setup = {
            k: v for k, v in self.prior_setup.items()
            if k in ('teff', 'logg', 'z', 'dist', 'rad', 'Av', 'norm')}
        sub.out_folder = self.out_folder.rstrip('/') + '/' + subfolder
        sub.verbose = False
        if self._bma:
            sub.bma = True
            sub.models = grids
        else:
            sub.bma = False
            sub.grid = self._grid
        sub.initialize()
        if self._bma:
            sub.fit_bma()
        else:
            sub.fit_dynesty()
        ev = {}
        for g in grids:
            try:
                with open(f'{sub.out_folder}/{g}_out.pkl', 'rb') as fh:
                    ev[g] = float(pickle.load(fh)['global_lnZ'])
            except (FileNotFoundError, KeyError):
                pass
        return ev

    def _binary_model_comparison(self):
        """Does the broadband SED alone favour a binary? Reports, per grid, the
        SED-only Bayesian evidence of the two-component vs single-star model
        (``delta_lnZ > 0`` favours the binary).

        BOTH models condition on the same data -- the photometry, NOT the
        contrast. Including the contrast would trivially favour the binary (it
        is a companion detection), so this asks the separate, well-posed
        question of whether the SED shows two temperatures on its own. If
        ``delta_lnZ <= 0`` the secondary is established by the contrast rather
        than the SED, so trust the dilution but not the SED-derived teff2. The
        contrast still does its job in the main fit (the dilution). Roughly
        triples the run time; a failure here is logged, not fatal.
        """
        grids = list(self._bma_models) if self._bma else [self._grid]
        print(colored('\t\t*** SED-ONLY BINARY-vs-SINGLE EVIDENCE ***', 'cyan'))
        try:
            z_single = self._evidence_fit(False, grids, 'single_star_comparison')
            z_binary = self._evidence_fit(True, grids, 'binary_sed_comparison')
        except Exception as e:
            print(colored(f'\t\t\tmodel comparison failed ({e}); the binary '
                          f'fit is unaffected.', 'yellow'))
            return

        comp = {}
        for g in grids:
            if g not in z_binary or g not in z_single:
                continue
            zb, zs = z_binary[g], z_single[g]
            comp[g] = {'lnZ_binary_sed': zb, 'lnZ_single': zs,
                       'delta_lnZ': zb - zs}
            verdict = ('SED favours binary' if zb - zs > 0
                       else 'SED alone does not require binary')
            print(colored(f'\t\t\t{g}: ln Z  binary {zb:.2f}  single {zs:.2f}'
                          f'  ->  Delta {zb - zs:+.2f}  ({verdict})', 'cyan'))
        self.binary_comparison = comp
        for g, c in comp.items():
            p = f'{self.out_folder}/{g}_out.pkl'
            try:
                with open(p, 'rb') as fh:
                    o = pickle.load(fh)
                o['binary_vs_single'] = c
                with open(p, 'wb') as fh:
                    pickle.dump(o, fh)
            except FileNotFoundError:
                pass

    def initialize(self):
        """Initialize the fitter.

        To be run only after every input is added.
        This function calculates the number of dimensions, runs the prior
        creation, creates output directory, initializes coordinators and sets
        up global variables.
        """
        global prior_dict, coordinator, fixed, order, star, use_norm, av_law
        global binary, contrast, jitter_shared
        jitter_shared = (getattr(self, 'jitter', True) == 'shared')
        self.start = time.time()
        err_msg = 'No star is detected. Please create an instance of Star.'
        if self.star is None:
            er = InputError(err_msg)
            er.log(self.out + '/output.log')
            er.__raise__()

        # Unresolved-binary mode gate. The secondary is degenerate without a
        # flux-ratio constraint, so refuse to run without one; it also needs the
        # explicit rad/dist (non-norm) model and a parallax-based distance.
        binary = bool(self.binary)
        contrast = None
        if binary:
            if self.norm:
                raise ValueError('binary mode requires norm=False (explicit '
                                 'rad/dist), not the normalisation parametrisation.')
            if not self.contrast and not getattr(self, '_skip_contrast_gate',
                                                  False):
                raise ValueError('binary mode requires at least one flux-ratio '
                                 'constraint: Fitter.add_contrast(filter, '
                                 'delta_mag, sigma) or set Fitter.contrast.')
            if self.star.dist is None or self.star.dist == -1:
                raise ValueError('binary mode requires a parallax-based distance '
                                 '(shared between the two components).')
            # contrast is optional only for the internal SED-only comparison fit
            # (_skip_contrast_gate); a normal binary fit always has one.
            contrast = self._prep_contrast() if self.contrast else None
        # For a standalone fit with a grid of limited wavelength coverage,
        # exclude out-of-coverage bands (e.g. Coelho's extrapolated IR). Not
        # applied in BMA, where every grid must share the same band set.
        if not self._bma:
            self._apply_grid_coverage()
        star = self.star
        if not self._bma:
            global interpolator
            interpolator = self._interpolator
        use_norm = self.norm
        # Extinction law
        av_law = self._av_law

        # Declare order of parameters.
        if not self.norm:
            order = np.array(
                [
                    'teff', 'logg', 'z',
                    'dist', 'rad', 'Av'
                ]
            )
            if binary:
                order = np.append(order, ['teff2', 'logg2', 'rad2'])
        else:
            order = np.array(['teff', 'logg', 'z', 'norm', 'Av'])

        # Create output directory
        if self.out_folder is None:
            self.out_folder = self.star.starname + '/'
        create_dir(self.out_folder)

        self.star.save_mags(self.out_folder + '/')

        # Parameter coordination.
        # Order for the parameters are:
        # teff, logg, z, dist, rad, Av, noise
        # or
        # teff, logg, z, norm, Av, noise
        npars = 6 if not self.norm else 5
        if binary:
            npars += 3  # teff2, logg2, rad2
        # Noise model: one jitter term per band by default, or a single shared
        # jitter term when ``jitter == 'shared'``.
        npars += 1 if getattr(self, 'jitter', True) == 'shared' else self.star.used_filters.sum()
        npars = int(npars)
        self.coordinator = np.zeros(npars)  # 1 for fixed params
        self.fixed = np.zeros(npars)
        coordinator = self.coordinator
        fixed = self.fixed

        # Setup priors.
        self.default_priors = self._default_priors()
        self.create_priors_from_setup()
        prior_dict = self.priors

        # Fast / bulk mode: deactivate the per-band excess-noise (jitter) terms
        # by fixing every ``*_noise`` parameter to 0. This removes one sampling
        # dimension per band (the jitter terms dominate the dimensionality of a
        # well-covered SED) and makes the likelihood fall back to the catalogue
        # errors alone — no likelihood change needed, the fixed-parameter
        # machinery handles it. ``get_ndim`` below subtracts these from ndim.
        if not self.jitter:
            for i, p in enumerate(order):
                if str(p).endswith('_noise'):
                    self.coordinator[i] = 1
                    self.fixed[i] = 0.0

        # Get dimensions.
        self.ndim = self.get_ndim()

        # warnings
        if len(self._setup) == 1:
            logger.warning('Using default setup values')

        # BMA settings
        # if BMA is used, load all interpolators requested.
        if self._bma:
            if self.n_samples is None:
                self.n_samples = 'max'

            if self.star.offline:
                if self.star.temp is None:
                    self.star.temp = 4001
                off_msg = 'Offline mode assumes that the stellar'
                off_msg += ' temperature is greater than 4000 K'
                off_msg += '. If you believe this is not the case then please '
                off_msg += 'add a temperature to the Star constructor. '
                off_msg += f'The input temperature is {self.star.temp}'
                print(colored(off_msg, 'yellow'))

            for mod in self._bma_models:
                # We'll assume that if ARIADNE is running in offline mode
                # Then the star will have > 4000 K
                if (mod.lower() in ['btcond', 'btnextgen'] and
                    self.star.temp > 4000) or \
                        (mod.lower() in ['ck04', 'kurucz'] and
                         self.star.temp < 4000) or \
                        (mod.lower() == 'coelho' and self.star.temp < 3500) or \
                        (mod.lower() == 'bosz' and
                         (self.star.temp < 2800 or self.star.temp > 16000)) or \
                        (mod.lower() == 'sphinx' and self.star.temp > 4000) or \
                        (mod.lower() == 'tlusty' and self.star.temp < 15000):
                    continue

                df = self.load_interpolator(mod.lower())

                self._interpolators.append(df)
                self._grids.append(mod)
            thr = self._threads if self._sequential else len(
                self._interpolators)
        else:
            thr = self._threads
        en = 'Bayesian Model Averaging' if self._bma else self._engine
        display_routine(en, self._nlive, self._dlogz, self.ndim, self._bound,
                        self._sample, thr, self._dynamic)
        self.show_priors()

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = 6 if not self.norm else 5
        if self.binary:
            ndim += 3  # teff2, logg2, rad2
        ndim += 1 if getattr(self, 'jitter', True) == 'shared' else self.star.used_filters.sum()
        ndim -= self.coordinator.sum()
        return int(ndim)

    @staticmethod
    def _gaia_color_absmag(star):
        """Gaia BP-RP colour and absolute G magnitude from the star.

        Returns ``(bp_rp, abs_g)``, either being ``None`` when its inputs are
        missing (no Gaia photometry, or no parallax/distance). This is the robust
        cool-dwarf signal: M dwarfs almost always have Gaia BP/RP photometry even
        when they lack a Gaia Teff/luminosity estimate.
        """
        try:
            fn = list(star.filter_names)
            mags = star.mags
            bp = mags[fn.index('GaiaDR2v2_BP')]
            g = mags[fn.index('GaiaDR2v2_G')]
            rp = mags[fn.index('GaiaDR2v2_RP')]
        except (ValueError, AttributeError, KeyError, IndexError, TypeError):
            return None, None
        bp_rp = (bp - rp) if (bp and rp) else None
        abs_g = None
        dist = getattr(star, 'dist', None)
        if g and dist and dist > 0:
            abs_g = g + 5.0 - 5.0 * np.log10(dist)
        return bp_rp, abs_g

    @staticmethod
    def _cool_dwarf_flags(temp, lum, bp_rp=None, abs_g=None):
        """Classify a star for regime-aware priors.

        Returns ``(is_cool, is_cool_dwarf)``. The FGK population priors mishandle
        M dwarfs (Teff pulled ~600 K hot into a hot-giant degeneracy; logg left
        wide), so cool stars get an M/late-K Teff prior and cool *dwarfs* a tight
        main-sequence logg prior.

        Primary signal is Gaia BP-RP colour (>1.8 -> M/late-K) and absolute G
        magnitude (>8 -> on the main sequence, excluding red giants), because M
        dwarfs frequently lack the Gaia Teff/luminosity estimates. Falls back to
        the Teff estimate (< 4000 K) and luminosity (< 0.5 Lsun) when colour /
        parallax are unavailable (e.g. non-Gaia input).
        """
        if bp_rp is not None:
            is_cool = bp_rp > 1.8
        else:
            # temp > 0 (not just != 0) so the -1 / -999 "no Gaia data" sentinels
            # are treated as unknown, NOT as cool (-1 < 4000 would misfire).
            is_cool = temp is not None and temp > 0 and temp < 4000
        if abs_g is not None:
            dwarfish = abs_g > 8.0
        elif lum is not None and lum > 0:
            dwarfish = lum < 0.5
        else:
            dwarfish = False
        return is_cool, bool(is_cool and dwarfish)

    def _default_priors(self):
        global order
        defaults = dict()

        # Initialize prior source tracking
        if not hasattr(self, 'prior_sources'):
            self.prior_sources = {}

        # Resolve spectroscopic parameters: prefer spectroscopic_params,
        # fall back to rave_params for backward compatibility.
        spec = getattr(self.star, 'spectroscopic_params', None)
        if spec is None:
            spec = getattr(self.star, 'rave_params', None)
        spec_source = spec.get('source', 'spectroscopic') if spec else None

        # Per-field null-safe helper: a spectroscopic field is usable only if
        # the spec dict exists and both the value and its error are not None.
        # This handles partial spec dicts (e.g. Hypatia [Fe/H]-only results
        # where teff/logg are None) by falling back per-field to the
        # population/default prior rather than crashing.
        def _spec_has(field):
            return (spec is not None and
                    spec.get(field) is not None and
                    spec.get(field + '_err') is not None)

        # Regime detection for cool stars: the FGK population priors are wrong
        # for M dwarfs (Teff pulled ~600 K hot into a hot-giant degeneracy; logg
        # left wide open). Use the Gaia Teff/luminosity estimates to switch in
        # M-dwarf-appropriate Teff/logg priors below. Spectroscopic priors, when
        # present, still take precedence over these.
        _bp_rp, _abs_g = self._gaia_color_absmag(self.star)
        is_cool, is_cool_dwarf = self._cool_dwarf_flags(
            getattr(self.star, 'temp', None), getattr(self.star, 'lum', None),
            _bp_rp, _abs_g)

        _grid_l = (getattr(self, '_grid', '') or '').lower()
        # Teff: standalone grids that sit outside the FGK population prior need
        # their own wide uniform prior. Koester white dwarfs span 5000-80000 K;
        # TLUSTY hot stars 15-55 kK. The FGK population prior assigns ~zero
        # probability there, so it cannot reach either regime.
        if _grid_l == 'koester':
            defaults['teff'] = st.uniform(loc=5000, scale=75000)  # 5-80 kK (WD)
            self.prior_sources['teff'] = 'koester_wd_uniform'
            logger.info('Using uniform Teff prior for the Koester WD grid')
        elif _grid_l == 'tlusty':
            defaults['teff'] = st.uniform(loc=15000, scale=40000)  # 15-55 kK
            self.prior_sources['teff'] = 'tlusty_uniform'
            logger.info('Using uniform Teff prior for the TLUSTY hot-star grid')
        # Teff: spectroscopic star-specific or population prior
        elif _spec_has('teff'):
            defaults['teff'] = st.norm(loc=spec['teff'],
                                       scale=spec['teff_err'])
            self.prior_sources['teff'] = spec_source
            logger.info('Using %s Teff prior (star-specific)', spec_source)
        elif is_cool:
            # M / late-K dwarf range; uniform avoids imposing the FGK centre.
            defaults['teff'] = st.uniform(loc=2300, scale=1900)  # 2300-4200 K
            self.prior_sources['teff'] = 'mdwarf_uniform'
            logger.info('Using cool-star uniform Teff prior (Gaia Teff < 4000 K)')
        else:
            with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
                defaults['teff'] = pickle.load(jar)
            self.prior_sources['teff'] = 'rave_population'
            logger.info('Using population Teff prior (no spectroscopic match)')

        # logg: spectroscopic, isochrone, or population prior
        if _grid_l == 'koester':
            defaults['logg'] = st.uniform(loc=6.5, scale=3.0)  # 6.5-9.5 (WD)
            self.prior_sources['logg'] = 'koester_wd_uniform'
            logger.info('Using uniform logg prior for the Koester WD grid')
        elif _spec_has('logg'):
            defaults['logg'] = st.norm(loc=spec['logg'],
                                       scale=spec['logg_err'])
            self.prior_sources['logg'] = spec_source
            logger.info('Using %s logg prior (star-specific)', spec_source)
        elif is_cool_dwarf:
            # M dwarfs sit tightly on the main sequence; this replaces both the
            # wide uniform default and the (typically uninformative, ~3 dex)
            # isochrone estimate, which is what let broadband fits drift to the
            # hot-giant solution.
            defaults['logg'] = st.norm(loc=5.0, scale=0.25)
            self.prior_sources['logg'] = 'mdwarf_main_sequence'
            logger.info('Using M-dwarf main-sequence logg prior N(5.0, 0.25)')
        elif self.star.get_logg:
            defaults['logg'] = st.norm(
                loc=self.star.logg, scale=self.star.logg_e)
            self.prior_sources['logg'] = 'isochrone'
            logger.info('Using isochrone logg estimate')
        else:
            defaults['logg'] = st.uniform(loc=3.5, scale=2.5)
            self.prior_sources['logg'] = 'uniform_default'
            logger.info('Using population logg prior')

        # [Fe/H]: spectroscopic or population prior
        if _grid_l == 'koester':
            # White dwarfs have no metallicity axis; the Koester grid carries a
            # single dummy z=0 node, so pin z to it. The trilinear interpolator
            # handles the resulting length-1 third axis as long as z is fixed at
            # exactly that node (z index is 2 in both norm and non-norm orders).
            z_idx = 2
            self.coordinator[z_idx] = 1
            self.fixed[z_idx] = 0.0
            defaults['z'] = None
            self.prior_sources['z'] = 'koester_fixed_zero'
            logger.info('Pinning [Fe/H]=0 for the Koester WD grid (no z axis)')
        elif _spec_has('feh'):
            defaults['z'] = st.norm(loc=spec['feh'], scale=spec['feh_err'])
            self.prior_sources['z'] = spec_source
            logger.info('Using %s [Fe/H] prior (star-specific)', spec_source)
        else:
            defaults['z'] = st.norm(loc=-0.125, scale=0.234)
            self.prior_sources['z'] = 'population'
            logger.info('Using population [Fe/H] prior')
        # Distance prior setup.
        if not self._norm:
            if self.star.dist != -1:
                _d_scale = 3 * self.star.dist_e
                defaults['dist'] = st.truncnorm(
                    a=-self.star.dist / _d_scale, b=1e100,
                    loc=self.star.dist, scale=_d_scale,
                )
                self.prior_sources['dist'] = 'gaia'
            else:
                defaults['dist'] = st.uniform(loc=1, scale=3000)
                self.prior_sources['dist'] = 'uniform_default'
            # Radius prior setup.
            if _grid_l == 'koester':
                # White dwarfs are ~0.004-0.02 R_sun; the main-sequence floor of
                # 0.05 R_sun would exclude every WD radius. Use a degenerate
                # range instead.
                defaults['rad'] = st.uniform(loc=0.003, scale=0.047)  # 0.003-0.05
                self.prior_sources['rad'] = 'koester_wd_uniform'
            else:
                defaults['rad'] = st.uniform(loc=0.05, scale=100)
                self.prior_sources['rad'] = 'uniform_default'
        # Normalization prior setup.
        else:
            # up = 1 / 1e-30
            # defaults['norm'] = st.truncnorm(a=0, b=up, loc=1e-20, scale=1e-10)
            defaults['norm'] = st.uniform(loc=0, scale=1e-20)
            self.prior_sources['norm'] = 'uniform_default'
        # Extinction prior setup.
        if self.star.Av == 0.:
            av_idx = 4 if self._norm else 5
            self.coordinator[av_idx] = 1
            self.fixed[av_idx] = 0
            defaults['Av'] = None
            self.prior_sources['Av'] = 'fixed_zero'
        else:
            if self.star.Av_e is None:
                defaults['Av'] = st.uniform(loc=0, scale=self.star.Av)
                self.prior_sources['Av'] = 'dustmap_uniform'
            else:
                mu = self.star.Av
                sig = self.star.Av_e
                low = 0
                up = 100
                b, a = (up - mu) / sig, (low - mu) / sig
                defaults['Av'] = st.truncnorm(loc=mu, scale=sig, a=a, b=b)
                self.prior_sources['Av'] = 'dustmap'
        # Noise model prior setup. Per-band jitter by default; a single shared
        # jitter term when ``jitter == 'shared'`` -- the likelihood broadcasts
        # the one term across all bands, keeping the error/model-adequacy buffer
        # while collapsing N nuisance dimensions to 1.
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        if getattr(self, 'jitter', True) == 'shared':
            mflx, merr = float(np.median(flxs)), float(np.median(errs))
            defaults['shared_noise'] = st.truncnorm(
                loc=0, scale=merr * 10, a=0, b=(1 - mflx) / merr)
            order = np.append(order, 'shared_noise')
        else:
            for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
                p_ = get_noise_name(filt) + '_noise'
                sigma = flx_e * 10
                b = (1 - flx) / flx_e
                defaults[p_] = st.truncnorm(loc=0, scale=sigma, a=0, b=b)
                order = np.append(order, p_)
        return defaults

    def create_priors_from_setup(self):
        """Create priors from the manual setup."""
        prior_dict = dict()
        keys = self.prior_setup.keys()
        noise = []
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        if getattr(self, 'jitter', True) == 'shared':
            noise = ['shared_noise']
        else:
            for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
                p_ = get_noise_name(filt) + '_noise'
                noise.append(p_)
        prior_out = 'Parameter\tPrior\tValues\n'
        if 'norm' in keys and ('rad' in keys or 'dist' in keys):
            er = PriorError('rad or dist', 1)
            er.log(self.out_folder + '/output.log')
            er.__raise__()
        for k in keys:
            if type(self.prior_setup[k]) == str:
                if self.prior_setup[k] == 'default':
                    prior_dict[k] = self.default_priors[k]
                    prior_out += k + '\tdefault\n'
                if self.prior_setup[k].lower() == 'rave':
                    # RAVE prior only available for teff and logg. It's already
                    # the default for [Fe/H]
                    if k == 'logg' or k == 'teff':
                        if k == 'teff':
                            with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
                                prior_dict[k] = pickle.load(jar)
                        else:  # k == 'logg'
                            with open(priorsdir + '/logg_ppf.pkl', 'rb') as jar:
                                prior_dict[k] = pickle.load(jar)
                        prior_out += k + '\tRAVE\n'

            else:
                prior = self.prior_setup[k][0]
                if prior == 'fixed':
                    value = self.prior_setup[k][1]
                    idx = np.where(k == order)[0]
                    self.coordinator[idx] = 1
                    self.fixed[idx] = value
                    prior_out += k + '\tfixed\t{}\n'.format(value)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_fixed'
                if prior == 'normal':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    prior_dict[k] = st.norm(loc=mu, scale=sig)
                    prior_out += k + '\tnormal\t{}\t{}\n'.format(mu, sig)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_normal'
                if prior == 'truncnorm':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    low = self.prior_setup[k][3]
                    up = self.prior_setup[k][4]
                    b, a = (up - mu) / sig, (low - mu) / sig
                    prior_dict[k] = st.truncnorm(a=a, b=b, loc=mu, scale=sig)
                    prior_out += k
                    prior_out += '\ttruncatednormal\t{}\t{}\t{}\t{}\n'.format(
                        mu, sig, low, up)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_truncnorm'
                if prior == 'uniform':
                    low = self.prior_setup[k][1]
                    up = self.prior_setup[k][2]
                    prior_dict[k] = st.uniform(loc=low, scale=up - low)
                    prior_out += k + '\tuniform\t{}\t{}\n'.format(low, up)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_uniform'
        for par in noise:
            prior_dict[par] = self.default_priors[par]
        # Binary secondary shares the primary's resolved priors. Sharing the
        # exact prior keeps the teff1>=teff2 swap in the prior transform
        # measure-preserving (and gives a sensible default the user need not
        # restate). prior_out already records the primary entries.
        if self.binary:
            for tgt, src in (('teff2', 'teff'), ('logg2', 'logg'),
                             ('rad2', 'rad')):
                if src in prior_dict:
                    prior_dict[tgt] = prior_dict[src]
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[tgt] = self.prior_sources.get(src)
        ff = open(self.out_folder + '/prior.dat', 'w')
        ff.write(prior_out)
        ff.close()
        del ff
        self.priors = prior_dict
        pass

    def show_priors(self, return_string=False):
        """Display priors in statistical notation.

        Parameters
        ----------
        return_string : bool, optional
            If True, return the formatted string instead of printing.
            Default: False

        Returns
        -------
        str or None
            Formatted prior information if return_string=True, else None
        """
        global order
        _T = '\t\t\t'
        c = np.random.choice(colors)

        # Physical parameters to display (skip noise priors)
        physical_params = []
        for param_name in self.priors.keys():
            if param_name.endswith('_noise'):
                continue
            physical_params.append(param_name)

        # Format each prior
        prior_strs = {}
        for param_name in physical_params:
            prior_obj = self.priors[param_name]
            param_idx = np.where(order == param_name)[0]
            if len(param_idx) > 0 and self.coordinator[param_idx[0]] == 1:
                fixed_value = self.fixed[param_idx[0]]
                prior_str = f"Fixed({fixed_value:.4f})"
            else:
                prior_str = self._format_prior_notation(param_name, prior_obj)
            prior_strs[param_name] = prior_str

        # Sort parameters in logical order
        param_order = ['teff', 'logg', 'z', 'dist', 'rad', 'norm', 'Av']
        sorted_params = [p for p in param_order if p in physical_params]
        sorted_params.extend([p for p in physical_params if p not in param_order])

        # Build table
        lines = []
        lines.append(f'{_T}{"Parameter":24s}  {"Prior"}')
        lines.append(f'{_T}{"-" * 52}')
        for param_name in sorted_params:
            lines.append(f'{_T}{param_name:24s}  {prior_strs[param_name]}')

        output = '\n'.join(lines)

        if return_string:
            return output
        else:
            print(colored(output, c))
            print()
            return None

    def _format_prior_notation(self, param_name, prior_obj):
        """Format scipy.stats distribution in statistical notation.

        Parameters
        ----------
        param_name : str
            Parameter name (e.g., 'teff', 'logg')
        prior_obj : scipy.stats distribution
            The prior distribution object

        Returns
        -------
        str
            Formatted statistical notation
        """
        # RAVE population prior: loaded as an InterpolatedUnivariateSpline
        # from teff_ppf.pkl / logg_ppf.pkl. Detect by type since manual
        # prior setups don't populate prior_sources.
        if 'Spline' in type(prior_obj).__name__:
            return "RAVE (population)"
        if hasattr(self, 'prior_sources') and param_name in self.prior_sources:
            if self.prior_sources[param_name] == 'rave_population':
                return "RAVE (population)"

        # Frozen scipy distributions expose .dist (the class) and .kwds/.args
        # (the frozen parameters). Use the underlying class name to identify
        # the distribution.
        if hasattr(prior_obj, 'dist'):
            inner_name = type(prior_obj.dist).__name__
            kwds = dict(getattr(prior_obj, 'kwds', {}) or {})
            args = list(getattr(prior_obj, 'args', ()) or ())
        else:
            inner_name = type(prior_obj).__name__
            kwds, args = {}, []

        def _pop(key, idx, default=0.0):
            if key in kwds:
                return kwds[key]
            if idx < len(args):
                return args[idx]
            return default

        try:
            if 'norm_gen' in inner_name and 'truncnorm' not in inner_name:
                mu = _pop('loc', 0)
                sigma = _pop('scale', 1, 1.0)
                return f"N({mu:.3f}, {sigma:.3f})"

            if 'uniform_gen' in inner_name:
                a = _pop('loc', 0)
                scale = _pop('scale', 1, 1.0)
                b = a + scale
                return f"U({a:.2f}, {b:.2f})"

            if 'truncnorm_gen' in inner_name:
                a_std = _pop('a', 0)
                b_std = _pop('b', 1)
                mu = _pop('loc', 2)
                sigma = _pop('scale', 3, 1.0)
                a_act = mu + a_std * sigma
                b_act = mu + b_std * sigma
                a_str = "-inf" if a_std < -1e10 else f"{a_act:.2f}"
                b_str = "inf" if b_std > 1e10 else f"{b_act:.2f}"
                return f"TN({mu:.2f}, {sigma:.2f}, [{a_str}, {b_str}])"

            return inner_name

        except (AttributeError, TypeError):
            return f"{inner_name} (custom)"

    def fit(self):
        """Run fitting routine."""
        self.fit_dynesty()
        if self.binary:
            self._binary_model_comparison()
        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time,
            self.out_folder, self._engine, self.norm, grid=self._grid)
        pass

    def fit_bma(self):
        """Perform the fit with different models and the average the output.

        Only works with dynesty.
        """
        if len(self.star.filter_names[self.star.filter_mask]) <= 5:
            print(colored('\t\t\tNOT ENOUGH POINTS TO MAKE THE FIT! !', 'red'))
            return

        global interpolator, _BMA_FITTER

        n_jobs = min(max(int(self.n_grid_jobs), 1), len(self._grids))
        if n_jobs > 1:
            # Across-grid parallelism: the grids are independent fits, so run
            # several at once (one core each). Each worker writes its own
            # ``_out.pkl``, which we read back below for averaging.
            ncpu = os.cpu_count() or 1
            if n_jobs > ncpu:
                logger.warning(
                    'n_grid_jobs=%d exceeds the %d available CPU cores; '
                    'this will oversubscribe the machine.', n_jobs, ncpu)
            _BMA_FITTER = self
            print(f'\t\t\tFITTING {len(self._grids)} MODELS, '
                  f'{n_jobs} CONCURRENTLY')
            with Pool(n_jobs) as pool:
                for gr, err in pool.map(_bma_grid_worker,
                                        range(len(self._grids))):
                    if err is not None:
                        dump_out = self.out_folder + '/' + gr + '_DUMP.pkl'
                        DynestyError(dump_out, gr, err).__raise__()
        else:
            for intp, gr in zip(self._interpolators, self._grids):
                interpolator = intp
                self.grid = gr
                out_file = self.out_folder + '/' + gr + '_out.pkl'
                print('\t\t\tFITTING MODEL : ' + gr)
                try:
                    self.fit_dynesty(out_file=out_file)
                except ValueError as e:
                    dump_out = self.out_folder + '/' + gr + '_DUMP.pkl'
                    pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                    DynestyError(dump_out, gr, e).__raise__()
                    continue

        # Now that the fitting finished, read the outputs and average
        # the posteriors
        outs = []
        for g in self._grids:
            in_folder = f'{self.out_folder}/{g}_out.pkl'
            outs.append(in_folder)
            # with open(in_folder, 'rb') as out:
            #     outs.append(pickle.load(out))
        c = np.random.choice(self.colors)
        avgd = self.bayesian_model_average(outs, self._grids, self._norm,
                                           self.n_samples, c)
        self.save_bma(avgd)

        if self.binary:
            self._binary_model_comparison()

        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time, self.out_folder,
            'Bayesian Model Averaging', self.norm)
        pass

    def _bma_dynesty(self, intp, grid):
        global interpolator
        interpolator = intp

        # Parallel parallelized routine experiment
        if self.experimental:
            if self._dynamic:
                with Pool(self._threads) as executor:
                    sampler = dynesty.DynamicNestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz_init=self._dlogz,
                                       nlive_batch=self._nlive,
                                       wt_kwargs={'pfrac': .95})
            else:
                with Pool(self._threads) as executor:
                    sampler = dynesty.NestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz=self._dlogz)

        elif self._dynamic:
            sampler = dynesty.DynamicNestedSampler(
                dynesty_loglike_bma, pt_dynesty, self.ndim,
                bound=self._bound, sample=self._sample, logl_args=([intp])

            )
            sampler.run_nested(dlogz_init=self._dlogz,
                               nlive_init=self._nlive,
                               wt_kwargs={'pfrac': .95})
        else:
            try:
                self.sampler = dynesty.NestedSampler(
                    dynesty_loglike_bma, pt_dynesty, self.ndim,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample,
                    logl_args=([intp])
                )
                self.sampler.run_nested(dlogz=self._dlogz)
            except Error:
                dump_out = self.out_folder + '/' + grid + '_DUMP.pkl'
                pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                er = DynestyError(dump_out, grid)
                er.log(self.out + '/output.log')
                er.__raise__()

        results = self.sampler.results
        out_file = self.out_folder + '/' + grid + '_out.pkl'
        self.save(out_file, results=results)
        pass

    def fit_dynesty(self, out_file=None):
        """Run dynesty."""
        # Set up some globals
        global mask, flux, flux_er, filts, wave
        mask = star.filter_mask
        flux = star.flux[mask]
        flux_er = star.flux_er[mask]
        filts = star.filter_names[mask]
        wave = star.wave[mask]
        if self._dynamic:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.DynamicNestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample,
                        pool=executor, walks=self.walks,
                        queue_size=self._threads
                    )
                    self.sampler.run_nested(dlogz_init=self._dlogz,
                                            nlive_init=self._nlive,
                                            wt_kwargs={'pfrac': 1})
            else:
                self.sampler = dynesty.DynamicNestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=self.walks,
                    bound=self._bound, sample=self._sample

                )
                self.sampler.run_nested(dlogz_init=self._dlogz,
                                        nlive_init=self._nlive,
                                        wt_kwargs={'pfrac': 1})
        else:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.NestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample,
                        pool=executor, walks=self.walks,
                        queue_size=self._threads,
                    )
                    self.sampler.run_nested(dlogz=self._dlogz)
            else:
                self.sampler = dynesty.NestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=self.walks,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample
                )
                self.sampler.run_nested(dlogz=self._dlogz)
        results = self.sampler.results
        if out_file is None:
            out_file = f'{self.out_folder}/{self._grid}_out.pkl'
        self.save(out_file, results=results)
        pass

    def save(self, out_file, results=None):
        """Save the dynesty output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ: The global evidence.
        lnZerr: The global evidence error.
        posterior_samples: A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed: An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit: The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star: The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)
        engine: The fitting engine used (Dynesty)

        Also creates a log file with the best fit parameters and 1 sigma
        error bars.

        """
        out = dict()
        logdat = '#Parameter\tmedian\tupper\tlower\t3sig_CI\n'
        log_out = self.out_folder + '/' + 'best_fit.dat'
        lnz, lnzer, posterior_samples = self.dynesty_results(results)

        mask = self.star.filter_mask

        # Save global evidence
        out['dynesty'] = results
        out['global_lnZ'] = lnz
        out['global_lnZerr'] = lnzer

        # Create raw samples holder

        out['posterior_samples'] = dict()
        j = 0
        k = 0  # filter counter
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                samples = posterior_samples[:, j]
                if 'noise' in param:
                    filt = self.star.filter_names[mask][k]
                    flx = self.star.flux[mask][k]
                    _, samples = flux_to_mag(flx, samples, filt)
                    k += 1
                out['posterior_samples'][param] = samples
                j += 1
            else:
                out['posterior_samples'][param] = self.fixed[i]

        # Save loglike, priors and posteriors.

        out['posterior_samples']['loglike'] = np.zeros(
            posterior_samples.shape[0]
        )

        # If normalization constant was fitted, create a distribution of radii
        # only if there's a distance available.

        if use_norm and star.dist != -1:
            rad = self._get_rad(
                out['posterior_samples']['norm'], star.dist, star.dist_e
            )
            out['posterior_samples']['rad'] = rad

        # Create a distribution of masses.

        logg_samp = out['posterior_samples']['logg']
        rad_samp = out['posterior_samples']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        out['posterior_samples']['grav_mass'] = mass_samp

        # Create a distribution of luminosities.

        teff_samp = out['posterior_samples']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        out['posterior_samples']['lum'] = lum_samp

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['posterior_samples']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            out['posterior_samples']['AD'] = ad_samp

        for i in range(posterior_samples.shape[0]):
            theta = build_params(
                posterior_samples[i, :], flux, flux_er, filts, coordinator,
                fixed, self.norm)
            out['posterior_samples']['loglike'][i] = log_likelihood(
                theta, flux, flux_er, wave, filts, interpolator, self.norm,
                av_law, binary=self.binary, contrast=contrast)

        if self.binary:
            self._add_binary_dilution(out)

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.
        # do only if not bma

        if not self.bma:
            out['best_fit_averaged'] = dict()
            out['uncertainties_averaged'] = dict()
            out['confidence_interval_averaged'] = dict()
            best_theta = np.zeros(order.shape[0])

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if 'noise' in param:
                        continue
                    samp = out['posterior_samples'][param]

                    if param == 'z':
                        logdat = out_filler(samp, logdat, param, '[Fe/H]', out)
                    elif param == 'norm':
                        logdat = out_filler(samp, logdat, param, '(R/D)^2',
                                            out, fmt='e')
                        if star.dist != 1:
                            logdat = out_filler(
                                out['posterior_samples']['rad'], logdat, 'rad',
                                'R', out
                            )
                    else:
                        logdat = out_filler(samp, logdat, param, param, out)
                else:
                    logdat = out_filler(
                        0, logdat, param, param, out, fixed=self.fixed[i]
                    )
                best_theta[i] = out['best_fit_averaged'][param]

            # Add derived mass to best fit dictionary.

            samp = out['posterior_samples']['grav_mass']
            logdat = out_filler(
                samp, logdat, 'grav_mass', 'grav_mass', out
            )

            # Add derived luminosity to best fit dictionary.

            samp = out['posterior_samples']['lum']
            logdat = out_filler(samp, logdat, 'lum', 'lum', out)

            # Add derived angular diameter to best fit dictionary.

            if not use_norm:
                samp = out['posterior_samples']['AD']
                logdat = out_filler(samp, logdat, 'AD', 'AD', out)

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if 'noise' not in param:
                        continue
                    samp = out['posterior_samples'][param]
                    logdat = out_filler(samp, logdat, param,
                                        param, out, fmt='f')

            # Fill in best loglike, prior and posterior.

            out['best_fit_averaged']['loglike'] = log_likelihood(
                best_theta, flux, flux_er, wave,
                filts, interpolator, self.norm, av_law,
                binary=self.binary, contrast=contrast
            )

            # Spectral type
            # Load Mamajek spt table
            mamajek_spt = np.loadtxt(
                filesdir + '/mamajek_spt.dat', dtype=str, usecols=[0])
            mamajek_temp = np.loadtxt(
                filesdir + '/mamajek_spt.dat', usecols=[1])

            # Find spt
            spt_idx = np.argmin(abs(mamajek_temp - out['best_fit_averaged']['teff']))
            spt = mamajek_spt[spt_idx]
            out['spectral_type'] = spt

        # Utilities for plotting.

        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['engine'] = self._engine
        out['norm'] = self.norm
        out['model_grid'] = self.grid
        out['av_law'] = av_law
        out['binary'] = self.binary

        # Fit-health diagnostics (additive; never changes the fit). Grid axes
        # come from the interpolator the fit used (free, no reload); the
        # grid-edge check is skipped gracefully if they can't be obtained.
        grid_axes = None
        _interp = globals().get('interpolator', None)
        if _interp is not None and hasattr(_interp, 'index_columns'):
            try:
                ii = _interp.index_columns          # (logg, teff, z)
                grid_axes = {'logg': ii[0], 'teff': ii[1], 'z': ii[2]}
            except Exception:
                grid_axes = None
        out['diagnostics'] = self._fit_health(
            results, param_samples=out['posterior_samples'],
            grid_axes=grid_axes)
        if self.verbose:
            self._print_fit_health(out['diagnostics'], self.grid)

        if not self.bma:
            with open(log_out, 'w') as logfile:
                logfile.write(logdat)
        try:
            out['irx'] = self._compute_irx(out, [self._grid], av_law)
        except Exception as e:                       # never break the save
            logger.warning('IR-excess quantification skipped: %s', e)
            out['irx'] = {}
        pickle.dump(out, open(out_file, 'wb'))
        pass

    def _irx_theta(self, d):
        """Pack a best-fit / posterior-sample dict into the physical theta
        vector model_grid expects, in the same order the fit used."""
        if self._norm:
            order = ['teff', 'logg', 'z', 'norm', 'Av']
        else:
            order = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
            if self.binary:
                order += ['teff2', 'logg2', 'rad2']
        return np.array([float(d[k]) for k in order])

    def _compute_irx(self, out, grids, av_law):
        """Quantify IR excess on the bands every in-play grid reaches.

        The star's IR-excess bands (``config.irx_bands``, routed out of the
        photosphere fit by ``Star``) are compared to the model predicted at the
        best fit. Reach gating follows the BMA rule: a band is *quantified* only
        if every in-play grid reaches it -- the common-reach set is the
        intersection of the grids' ``finite_bands``. Bands the star has but that
        lie beyond the common reach are recorded plot-only (model = NaN), so the
        SED can still show them.

        The model flux and its uncertainty per band are the (evidence-weighted,
        for BMA) posterior-predictive: draws pick a grid by weight and a
        posterior sample from that grid, then evaluate the model on that grid.
        Reported per band: model, model_err, observed, excess = obs - model and
        its significance sigma = excess / sqrt(obs_err^2 + model_err^2). A
        '_summary' entry lists the quantified, plot-only and >=3 sigma bands.
        """
        star = self.star
        mask = np.asarray(getattr(star, 'irx_filter_mask', []), dtype=int)
        if mask.size == 0:
            return {}
        names = np.asarray(star.filter_names)
        irx_filt = names[mask]
        irx_wave = np.asarray(star.wave)[mask]
        irx_obs = np.asarray(star.flux)[mask]
        irx_obs_e = np.asarray(star.flux_er)[mask]

        interps = {g: load_grid(g) for g in grids}
        common = set.intersection(*[grid_irx_reach(g, i)
                                    for g, i in interps.items()])

        # Posterior mixture: (grid, weight, posterior_samples) per member.
        if out.get('weights') and out.get('originals'):
            members = [(g, float(out['weights'].get(g, 0.0)),
                        out['originals'][g])
                       for g in grids if g in out['originals']]
        else:
            members = [(grids[0], 1.0, out['posterior_samples'])]
        wsum = sum(w for _, w, _ in members) or 1.0
        members = [(g, w / wsum, ps) for g, w, ps in members]

        rng = np.random.default_rng(0)
        ndraw = 256
        gpick = rng.choice(len(members), size=ndraw,
                           p=[w for _, w, _ in members])

        result = {}
        for f, wv, fo, fe in zip(irx_filt, irx_wave, irx_obs, irx_obs_e):
            f = str(f)
            if f not in common:
                result[f] = dict(wave=float(wv), obs=float(fo),
                                 obs_err=float(fe), model=float('nan'),
                                 model_err=float('nan'), excess=float('nan'),
                                 sigma=float('nan'), quantified=False)
                continue
            band = np.array([f])
            wl = np.array([float(wv)])
            draws = np.empty(ndraw)
            for d in range(ndraw):
                g, _w, ps = members[gpick[d]]
                n = len(ps['teff'])
                j = int(rng.integers(n))
                theta = self._irx_theta({k: ps[k][j] for k in ps
                                         if hasattr(ps[k], '__len__')})
                m = model_grid(theta, band, wl, interps[g], self._norm,
                               av_law, binary=self.binary)
                draws[d] = float(np.asarray(m).ravel()[0])
            model = float(np.nanmean(draws))
            model_err = float(np.nanstd(draws))
            excess = float(fo) - model
            denom = float(np.hypot(float(fe), model_err))
            sigma = excess / denom if denom > 0 else float('nan')
            result[f] = dict(wave=float(wv), obs=float(fo), obs_err=float(fe),
                             model=model, model_err=model_err, excess=excess,
                             sigma=sigma, quantified=True)
        q = {k: v for k, v in result.items() if v['quantified']}
        result['_summary'] = dict(
            quantified=sorted(q),
            plot_only=sorted(k for k, v in result.items()
                             if k != '_summary' and not v['quantified']),
            significant=sorted(k for k, v in q.items()
                               if np.isfinite(v['sigma']) and v['sigma'] >= 3),
            common_reach_um=(float(np.nanmax([v['wave'] for v in q.values()]))
                             if q else float('nan')))
        return result

    def save_bma(self, avgd):
        """Save BMA output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ : The global evidences.
        posterior_samples : A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed : An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit : The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star : The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)

        Also creates a log file with the best fit parameters, 1 sigma
        error bars and 3 sigma CIs.

        """
        out = dict()
        logdat_samples = '#Parameter\tmedian\tupper\tlower\t3sig_low\t3sig_up\n'
        logdat_average = '#Parameter\tmedian\tupper\tlower\t3sig_low\t3sig_up\n'
        log_out_samples = f'{self.out_folder}/best_fit_sample.dat'
        log_out_average = f'{self.out_folder}/best_fit_average.dat'
        prob_out = f'{self.out_folder}/model_probabilities.dat'
        synth_out = f'{self.out_folder}/synthetic_fluxes.dat'

        # Save global evidence of each model.
        out['lnZ'] = avgd['evidences']

        # Save original samples.
        out['originals'] = avgd['originals']

        # Save weights.
        out['weights'] = avgd['weights']

        # Create raw samples holder
        out['weighted_samples'] = dict()
        out['weighted_average'] = dict()
        j = 0
        for i, par in enumerate(order):
            if not self.coordinator[i]:
                out['weighted_samples'][par] = avgd['weighted_samples'][par]
                out['weighted_average'][par] = avgd['weighted_average'][par]
                j += 1
            else:
                out['weighted_samples'][par] = self.fixed[i]
                out['weighted_average'][par] = self.fixed[i]
        # Carry over the binary per-band dilution traces (not in `order`).
        for par in avgd['weighted_samples']:
            if par.startswith('dilution_'):
                out['weighted_samples'][par] = avgd['weighted_samples'][par]
                out['weighted_average'][par] = avgd['weighted_average'][par]

        # If normalization constant was fitted, create a distribution of radii.

        if use_norm and star.dist != -1:
            rad_sampled = self._get_rad(
                out['weighted_samples']['norm'], star.dist, star.dist_e
            )
            rad_averageed = self._get_rad(
                out['weighted_average']['norm'], star.dist, star.dist_e
            )
            out['weighted_samples']['rad'] = rad_sampled
            out['weighted_average']['rad'] = rad_averageed

        # Create a distribution of masses

        logg_samp = out['weighted_samples']['logg']
        logg_average = out['weighted_average']['logg']
        rad_samp = out['weighted_samples']['rad']
        rad_average = out['weighted_average']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        mass_average = self._get_mass(logg_average, rad_average)
        out['weighted_samples']['grav_mass'] = mass_samp
        out['weighted_average']['grav_mass'] = mass_average

        # Create a distribution of luminosities.

        teff_samp = out['weighted_samples']['teff']
        teff_average = out['weighted_average']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        lum_average = self._get_lum(teff_average, rad_average)
        out['weighted_samples']['lum'] = lum_samp
        out['weighted_average']['lum'] = lum_average

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['weighted_samples']['dist']
            dist_average = out['weighted_average']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            ad_average = self._get_angular_diameter(rad_average, dist_average)
            out['weighted_samples']['AD'] = ad_samp
            out['weighted_average']['AD'] = ad_average

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.

        out['best_fit_samples'] = dict()
        out['uncertainties_samples'] = dict()
        out['confidence_interval_samples'] = dict()
        out['best_fit_averaged'] = dict()
        out['uncertainties_averaged'] = dict()
        out['confidence_interval_averaged'] = dict()
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if 'noise' in param:
                    continue
                samp = out['weighted_samples'][param]
                sampw = out['weighted_average'][param]

                if param == 'z':
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                '[Fe/H]', out, method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                '[Fe/H]', out,
                                                method='averaged')
                elif param == 'norm':
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                '(R/D)^2', out, fmt='e',
                                                method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                '(R/D)^2', out, fmt='e',
                                                method='averaged')
                    if star.dist != 1:
                        logdat_samples = out_filler(
                            out['weighted_samples']['rad'], logdat_samples,
                            'rad', 'R', out, method='samples'
                        )
                        logdat_average = out_filler(
                            out['weighted_average']['rad'], logdat_average,
                            'rad', 'R', out, method='averaged'
                        )

                else:
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                param, out, method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                param, out, method='averaged')
            else:
                logdat_samples = out_filler(
                    0, logdat_samples, param, param, out, fixed=self.fixed[i],
                    method='samples'
                )
                logdat_average = out_filler(
                    0, logdat_average, param, param, out, fixed=self.fixed[i],
                    method='averaged'
                )

        # Add derived mass to best fit dictionary.

        samp = out['weighted_samples']['grav_mass']
        sampw = out['weighted_average']['grav_mass']
        logdat_samples = out_filler(samp, logdat_samples, 'grav_mass',
                                    'grav_mass', out, method='samples')
        logdat_average = out_filler(sampw, logdat_average, 'grav_mass',
                                    'grav_mass', out, method='averaged')

        # Add derived luminosity to best fit dictionary.

        samp = out['weighted_samples']['lum']
        sampw = out['weighted_average']['lum']
        logdat_samples = out_filler(samp, logdat_samples, 'lum', 'lum', out,
                                    method='samples')
        logdat_average = out_filler(sampw, logdat_average, 'lum', 'lum', out,
                                    method='averaged')

        # Add derived angular diameter to best fit dictionary.

        if not use_norm:
            samp = out['weighted_samples']['AD']
            sampw = out['weighted_average']['AD']
            logdat_samples = out_filler(samp, logdat_samples, 'AD', 'AD', out,
                                        method='samples')
            logdat_average = out_filler(sampw, logdat_average, 'AD', 'AD', out,
                                        method='averaged')

        # Add estimated age to best fit dictionary. This is done with the wider
        # sampled distribution instead of the averaged one in order to save time

        if self.estimate_age:
            age_samp, mass_samp, eep_samp = self._run_estimate_age(
                out['best_fit_samples'],
                out['uncertainties_samples'],
                c=choice(self.colors)
            )
            # Create new thingy for MIST samples. Sadly now everything done before
            # this update will be incompatible :(
            out['mist_samples'] = dict()
            out['mist_samples']['age'] = age_samp
            out['mist_samples']['iso_mass'] = mass_samp
            out['mist_samples']['eep'] = eep_samp
            logdat_samples = out_filler(age_samp, logdat_samples, 'age', 'age', out,
                                        method='samples')
            logdat_samples = out_filler(mass_samp, logdat_samples, 'iso_mass',
                                        'iso_mass', out, method='samples')
            logdat_samples = out_filler(eep_samp, logdat_samples, 'eep', 'eep', out,
                                        method='samples')
            # Ugly... but faster than doing the kde 2x times...
            age = out['best_fit_samples']['age']
            age_unc = out['uncertainties_samples']['age']
            age_ci = out['confidence_interval_samples']['age']
            iso_mass = out['best_fit_samples']['iso_mass']
            iso_mass_unc = out['uncertainties_samples']['iso_mass']
            iso_mass_ci = out['confidence_interval_samples']['iso_mass']
            eep = out['best_fit_samples']['eep']
            eep_unc = out['uncertainties_samples']['eep']
            eep_ci = out['confidence_interval_samples']['eep']
            out['best_fit_averaged']['age'] = age
            out['best_fit_averaged']['iso_mass'] = iso_mass
            out['best_fit_averaged']['eep'] = eep
            out['uncertainties_averaged']['age'] = age_unc
            out['uncertainties_averaged']['iso_mass'] = iso_mass_unc
            out['uncertainties_averaged']['eep'] = eep_unc
            out['confidence_interval_averaged']['age'] = age_ci
            out['confidence_interval_averaged']['iso_mass'] = iso_mass_ci
            out['confidence_interval_averaged']['eep'] = eep_ci
            logdat_average += f'age\t{age:.4f}\t'
            logdat_average += f'{age_unc[1]:.4f}\t{age_unc[0]:.4f}\t'
            logdat_average += f'{age_ci[0]:.4f}\t{age_ci[1]}\n'
            logdat_average += f'iso_mas\t{iso_mass:.4f}\t'
            logdat_average += f'{iso_mass_unc[1]:.4f}\t{iso_mass_unc[0]:.4f}\t'
            logdat_average += f'{iso_mass_ci[0]:.4f}\t{iso_mass_ci[1]}\n'
            logdat_average += f'eep\t{eep:.4f}\t'
            logdat_average += f'{eep_unc[1]:.4f}\t{eep_unc[0]:.4f}\t'
            logdat_average += f'{eep_ci[0]:.4f}\t{eep_ci[1]}\n'
        else:
            out['mist_samples'] = {}
        ###
        probdat = ''

        max_prob = 0
        max_prob_mod = ''
        for k in avgd['weights'].keys():
            if avgd['weights'][k] > max_prob:
                max_prob = avgd['weights'][k]
                max_prob_mod = k
            probdat += f'{k}_probability\t{avgd["weights"][k]:.4f}\n'

        # Get synthetic mag and fluxes for highest probability model.
        synth_mod = 'btsettl'
        intp = self.load_interpolator(synth_mod)
        ogteff = avgd['originals'][max_prob_mod]['teff']
        oglogg = avgd['originals'][max_prob_mod]['logg']
        ogfeh = avgd['originals'][max_prob_mod]['z']
        fluxes = np.zeros((len(oglogg), len(filter_names)))
        for i, t, g, z in zip(range(len(oglogg)), ogteff, oglogg, ogfeh):
            fluxes[i, :] = get_interpolated_flux(t, g, z, filter_names, intp)
        synthdat = 'Filter\tFlux\n'
        for i, filt in enumerate(filter_names):
            samp = fluxes[:, i]
            # xx, pdf = estimate_pdf(samp)
            # cdf = estimate_cdf(samp, hdr=True)
            b, _, _ = credibility_interval(samp, alpha=1)
            synthdat += f'{filt}\t{b:.6e}\n'

        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if 'noise' not in param:
                    continue
                samp = out['weighted_samples'][param]
                sampw = out['weighted_average'][param]
                logdat_samples = out_filler(samp, logdat_samples, param, param,
                                            out, fmt='f', method='samples')
                logdat_average = out_filler(sampw, logdat_average, param, param,
                                            out, fmt='f', method='averaged')

        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['norm'] = self.norm
        out['engine'] = 'Bayesian Model Averaging'
        out['av_law'] = av_law
        out['binary'] = self.binary

        # Spectral type

        # Load Mamajek spt table
        mamajek_spt, mamajek_temp = np.loadtxt(f'{filesdir}/mamajek_spt.dat',
                                               dtype=str, usecols=[0, 1],
                                               unpack=True)

        mamajek_temp = mamajek_temp.astype(float)

        # Find spt
        spt_idx = np.argmin(
            abs(mamajek_temp - out['best_fit_averaged']['teff']))
        spt = mamajek_spt[spt_idx]
        out['spectral_type'] = spt
        out_file = f'{self.out_folder}/BMA.pkl'
        with open(log_out_samples, 'w') as logfile:
            logfile.write(logdat_samples)
        with open(log_out_average, 'w') as logfile:
            logfile.write(logdat_average)
        with open(prob_out, 'w') as logfile:
            logfile.write(probdat)
        with open(synth_out, 'w') as logfile:
            logfile.write(synthdat)
        try:
            out['irx'] = self._compute_irx(out, list(self._grids), av_law)
        except Exception as e:                       # never break the save
            logger.warning('IR-excess quantification skipped: %s', e)
            out['irx'] = {}
        pickle.dump(out, open(out_file, 'wb'))
        self.out = out
        # NetCDF is an optional ecosystem-interop export (needs arviz). Don't
        # let a missing optional dependency or an export hiccup discard the
        # primary BMA.pkl result that was just written.
        nc_file = f'{self.out_folder}/ariadne_result.nc'
        try:
            self.to_netcdf(nc_file)
        except Exception as e:
            logger.warning('NetCDF export skipped (%s); BMA.pkl is written.', e)

    @staticmethod
    def bayesian_model_average(outputs, grids, norm, nsamples, c='white'):
        """Perform Bayesian Model Averaging.

        Parameters
        ----------
        outputs: array_like
            An array or list of output pickle files from the individual model
            fits.
        grids: array_like
            An array or list of the model grids used to model.
        norm: bool
            Flag indicating if the normalization factor was fit for.
        nsamples: int
            The number of samples to sample from the averaged distribution.
        c: str, optional
            Termcolor color.

        """
        evidences = []
        post_samples = []
        model_posteriors = []
        # Read and extract model posterior information.
        for o in outputs:
            with open(o, 'rb') as f:
                model_posteriors.append(pickle.load(f))
        # Extract the evidences of each model.
        for o in model_posteriors:
            evidences.append(o['global_lnZ'])
            post_samples.append(o['posterior_samples'])
        # Resolve the sample count. ``n_samples`` defaults to the sentinel
        # 'max' (set in ``initialize`` when the user leaves it unset); interpret
        # it as the largest per-model posterior length. Without this, 'max'
        # reaches ``np.zeros``/``choice`` below and raises "'str' object cannot
        # be interpreted as an integer" — so a BMA run used to crash at the
        # averaging step unless n_samples was set explicitly.
        if not isinstance(nsamples, (int, np.integer)):
            nsamples = max(len(v) for o in post_samples for v in o.values()
                           if hasattr(v, '__len__'))
        # Per-model fit-health diagnostics + evidence errors (additive).
        per_model_diag = {}
        evidence_err = {}
        for o in model_posteriors:
            g = o['model_grid']
            if 'diagnostics' in o:
                per_model_diag[g] = o['diagnostics']
            evidence_err[g] = o.get('global_lnZerr', float('nan'))
        # Convert evidences to weights/probabilities via a softmax of the
        # log-evidences. Shift by the MAX (not the min) so the largest exponent
        # is 0: softmax is shift-invariant, so the weights are unchanged in the
        # normal regime, but shifting by the min lets exp() overflow to inf ->
        # nan when one grid's log-evidence dominates (delta lnZ > ~709).
        evidences = np.array(evidences)
        weights = evidences - evidences.max()
        weights = [np.exp(e) / np.exp(weights).sum() for e in weights]
        weights = np.array(weights)
        # We're not averaging these
        ban = ['loglike', 'priors', 'posteriors']
        if norm:  # if normalization was used, we won't average the radius
            ban.append('rad')
        # Create an output dictionary for the averaged samples.
        out = dict()
        out['originals'] = dict()
        out['weights'] = dict()
        # Populate the dict with the probabilities and individual model
        # posteriors
        for i, o in enumerate(model_posteriors):
            out['weights'][o['model_grid']] = weights[i]
            out['originals'][o['model_grid']] = o['posterior_samples']
        out['weighted_samples'] = dict()
        out['weighted_average'] = dict()

        print(colored('\t\t*** AVERAGING POSTERIOR SAMPLES ***', c))
        for k in tqdm(post_samples[0].keys()):
            if k in ban:  # Skip things that are not main parameters.
                continue
            try:  # Skip fixed params.
                len(post_samples[0][k])
            except TypeError:
                continue
            traces = []
            extended_weights = []
            out['weighted_samples'][k] = np.zeros(nsamples)
            out['weighted_average'][k] = np.zeros(nsamples)
            for i, o in enumerate(post_samples):
                # This is for the weighted sampling.
                traces.append(o[k])
                extended_weights.append(np.ones(len(o[k])) * weights[i])
                # This is for the weighted averaging.
                # The weighted averaging consists of taking the weighted
                # average of the posterior samples, supersampled to maked them
                # coincide in length.
                weighted_samples = choice(o[k], nsamples) * weights[i]
                out['weighted_average'][k] += weighted_samples
            # Do the weighted sampling. For this we're going to estimate the
            # KDE of the 'master' posterior built by taking random samples
            # from each model where the number of samples is proportional
            # to the model's relative probability. This is equivalent
            # to taking the KDE of each model and then do the weighted average
            avg_kde = st.gaussian_kde(np.concatenate(traces),
                                   weights=np.concatenate(extended_weights))
            out['weighted_samples'][k] = avg_kde.resample(nsamples)[0]
        # Now we save the evidences.
        out['evidences'] = dict()
        for e, g in zip(evidences, grids):
            out['evidences'][g] = e
        out['evidences_err'] = evidence_err
        out['per_model_diagnostics'] = per_model_diag
        # Surface per-grid fit health to the user.
        if per_model_diag:
            print(colored('\t\t*** PER-GRID FIT HEALTH ***', c))
            for g in grids:
                if g in per_model_diag:
                    Fitter._print_fit_health(per_model_diag[g], g)
        return out

    @staticmethod
    def dynesty_results(results):
        """Extract posterior samples, global evidence and its error."""
        weights = np.exp(results['logwt'] - results['logz'][-1])
        posterior_samples = resample_equal(results.samples, weights)
        lnz = results.logz[-1]
        lnzer = results.logzerr[-1]
        return lnz, lnzer, posterior_samples

    @staticmethod
    def _fit_health(results, param_samples=None, grid_axes=None,
                    edge_frac=0.02):
        """Compute additive fit-health diagnostics for a single grid fit.

        Purely diagnostic — it never alters the fit. Returns a dict with:

        - ``logz`` / ``logzerr`` : the grid's log-evidence and its error.
        - ``eff_samples`` : Kish effective sample size of the importance
          weights (a low value flags an under-sampled / unreliable posterior).
        - ``dlogz_final`` : the last log-evidence increment (a convergence
          proxy; small means the run had stopped accumulating evidence).
        - ``edge_flags`` : a list of teff/logg/z parameters whose posterior
          median sits within ``edge_frac`` of a grid boundary, i.e. the fit is
          pegged against the edge of the model grid and the (tight) error bar
          is not trustworthy. Empty when ``grid_axes`` is unavailable.

        Parameters
        ----------
        results : dynesty results
            The sampler results object.
        param_samples : dict, optional
            ``{param_name: samples}`` (e.g. the saved ``posterior_samples``),
            used for the grid-edge check.
        grid_axes : dict, optional
            ``{'teff': arr, 'logg': arr, 'z': arr}`` of grid node values.
        edge_frac : float
            Fraction of the grid span that counts as "at the edge".
        """
        diag = {}
        try:
            diag['logz'] = float(results.logz[-1])
            diag['logzerr'] = float(results.logzerr[-1])
        except Exception:
            diag['logz'] = float('nan')
            diag['logzerr'] = float('nan')
        try:
            logwt = np.asarray(results['logwt'])
            w = np.exp(logwt - results.logz[-1])
            w = w / w.sum()
            diag['eff_samples'] = int(round(1.0 / np.sum(w ** 2)))
        except Exception:
            diag['eff_samples'] = None
        try:
            diag['dlogz_final'] = float(results.logz[-1] - results.logz[-2])
        except Exception:
            diag['dlogz_final'] = None

        flags = []
        if param_samples is not None and grid_axes is not None:
            for p, axis in grid_axes.items():
                s = param_samples.get(p)
                if s is None:
                    continue
                s = np.atleast_1d(np.asarray(s, dtype=float))
                if s.size < 2:          # fixed parameter, nothing to flag
                    continue
                axis = np.asarray(axis, dtype=float)
                lo, hi = float(axis.min()), float(axis.max())
                span = hi - lo
                if span <= 0:
                    continue
                med = float(np.median(s))
                if med <= lo + edge_frac * span:
                    flags.append({'param': p, 'edge': 'lower',
                                  'median': med, 'bound': lo})
                elif med >= hi - edge_frac * span:
                    flags.append({'param': p, 'edge': 'upper',
                                  'median': med, 'bound': hi})
        diag['edge_flags'] = flags
        return diag

    @staticmethod
    def _print_fit_health(diag, grid_name=None):
        """Print a one-line fit-health summary plus any grid-edge warnings."""
        tag = f' — {grid_name}' if grid_name else ''
        lnz = diag.get('logz', float('nan'))
        lnzerr = diag.get('logzerr', float('nan'))
        neff = diag.get('eff_samples')
        neff_s = 'n/a' if neff is None else str(neff)
        print(colored(
            f'\t\t\tFit health{tag}: log Z = {lnz:.3f} +/- {lnzerr:.3f} | '
            f'eff. samples {neff_s}', 'cyan'))
        for f in diag.get('edge_flags', []):
            print(colored(
                f"\t\t\t  WARNING: {f['param']} posterior pegged at the "
                f"{f['edge']} grid edge (median {f['median']:.4g}, "
                f"bound {f['bound']:.4g}) — error bar may be unreliable.",
                'yellow'))

    @staticmethod
    def _get_mass(logg, rad):
        """Calculate mass from logg and radius."""
        # Solar logg = 4.437
        # g = g_Sol * M / R**2
        mass = logg + 2 * np.log10(rad) - 4.437
        mass = 10 ** mass
        return mass

    @staticmethod
    def _get_lum(teff, rad):
        sb = sigma_sb.to(u.solLum / u.K ** 4 / u.solRad ** 2).value
        L = 4 * np.pi * rad ** 2 * sb * teff ** 4
        return L

    @staticmethod
    def _get_rad(samples, dist, dist_e):
        """Calculate radius from the normalization constant and distance."""
        norm = samples
        # Create a synthetic distribution for distance.
        # N = (R / D) ** 2
        b, a = (np.inf - dist) / dist_e, (0 - dist) / dist_e
        d = st.truncnorm(loc=dist, scale=dist_e, a=a, b=b).rvs(
            size=norm.shape[0])
        n = np.sqrt(norm)
        r = n * d  # This is in pc
        r *= u.pc.to(u.solRad)  # Transform to Solar radii
        return r

    @staticmethod
    def _get_angular_diameter(rad, dist):
        diameter = 2 * rad
        ad = (diameter / (dist * u.pc.to(u.solRad))) * u.rad.to(u.marcsec)
        return ad

    @staticmethod
    def load_interpolator(model):
        """Load a grid interpolator by name."""
        return load_grid(model)

    def _run_estimate_age(self, bf, unc, c='white'):
        """Estimate age using MIST isochrones.

        Parameters
        ----------
        bf: dict
            A dictionary with the best fit parameters.
        unc: dict
            A dictionary with the uncertainties.

        """
        print(
            colored(
                '\t\t*** ESTIMATING AGE AND MASS USING MIST ISOCHRONES ***', c
            )
        )
        params = dict()  # params for isochrones.
        for i, k in enumerate(order):
            if k == 'logg' or 'noise' in k:
                continue
            if k == 'teff':
                par = 'Teff'
            if k == 'z':
                par = 'feh'
            if k == 'dist':
                par = 'distance'
            if k == 'rad':
                par = 'radius'
            if k == 'Av':
                par = 'AV'
            if not self.coordinator[i]:
                if k != 'norm':
                    params[par] = (bf[k], max(unc[k]))
                if k == 'norm' and self.star.dist != -1:
                    params['distance'] = (self.star.dist, self.star.dist_e)
                if par == 'distance':
                    err = max(unc[k])
                    params['parallax'] = (1000 / bf[k], 1000 * err / bf[k] ** 2)
            else:
                continue

        params['mass'] = (bf['grav_mass'], max(unc['grav_mass']))
        if self.star.lum != 0 and self.star.lum_e != 0:
            params['logL'] = (np.log10(bf['lum']),
                              abs(np.log10(max(unc['lum']))))
        mags = self.star.mags[iso_mask == 1]
        mags_e = self.star.mag_errs[iso_mask == 1]
        used_bands = []
        for m, e, b in zip(mags, mags_e, iso_bands):
            if m != 0:
                params[b] = (m, e)
                used_bands.append(b)

        age_samp, mass_samp, eep_samp = estimate(
            used_bands, params, logg=False,
            out_folder=self.out_folder,
            threads=self._threads, dlogz=self.isochrone_dlogz)

        return age_samp, mass_samp, eep_samp

    def to_dict(self):
        """Return BMA results as a structured dict matching the ecosystem spec.

        Returns a dictionary with keys ``posterior``, ``observed_data``, and
        ``sample_stats``, each containing numpy arrays ready for conversion to
        ``arviz.InferenceData``.

        Raises
        ------
        RuntimeError
            If the fitter has not been run yet (``self.out`` is not populated).
        """
        if not hasattr(self, 'out') or self.out is None:
            raise RuntimeError(
                'No results available. Run the fitter before calling to_dict().'
            )
        out = self.out
        ws = out.get('weighted_samples', {})

        # Determine number of draws from any free parameter
        n_draws = None
        for v in ws.values():
            if hasattr(v, '__len__') and len(v) > 1:
                n_draws = len(v)
                break
        if n_draws is None:
            n_draws = 100_000  # fallback

        def _to_chain(arr_or_scalar):
            """Reshape to (1, n_draws) for single-chain arviz convention."""
            if hasattr(arr_or_scalar, '__len__') and len(arr_or_scalar) > 1:
                return np.array(arr_or_scalar).reshape(1, -1)
            return np.full((1, n_draws), float(arr_or_scalar))

        # Map internal names → ecosystem spec names
        param_map = {
            'teff': 'Teff',
            'logg': 'logg',
            'z': 'feh',
            'rad': 'radius',
            'lum': 'luminosity',
            'dist': 'distance',
            'Av': 'Av',
        }

        posterior = {}
        for internal, spec in param_map.items():
            if internal in ws:
                posterior[spec] = _to_chain(ws[internal])

        # Observed photometry
        mask = self.star.filter_mask
        observed_data = {
            'wavelength': self.star.wave[mask],  # already in micron
            'flux': self.star.flux[mask],
            'flux_err': self.star.flux_er[mask],
        }

        # Model evidence and weights
        weights = out.get('weights', {})
        lnZ = out.get('lnZ', {})

        model_names = list(weights.keys())
        weight_vals = np.array([weights[m] for m in model_names])
        lnZ_vals = np.array([lnZ.get(m, 0.0) for m in model_names])
        bma_log_evidence = float(np.sum(weight_vals * lnZ_vals))

        sample_stats = {
            'log_evidence': bma_log_evidence,
            'model_names': model_names,
            'model_weights': weight_vals,
        }

        # Per-model posteriors from originals
        originals = out.get('originals', {})
        model_posteriors = {}
        for model_name, model_samples in originals.items():
            n_model = None
            for v in model_samples.values():
                if hasattr(v, '__len__') and len(v) > 1:
                    n_model = len(v)
                    break
            if n_model is None:
                continue

            def _to_model_chain(a, nd=n_model):
                if hasattr(a, '__len__') and len(a) > 1:
                    return np.array(a).reshape(1, -1)
                return np.full((1, nd), float(a))

            mp = {}
            for internal, spec in param_map.items():
                if internal in model_samples:
                    mp[spec] = _to_model_chain(model_samples[internal])
            if 'loglike' in model_samples:
                mp['log_likelihood'] = _to_model_chain(
                    model_samples['loglike'])
            model_posteriors[model_name] = mp

        # MIST isochrone samples
        mist = out.get('mist_samples', {})
        mist_posterior = {}
        for k in ('age', 'iso_mass', 'eep'):
            if k in mist and hasattr(mist[k], '__len__') and len(mist[k]) > 1:
                mist_posterior[k] = _to_chain(mist[k])

        # Filter names and bandwidths for SED reconstruction
        observed_data['filter_names'] = np.array(
            self.star.filter_names[mask].tolist()
            if hasattr(self.star.filter_names[mask], 'tolist')
            else list(self.star.filter_names[mask])
        )
        observed_data['bandwidths'] = self.star.bandpass[mask]

        # Model SED fluxes (injected by worker from SEDPlotter)
        model_sed = {}
        if 'model_sed' in out:
            ms = out['model_sed']
            if 'model_flux' in ms:
                model_sed['model_flux'] = np.array(ms['model_flux'])

        # Summary statistics for quick display
        summary = {}
        for src_key in ('best_fit_averaged', 'uncertainties_averaged',
                        'confidence_interval_averaged'):
            if src_key in out:
                summary[src_key] = {}
                for k, v in out[src_key].items():
                    if hasattr(v, '__len__'):
                        summary[src_key][k] = np.array(
                            [float(x) for x in v])
                    else:
                        summary[src_key][k] = float(v)

        return {
            'posterior': posterior,
            'observed_data': observed_data,
            'sample_stats': sample_stats,
            'model_posteriors': model_posteriors,
            'mist_posterior': mist_posterior,
            'model_sed': model_sed,
            'summary': summary,
        }

    def to_netcdf(self, path):
        """Export BMA results as arviz DataTree in netCDF4 format.

        Produces an ``arviz.InferenceData``-compatible ``.nc`` file
        following the ecosystem output spec.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``'ariadne_result.nc'``).
        """
        import arviz as az

        d = self.to_dict()

        ss = d['sample_stats']
        data = {
            'posterior': d['posterior'],
            'observed_data': d['observed_data'],
            'constant_data': {
                'log_evidence': np.array(ss['log_evidence']),
                'model_weights': ss['model_weights'],
                'model_names': np.array(ss['model_names']),
            },
        }

        # Add per-model posteriors as separate groups
        for model_name, mp in d.get('model_posteriors', {}).items():
            data[f'posterior_{model_name}'] = mp

        # Add MIST isochrone posteriors
        if d.get('mist_posterior'):
            data['posterior'].update(d['mist_posterior'])

        # Add model SED fluxes to observed_data
        if d.get('model_sed') and 'model_flux' in d['model_sed']:
            data['observed_data']['model_flux'] = d['model_sed']['model_flux']

        # Add summary stats to constant_data
        for stat_key in ('best_fit_averaged', 'uncertainties_averaged',
                         'confidence_interval_averaged'):
            if stat_key in d.get('summary', {}):
                for param, val in d['summary'][stat_key].items():
                    key = f'{stat_key}__{param}'
                    data['constant_data'][key] = np.array(val)

        dt = az.from_dict(data)
        dt.to_netcdf(path)
        logger.info(f'Wrote InferenceData to {path}')


#####################
# Dynesty wrappers


# Set by fit_bma before forking the grid-parallel Pool; read by the workers.
_BMA_FITTER = None


def _bma_grid_worker(index):
    """Fit one BMA model grid single-threaded and write its ``_out.pkl``.

    Runs in a forked worker process: the star photometry, priors, coordinator
    and other module globals are inherited from the parent, so only the
    per-grid interpolator is swapped in. The fit result is delivered via the
    on-disk pickle (the parent reads it back for averaging), so nothing needs
    to be pickled back to the parent. Returns ``(grid, None)`` on success or
    ``(grid, error)`` on failure.
    """
    global interpolator
    fitter = _BMA_FITTER
    intp = fitter._interpolators[index]
    grid = fitter._grids[index]
    out_file = f'{fitter.out_folder}/{grid}_out.pkl'
    # Forked processes inherit the parent's RNG state; reseed so each grid
    # draws independently.
    np.random.seed()
    interpolator = intp
    fitter._threads = 1   # grids are the unit of parallelism; no inner Pool
    fitter._grid = grid   # used by save() for out['model_grid']
    try:
        fitter.fit_dynesty(out_file=out_file)
    except ValueError as e:
        dump_out = f'{fitter.out_folder}/{grid}_DUMP.pkl'
        pickle.dump(fitter.sampler.results, open(dump_out, 'wb'))
        return grid, e
    return grid, None


def dynesty_loglike_bma(cube, interpolator):
    """Dynesty log likelihood wrapper for BMA."""
    theta = build_params(cube, coordinator, fixed, use_norm)
    return log_likelihood(theta, star, interpolator, use_norm, av_law)


def dynesty_log_like(cube):
    """Dynesty log likelihood wrapper."""
    theta = build_params(
        cube, flux, flux_er, filts, coordinator, fixed, use_norm
    )
    return log_likelihood(theta, flux, flux_er, wave,
                          filts, interpolator, use_norm, av_law,
                          binary=binary, contrast=contrast)


def pt_dynesty(cube):
    """Dynesty prior transform."""
    return prior_transform_dynesty(cube, flux, flux_er, filts, prior_dict,
                                   coordinator, use_norm, binary=binary,
                                   shared=jitter_shared)
