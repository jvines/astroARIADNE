"""Synthetic photometry in arbitrary (user-specified) filters, from the spectra cache.

ARIADNE's model grids only carry the ~100 bands ARIADNE knows about. To
predict the best-fit SED in any other filter (a pyphot library filter, a
``pyphot.Filter`` object or a ``(name, wavelength_um, transmission)`` tuple)
the requested filters are convolved ONCE against every node spectrum of the
BT-Settl groups of the spectra cache (``btsettl`` and ``btsettl_cool``), and
the node fluxes are arranged in a (logg, teff, feh, band) cube with the same
axes convention as the band grids so ``sed_library.model_grid`` can push the
posterior draws through it unchanged.

The convolution is a single matmul: pyphot's ``Filter.get_flux`` integrals

    photon: int F T lam dlam / int T lam dlam
    energy: int F T dlam / int T dlam

are evaluated with trapezoid weights on the spectrum's wavelength grid and
baked into a (n_wave, n_filters) coefficient matrix, which reproduces
``Filter.get_flux`` to rounding error. A band whose central-98% transmission
support the cached spectra do not span is NaN (the build_model_grids rule).
"""
import logging
import os

import astropy.units as u
import numpy as np
import pyphot

logger = logging.getLogger(__name__)

CACHE_GROUPS = ('btsettl', 'btsettl_cool')   # first wins where both have a node
_COVER_TRIM = 0.01                           # central-98% support, as the grids
FETCH_CMD = ("python -c \"from astroARIADNE.fetch import fetch_spectra_cache; "
             "fetch_spectra_cache()\"")


class SynthFilter:
    """A resolved filter: name, pyphot ``Filter``, support bounds and a key."""

    def __init__(self, name, filt):
        self.name = str(name)
        self.filt = filt
        wl = np.asarray(filt.wavelength.to('um').value, float)
        tr = np.asarray(filt.transmit, float)
        order = np.argsort(wl)
        wl, tr = wl[order], tr[order]
        cum = np.cumsum(tr * np.gradient(wl))
        cum /= cum[-1]
        self.lo = float(np.interp(_COVER_TRIM, cum, wl))
        self.hi = float(np.interp(1.0 - _COVER_TRIM, cum, wl))
        self.dtype = getattr(filt, 'dtype', 'photon')
        self.key = (self.name, self.dtype, len(wl), float(wl[0]), float(wl[-1]),
                    float(tr.sum()))
        self.wave_um = float(filt.cl.to('um').value)   # same as get_effective_wavelength


def _library_filter(name):
    from . import phot_utils
    getter = getattr(phot_utils, '_get_filter', None)
    if getter is not None:
        return getter(name)
    return phot_utils._PYPHOT_LIB[name]


def _is_tuple_spec(x):
    return (isinstance(x, tuple) and len(x) in (3, 4)
            and isinstance(x[0], str) and not isinstance(x[1], str))


def resolve_filter(spec):
    """Turn one user entry into a :class:`SynthFilter`."""
    if isinstance(spec, SynthFilter):
        return spec
    if isinstance(spec, str):
        try:
            return SynthFilter(spec, _library_filter(spec))
        except Exception as e:
            raise KeyError(f'synthetic_filters: {spec!r} is not a pyphot '
                           f'library filter ({e})') from None
    if _is_tuple_spec(spec):
        name, wl, tr = spec[:3]
        dtype = spec[3] if len(spec) == 4 else 'photon'
        if dtype not in ('photon', 'energy'):
            raise ValueError(f'synthetic_filters: detector type {dtype!r} for '
                             f'{name!r} must be "photon" or "energy"')
        wl = np.asarray(wl, float)
        tr = np.asarray(tr, float)
        if wl.shape != tr.shape or wl.ndim != 1 or wl.size < 2:
            raise ValueError(f'synthetic_filters: {name!r} needs matching 1-D '
                             'wavelength_um and transmission arrays')
        order = np.argsort(wl)
        filt = pyphot.Filter(wl[order] * u.um.to('AA') * u.AA, tr[order],
                             name=name, dtype=dtype, unit='Angstrom')
        return SynthFilter(name, filt)
    if hasattr(spec, 'transmit') and hasattr(spec, 'wavelength'):
        return SynthFilter(getattr(spec, 'name', 'user_filter'), spec)
    raise TypeError('synthetic_filters entries must be a pyphot filter name, '
                    'a pyphot Filter or a (name, wavelength_um, transmission'
                    '[, "photon"|"energy"]) tuple; got %r' % (type(spec),))


def resolve_filters(specs):
    """List of :class:`SynthFilter` from ``Fitter.synthetic_filters``."""
    if specs is None:
        return []
    if isinstance(specs, (str, SynthFilter)) or _is_tuple_spec(specs) \
            or hasattr(specs, 'transmit'):
        specs = [specs]
    out, seen = [], set()
    for s in specs:
        try:
            f = resolve_filter(s)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning('synthetic_filters: entry skipped: %s',
                           e.args[0] if e.args else e)
            continue
        if f.name in seen:
            continue
        seen.add(f.name)
        out.append(f)
    return out


def coeff_matrix(filts, w):
    """Bake pyphot's synthetic-flux integral into ``band = F @ coeff / den``.

    Returns ``(coeff[n_wave, nb], den[nb], covered[nb], supp[n_wave, nb])`` for
    spectrum wavelength grid ``w`` [um]. ``covered`` is False when the spectrum
    does not span the filter's central-98% support; ``supp`` marks each band's
    non-zero-response samples so a non-finite flux there NaNs the band without
    letting NaN * 0 elsewhere poison the matmul.
    """
    w = np.asarray(w, float)
    tw = np.empty_like(w)                       # trapezoid quadrature weights
    tw[1:-1] = 0.5 * (w[2:] - w[:-2])
    tw[0] = 0.5 * (w[1] - w[0])
    tw[-1] = 0.5 * (w[-1] - w[-2])
    coeff = np.zeros((len(w), len(filts)))
    den = np.zeros(len(filts))
    covered = np.zeros(len(filts), bool)
    for j, f in enumerate(filts):
        wl_f = f.filt.wavelength.to('um').value
        tr_f = np.asarray(f.filt.transmit, float)
        T = np.interp(w, wl_f, tr_f, left=0.0, right=0.0)
        weight = T * w if f.dtype == 'photon' else T
        coeff[:, j] = weight * tw
        den[j] = weight @ tw
        covered[j] = (w.min() <= f.lo) and (w.max() >= f.hi) and den[j] > 0
    return coeff, den, covered, coeff != 0.0


def convolve(F, coeff, den, covered, supp):
    """Band fluxes of spectra ``F`` (n, n_wave); NaN where not covered/finite."""
    F = np.asarray(F, float)
    bad = (~np.isfinite(F)).astype(np.float64) @ supp > 0
    fb = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0) @ coeff
    fb[:, covered] /= den[covered]
    fb[:, ~covered] = np.nan
    fb[bad] = np.nan
    return fb


def cache_node_fluxes(cache_path, filts, groups=CACHE_GROUPS, chunk=2000):
    """Convolve ``filts`` against every node spectrum of the BT-Settl groups.

    Returns ``(teff, logg, z, flux[n_nodes, nb])``; nodes present in more than
    one group are taken from the first group in ``groups``.
    """
    import h5py
    teffs, loggs, zs, fl = [], [], [], []
    seen = set()
    with h5py.File(cache_path, 'r') as h:
        for gname in groups:
            if gname not in h:
                continue
            g = h[gname]
            w = np.asarray(g['wavelength'][:], float)
            ct, cl, cz = g['teff'][:], g['logg'][:], g['z'][:]
            ops = coeff_matrix(filts, w)
            for s in range(0, len(ct), chunk):
                e = min(s + chunk, len(ct))
                keep = []
                for k in range(s, e):
                    key = (round(float(cl[k]), 4), round(float(ct[k]), 2),
                           round(float(cz[k]), 4))
                    if key in seen:
                        keep.append(False)
                    else:
                        seen.add(key)
                        keep.append(True)
                keep = np.array(keep)
                if not keep.any():
                    continue
                F = np.asarray(g['flux'][s:e], float)[keep]
                fl.append(convolve(F, *ops))
                teffs.append(ct[s:e][keep])
                loggs.append(cl[s:e][keep])
                zs.append(cz[s:e][keep])
    if not fl:
        raise KeyError(f'none of the groups {groups} found in {cache_path}')
    return (np.concatenate(teffs).astype(float), np.concatenate(loggs).astype(float),
            np.concatenate(zs).astype(float), np.concatenate(fl))


class NodeCube:
    """Duck-typed stand-in for the band-grid interpolator.

    Exposes ``index_columns`` (logg, teff, feh axes), ``grid`` (n_logg, n_teff,
    n_feh, n_band cube, NaN where no node) and ``column_index`` -- exactly what
    ``sed_library.model_grid`` reads.
    """

    def __init__(self, teff, logg, z, flux, names):
        ax_l = np.unique(np.round(logg, 4))
        ax_t = np.unique(np.round(teff, 2))
        ax_z = np.unique(np.round(z, 4))
        grid = np.full((len(ax_l), len(ax_t), len(ax_z), len(names)), np.nan)
        il = np.searchsorted(ax_l, np.round(logg, 4))
        it = np.searchsorted(ax_t, np.round(teff, 2))
        iz = np.searchsorted(ax_z, np.round(z, 4))
        grid[il, it, iz, :] = flux
        self.index_columns = (ax_l, ax_t, ax_z)
        self.grid = grid
        self.column_index = {n: i for i, n in enumerate(names)}
        finite = np.isfinite(grid).any(axis=(0, 1, 2))
        self.finite_bands = frozenset(n for n, i in self.column_index.items()
                                      if finite[i])


def build_cube(cache_path, filts):
    """:class:`NodeCube` of ``filts`` synthesized from the spectra cache."""
    t, g, z, f = cache_node_fluxes(cache_path, filts)
    return NodeCube(t, g, z, f, [x.name for x in filts])


def zero_points(sf):
    """(vega_zero_flux, ab_zero_flux) in erg s-1 cm-2 um-1 for a filter."""
    filt = sf.filt
    f_vega = float(filt.Vega_zero_flux.to('erg/(um * cm ** 2 * s)').value)
    try:
        f_ab = float(filt.AB_zero_flux.to('erg/(um * cm ** 2 * s)').value)
    except Exception:
        lam_um = float(filt.lpivot.to('um').value)
        f_nu = 3.631e-20                                # 3631 Jy, erg/s/cm2/Hz
        f_ab = f_nu * 2.99792458e14 / lam_um ** 2       # c [um/s] / lam^2
    return f_vega, f_ab


def user_record(sf, flux, flux_err, source):
    """One row of ``synthetic_user_filters.dat`` from a median flux and error."""
    nan = float('nan')
    rec = dict(wavelength_um=sf.wave_um, flux=nan, flux_err=nan, mag_vega=nan,
               mag_vega_err=nan, mag_ab=nan, mag_ab_err=nan, source=source)
    if np.isfinite(flux) and flux > 0:
        f_vega, f_ab = zero_points(sf)
        merr = 2.5 * flux_err / (np.log(10) * flux)
        rec.update(flux=float(flux), flux_err=float(flux_err),
                   mag_vega=float(-2.5 * np.log10(flux / f_vega)),
                   mag_vega_err=float(merr),
                   mag_ab=float(-2.5 * np.log10(flux / f_ab)),
                   mag_ab_err=float(merr))
    return rec


def user_table(records, header_note):
    """Text of ``synthetic_user_filters.dat``."""
    txt = ('# Predicted observed photometry in user-specified filters at the '
           'best-fit SED.\n'
           f'# {header_note}\n'
           '# flux in erg s-1 cm-2 um-1; Vega mag from pyphot Vega zero point, '
           'AB from 3631 Jy; errors are 1-sigma from posterior draws.\n'
           '# filter\twavelength_um\tflux\tflux_err\tmag_vega\tmag_vega_err\t'
           'mag_ab\tmag_ab_err\tsource\n')
    for n, r in records.items():
        txt += (f"{n}\t{r['wavelength_um']:.4f}\t{r['flux']:.6e}\t"
                f"{r['flux_err']:.6e}\t{r['mag_vega']:.4f}\t"
                f"{r['mag_vega_err']:.4f}\t{r['mag_ab']:.4f}\t"
                f"{r['mag_ab_err']:.4f}\t{r['source']}\n")
    return txt
