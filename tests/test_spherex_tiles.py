"""Ranged reads of a tile-compressed FLAGS plane (DR1/QR3 L2 layout).

Builds a small L2-shaped MEF in memory -- PRIMARY, IMAGE, RICE_1 row-tiled
FLAGS, VARIANCE -- and serves byte ranges from it, so the offsets walk and the
tile decoding run exactly as against IRSA.
"""
import io

import numpy as np
import pytest
from astropy.io import fits

from astroARIADNE.librarian import _spherex


class _BytesReader:
    multirange = True

    def __init__(self, data):
        self.data = data

    def get(self, key, ranges):
        return [self.data[a:b + 1] for a, b in ranges]


def _mef(flags, compress=True, tile=None):
    ny, nx = flags.shape
    rng = np.random.default_rng(1)
    hdus = [fits.PrimaryHDU(),
            fits.ImageHDU(rng.normal(size=(ny, nx)).astype('>f4'), name='IMAGE')]
    if compress:
        hdus.append(fits.CompImageHDU(flags, name='FLAGS', compression_type='RICE_1',
                                      tile_shape=tile or (1, nx)))
    else:
        hdus.append(fits.ImageHDU(flags, name='FLAGS'))
    hdus.append(fits.ImageHDU(rng.normal(size=(ny, nx)).astype('>f4'), name='VARIANCE'))
    buf = io.BytesIO()
    fits.HDUList(hdus).writeto(buf)
    return buf.getvalue()


def _flags(ny=60, nx=80):
    rng = np.random.default_rng(7)
    f = np.zeros((ny, nx), np.int32)
    m = rng.random((ny, nx)) < 0.2
    f[m] = rng.choice([1 << 1, 1 << 6, 1 << 21, (1 << 21) | (1 << 9)], m.sum())
    return f


@pytest.fixture(autouse=True)
def _fresh_layouts():
    _spherex._LAYOUTS.clear()
    yield
    _spherex._LAYOUTS.clear()


def test_compressed_flags_rows_decode_exactly():
    flags = _flags()
    rd = _BytesReader(_mef(flags))
    _, offs = _spherex._hdu_offsets(rd, 'dr1/level2/w/v/1/x.fits')
    off, _, z = offs['FLAGS']
    assert z is not None and z['width'] == flags.shape[1]
    assert offs['IMAGE'][2] is None and offs['VARIANCE'][2] is None
    y0, n = 13, 21
    desc = rd.get('k', [(off + y0 * z['rowbytes'], off + (y0 + n) * z['rowbytes'] - 1)])[0]
    rows = _spherex._decode_tile_rows(rd, 'k', z, desc)
    np.testing.assert_array_equal(rows, flags[y0:y0 + n])


def test_variance_offset_is_past_the_compressed_heap():
    flags = _flags()
    data = _mef(flags)
    rd = _BytesReader(data)
    _, offs = _spherex._hdu_offsets(rd, 'dr1/level2/w/v/2/x.fits')
    with fits.open(io.BytesIO(data)) as h:
        var = np.asarray(h['VARIANCE'].data)
    off, bitpix, z = offs['VARIANCE']
    nx = flags.shape[1]
    got = np.frombuffer(data[off:off + nx * 4 * 3], dtype=_spherex._FITS_DTYPE[bitpix])
    np.testing.assert_array_equal(got.reshape(3, nx), var[:3])


def test_plain_flags_still_read_as_pixels():
    flags = _flags()
    rd = _BytesReader(_mef(flags, compress=False))
    _, offs = _spherex._hdu_offsets(rd, 'qr2/level2/w/v/3/x.fits')
    assert offs['FLAGS'][2] is None


def test_non_row_tiles_are_refused_loudly():
    flags = _flags()
    rd = _BytesReader(_mef(flags, tile=(4, flags.shape[1])))
    with pytest.raises(_spherex.UnsupportedLayout):
        _spherex._hdu_offsets(rd, 'dr1/level2/w/v/4/x.fits')


def test_cached_layout_follows_each_files_own_heap():
    """The compressed FLAGS heap is a per-FILE size, so VARIANCE moves from file
    to file within one layout variant (DR1: 28376640 / 28221120 / 28108800 B
    for three D3 files of one l2b-v27 variant). A layout cached from the first
    file must not place the second file's VARIANCE."""
    key = "dr1/level2/w/v/5/{}.fits"
    a = _mef(_flags())
    busy = _flags()
    busy[:] = np.random.default_rng(3).integers(0, 1 << 30, busy.shape)   # incompressible
    b = _mef(busy)
    assert len(a) != len(b)
    _spherex._hdu_offsets(_BytesReader(a), key.format("a"))           # walks, caches
    calls = []

    class _Counting(_BytesReader):
        def get(self, k, ranges):
            calls.append(len(ranges))
            return super().get(k, ranges)
    _, offs = _spherex._hdu_offsets(_Counting(b), key.format("b"))
    assert calls == [2]                       # IMAGE + FLAGS headers, one request
    with fits.open(io.BytesIO(b)) as h:
        var = np.asarray(h["VARIANCE"].data)
    off, bitpix, _ = offs["VARIANCE"]
    nx = busy.shape[1]
    got = np.frombuffer(b[off:off + nx * 4 * 3], dtype=_spherex._FITS_DTYPE[bitpix])
    np.testing.assert_array_equal(got.reshape(3, nx), var[:3])
    # and the first file still reads right from the same cache
    _, offs_a = _spherex._hdu_offsets(_BytesReader(a), key.format("a"))
    with fits.open(io.BytesIO(a)) as h:
        var_a = np.asarray(h["VARIANCE"].data)
    off, bitpix, _ = offs_a["VARIANCE"]
    got = np.frombuffer(a[off:off + nx * 4 * 3], dtype=_spherex._FITS_DTYPE[bitpix])
    np.testing.assert_array_equal(got.reshape(3, nx), var_a[:3])
