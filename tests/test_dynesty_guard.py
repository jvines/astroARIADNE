"""The fitter's guard on dynesty's kmeans2 (see fitter._guard_dynesty_kmeans).

Without it a live set with zero spread in one dimension reaches scipy's kmeans2
as inf (dynesty divides by the spread, check_finite=False) and segfaults.
"""
import numpy as np

import astroARIADNE.fitter  # noqa: F401  (installs the guard)
import dynesty.bounding as db


def test_nonfinite_input_refuses_split():
    pts = np.random.default_rng(1).random((200, 4))
    pts[:, 3] = np.inf
    centres, labels = db.kmeans2(pts, k=np.ones((2, 4)), iter=10, minit='matrix', check_finite=False)
    assert np.all(labels == 0)


def test_finite_input_still_clusters():
    rng = np.random.default_rng(2)
    pts = np.vstack([rng.normal(0, 0.1, (100, 2)), rng.normal(5, 0.1, (100, 2))])
    _, labels = db.kmeans2(pts, k=np.array([[0.0, 0.0], [5.0, 5.0]]), iter=10, minit='matrix')
    assert len(set(labels[:100])) == 1 and len(set(labels[100:])) == 1 and labels[0] != labels[150]


def test_zero_spread_dimension_keeps_parent_ellipsoid():
    rng = np.random.default_rng(3)
    pts = rng.random((500, 5))
    pts[:, 3] = 1.0 - 2.0 ** -53            # every live point at the same coordinate
    ell = db.bounding_ellipsoid(pts)
    ells = db._bounding_ellipsoids(pts, ell)
    assert len(ells) >= 1
