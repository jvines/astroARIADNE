"""Download the pre-computed spectra cache from Zenodo."""
import hashlib
import os
import sys
import urllib.request

from importlib.resources import files

# ── Zenodo record ──────────────────────────────────────────────────────
# Update these after publishing the record.
ZENODO_RECORD_ID = "19141566"
ZENODO_FILENAME = "spectra_cache.h5"
ZENODO_SHA256 = "0eea6b70d44ebb13291a95fe06a9a2554f5ef6272efb3eec6292acf393932df9"
# ──────────────────────────────────────────────────────────────────────


def _download_url():
    if ZENODO_RECORD_ID is None:
        raise RuntimeError(
            "Zenodo record ID not configured. "
            "Set ZENODO_RECORD_ID in astroARIADNE/fetch.py after publishing."
        )
    return (
        f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
        f"/files/{ZENODO_FILENAME}?download=1"
    )


def _dest_path():
    return str(files("astroARIADNE").joinpath("Datafiles", ZENODO_FILENAME))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ProgressReporter:
    """Simple download progress bar for stdout."""

    def __init__(self):
        self._last_pct = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(int(block_num * block_size * 100 / total_size), 100)
        if pct != self._last_pct:
            self._last_pct = pct
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            mb = block_num * block_size / 1e6
            total_mb = total_size / 1e6
            sys.stdout.write(
                f"\r  [{bar}] {pct:3d}%  ({mb:.0f} / {total_mb:.0f} MB)"
            )
            sys.stdout.flush()
            if pct == 100:
                sys.stdout.write("\n")


def fetch_spectra_cache(force=False):
    """Download the spectra cache HDF5 from Zenodo.

    Parameters
    ----------
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    str
        Absolute path to the downloaded file.
    """
    dest = _dest_path()

    if os.path.isfile(dest) and not force:
        print(f"Spectra cache already exists: {dest}")
        return dest

    url = _download_url()
    print(f"Downloading spectra cache from Zenodo (record {ZENODO_RECORD_ID})...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_ProgressReporter())
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    if ZENODO_SHA256 is not None:
        print("Verifying checksum...", end=" ")
        actual = _sha256(tmp)
        if actual != ZENODO_SHA256:
            os.remove(tmp)
            raise RuntimeError(
                f"SHA-256 mismatch: expected {ZENODO_SHA256}, got {actual}"
            )
        print("OK")

    os.replace(tmp, dest)
    print(f"Spectra cache saved to {dest}")
    return dest
