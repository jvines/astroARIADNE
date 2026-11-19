"""Fitter pools fork regardless of the process-wide start method (#58, PR #81).

Workers read the photometry, priors and interpolator from module globals set
just before the Pool is created. With spawned workers (macOS's default, and
what any later ``set_start_method('spawn', force=True)`` restores) those
globals don't exist and dynesty dies with ``NameError: name 'flux' is not
defined``.
"""
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import multiprocessing as mp
    from astroARIADNE import fitter, isochrone
    # Undo any import-time choice, as other packages or user code may.
    mp.set_start_method('spawn', force=True)
    assert mp.get_start_method() == 'spawn'

    def read_global(_):
        return fitter.flux_probe

    fitter.flux_probe = 42          # set after import, as fit_dynesty does
    for P in (fitter.Pool, isochrone.Pool):
        with P(2) as pool:
            # spawned workers die on the missing global and map() would hang
            got = pool.map_async(read_global, range(4)).get(timeout=60)
            assert got == [42] * 4, got
    print('ok')
""")


def test_pools_fork_under_spawn_default():
    r = subprocess.run([sys.executable, '-c', SCRIPT], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith('ok')


def test_import_leaves_global_start_method_alone():
    r = subprocess.run(
        [sys.executable, '-c',
         'import multiprocessing as mp; mp.set_start_method("spawn"); '
         'import astroARIADNE.fitter; print(mp.get_start_method())'],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == 'spawn'
