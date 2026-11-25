"""The fit health of every grid reaches the saved BMA result.

``bayesian_model_average`` collects the per-grid diagnostics (prior-truncated
parameters, the mixing check from the distance posterior) and the evidence
errors.  ``save_bma`` builds the saved dictionary from scratch, so it has to
carry them over; otherwise BMA.pkl and ariadne_result.nc hold no fit health.
"""
import inspect

from astroARIADNE.fitter import Fitter


def test_fit_health_is_carried_into_the_saved_result():
    avgd = {
        'evidences_err': {'phoenix': 0.12, 'btsettl': 0.2},
        'per_model_diagnostics': {
            'phoenix': {'prior_flags': [{'param': 'rad', 'edge': 'lower'}],
                        'distance_mixing': {'mixed': False, 'shift': 2.3,
                                            'width': 0.4, 'reason': 'shift'}},
            'btsettl': {'prior_flags': [], 'distance_mixing': {'mixed': True}},
        },
    }
    out = {}
    Fitter._carry_fit_health(out, avgd)
    assert out['lnZerr'] == avgd['evidences_err']
    assert out['per_model_diagnostics']['phoenix']['distance_mixing'][
        'mixed'] is False
    assert out['per_model_diagnostics']['phoenix']['prior_flags'][0][
        'param'] == 'rad'


def test_missing_health_gives_empty_entries():
    out = {}
    Fitter._carry_fit_health(out, {})
    assert out['lnZerr'] == {} and out['per_model_diagnostics'] == {}


def test_save_bma_carries_the_fit_health():
    assert '_carry_fit_health(out, avgd)' in inspect.getsource(Fitter.save_bma)
