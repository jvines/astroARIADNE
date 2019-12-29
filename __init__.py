"""
ARIADNE is a module to easily fit SED models using nested sampling algorithms.

It allows to fit single models (Phoenix v2, BT-Settl, BT-Cond, BT-NextGen
Castelli & Kurucz 2004 and Kurucz 1993) or multiple models in a single run.
If multiple models are fit for, then ARIADNE automatically averages the
parameters posteriors as in the Bayesian Model Average framework. This
averages over the models and thus the averaged posteriors account for model
specific uncertainties.
"""
