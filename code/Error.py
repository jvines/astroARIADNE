"""Error handling module."""


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes
    ----------
    expression -- input expression in which the error occurred
    message -- explanation of the error

    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class InstanceError(Error):
    """Exception raised for instance errors.

    Attributes
    ----------
    in_inst -- input instance
    exp_inst -- expected instance
    message -- explanation of the error

    """

    def __init__(self, in_inst, exp_inst):
        self.in_inst = in_inst
        self.exp_inst = exp_inst
        self.message = 'Input object is an instance of ' + in_inst.__class__
        self.message += ' expected class is ' + exp_inst


class PriorError(Error):
    """Exception raised if problem arises with priors.

    Attributes
    ----------
    par -- parameter
    type -- error type, can be 0, 1
    message -- explanation of the error

    Notes
    -----
    Type 0 error means the parameter isn't recognized.
    Type 1 occurs if norm parameter is found alongside dist or rad.
    Type 2 occurs if the selected prior is illegal.
    """

    def __init__(self, par, type):
        self.message = 'Parameter ' + par
        if type == 0:
            self.message += " isn't recognized. The allowed parameters are: "
            self.message += 'teff, logg, z, dist, rad, norm, Av, inflation'
        if type == 1:
            self.message += ' has been found alongside norm. Parameters rad '
            self.message += 'and dist are incompatible with norm.'
