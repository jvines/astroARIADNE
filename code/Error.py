"""Error handling module."""
import sys


class Error(Exception):
    """Base class for exceptions in this module."""

    def __repr__(self):
        """Error identification for logging."""
        return self.errorname

    def __str__(self):
        """Error identification for logging."""
        return self.errorname

    def raise_(self):
        """Raise an exception and print the error message."""
        try:
            raise self
        except Error:
            self.warn()
            sys.exit()

    def warn(self):
        """Print error message."""
        print('An exception was catched!', end=': ')
        print(self, end='\nError message: ')
        print(self.message)
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes
    ----------
    expression -- input expression in which the error occurred
    message -- explanation of the error

    """

    def __init__(self, expression, message):
        self.errorname = 'InputError'
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
        self.errorname = 'InstanceError'
        self.in_inst = in_inst.__repr__()
        self.exp_inst = exp_inst.__repr__()
        self.message = 'Input object is an instance of ' + in_inst
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
        self.errorname = 'PriorError'
        self.par = par
        self.type = type
        self.message = 'Parameter ' + par
        if type == 0:
            self.message += " isn't recognized. The allowed parameters are: "
            self.message += 'teff, logg, z, dist, rad, norm, Av, inflation'
        if type == 1:
            self.message += ' has been found alongside norm. Parameters rad '
            self.message += 'and dist are incompatible with norm.'