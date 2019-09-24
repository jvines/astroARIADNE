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
