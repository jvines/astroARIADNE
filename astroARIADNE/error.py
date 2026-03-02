"""Error handling module."""
import logging
import sys
import traceback

logger = logging.getLogger(__name__)


class Error(Exception):
    """Base class for exceptions in this module."""

    def __repr__(self):
        """Error identification for logging."""
        return self.errorname

    def __str__(self):
        """Error identification for logging."""
        return self.errorname

    def __raise__(self):
        """Raise an exception and print the error message."""
        self.warn()
        sys.exit()

    def warn(self):
        """Log error message."""
        logger.error('%s: %s', self, self.message)

    def log(self, out):
        """Log the error."""
        log_f = open(out, 'a')
        log_f.write(self.message)
        log_f.close()

    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes
    ----------
    message -- explanation of the error

    """

    def __init__(self, message):
        self.errorname = 'InputError'
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
    Type 3 occurs if radius is 0
    Type 4 occurs if parallax is invalid

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
        if type == 2:
            self.message = f'The default prior for {par} is not permitted.'
            self.message += ' Allowed priors are normal, truncnorm, uniform and'
            self.message += ' default.'
        if type == 3:
            self.message = 'No radius found in Gaia, using default'
            self.message += ' radius prior.'
        if type == 4:
            self.message = 'The Gaia parallax is negative. Try fitting for the'
            self.message += ' normalization constant instead or make sure to'
            self.message += ' use a custom prior!'


class CatalogWarning(Error):
    """Exception raised for catalog errors.

    Does not terminate program. It only acts as a warning.
    Attributes
    ----------
    in_inst -- input instance
    exp_inst -- expected instance
    message -- explanation of the error

    Notes
    -----
    Type 0 means parallax is invalid
    Type 1 means parameter not found
    Type 2 means magnitude not found
    Type 3 means uncertainty not found
    Type 4 means uncertainty is 0
    Type 5 means star is not available in the catalog
    Type 6 means the selected magnitude was already retrieved
    Type 7 means catalog was manually skipped
    Type 8 means the given entry is either of bad quality or is a galaxy
    Type 9 means no EDR3 distance was found.

    """

    def __init__(self, par, type):
        self.errorname = 'CatalogWarning'
        if type == 0:
            if par == 'masked':
                self.message = 'Gaia DR3 parallax is masked (no astrometric solution)'
            else:
                self.message = 'Gaia DR3 parallax is non-positive: ' + str(par)
        if type == 1:
            self.message = 'Gaia DR3 parameter "' + str(par) + '" not found -- skipping'
        if type == 2:
            self.message = 'Magnitude ' + str(par) + ' not found in catalog -- skipping'
        if type == 3:
            self.message = 'No uncertainty for ' + str(par) + ' -- assigning default error'
        if type == 4:
            self.message = 'Zero uncertainty for ' + str(par) + ' -- assigning default error'
        if type == 5:
            self.message = 'Star not found in catalog ' + str(par) + ' -- skipping'
        if type == 6:
            self.message = 'Filter ' + str(par) + ' already retrieved -- skipping duplicate'
        if type == 7:
            self.message = 'Catalog ' + str(par) + ' skipped (user-requested)'
        if type == 8:
            self.message = ('Catalog ' + str(par)
                            + ': entry rejected (extended source, bad quality, or contaminated)')
        if type == 9:
            self.message = 'Unable to find distance in Bailer-Jones Gaia EDR3!'
            self.message += ' Calculating from Parallax instead.'

    def warn(self):
        """Log warning message."""
        logger.warning('%s', self.message)


class DynestyError(Error):
    """Exception raised when dynesty crashes."""

    def __init__(self, out, mod, e):
        self.errorname = 'DynestyError'
        self.message = 'ERROR OCCURRED DURING DYNESTY RUN.\n'
        self.message += f'WHILE FITTING MODEL {mod}\n'
        self.message += f'DUMPING `sampler.results` TO {out}\n'
        self.message += f'ERROR READS:\n{traceback.format_exc()}'


class StarWarning(Error):
    """Warnings for the Star class.

    Notes
    -----
    Type 0 means the warning is input detected.
    Type 1 means offline is true but no mag dict was given.
    Type 2 means the extinction map returned NaN.
    """

    def __init__(self, par, type):
        self.errorname = 'StarWarning'
        if type == 0:
            self.message = f'{par} input detected.'
            self.message += ' Overriding coordinate search.'
        if type == 1:
            self.message = 'You are running ARIADNE offline but no magnitudes '
            self.message += 'were given. Make sure you are providing a valid '
            self.message += 'mag_dict.'
        if type == 2:
            self.message = 'The selected extinction map returned a NaN! '
            self.message += 'Reverting back to the SFD map!'

    def warn(self):
        logger.warning('%s', self.message)

    def __raise__(self):
        """Raise an exception and log the error message."""
        logger.error('%s', self.message)
        sys.exit()
