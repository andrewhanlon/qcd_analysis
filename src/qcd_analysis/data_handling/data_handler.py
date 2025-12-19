import enum
import abc

import random
import numpy as np
import gvar as gv

import scipy.linalg

import matplotlib.pyplot as plt


class SamplingMode(enum.Enum):
    NONE = 0
    JACKKNIFE = 1
    BOOTSTRAP = 2

SAMPLING_MODE = SamplingMode.JACKKNIFE
NUM_JACK = None
BOOTSTRAPS = None

def turn_off_resampling():
    global SAMPLING_MODE
    SAMPLING_MODE = SamplingMode.NONE

def set_to_jackknife():
    global SAMPLING_MODE, BOOTSTRAPS
    SAMPLING_MODE = SamplingMode.JACKKNIFE
    BOOTSTRAPS = None

def set_to_bootstrap(num_confs, rebin, num_samples, seed, skip=0):
    global SAMPLING_MODE, NUM_SAMPLES, BOOTSTRAPS
    SAMPLING_MODE = SamplingMode.BOOTSTRAP
    num_bins = num_confs // rebin
    BOOTSTRAPS = np.zeros((num_samples, num_bins), dtype=np.int32)
    random.seed(seed)
    for samp_i in range(num_samples):
        for bin_i in range(num_bins):
            for skip_i in range(skip):
                random.randrange(num_bins)
            bin_i_map = random.randrange(num_bins)
            BOOTSTRAPS[samp_i,bin_i] = bin_i_map

    NUM_SAMPLES = BOOTSTRAPS.shape[0]
    print(BOOTSTRAPS)

def set_to_bootstrap_manual(bootstraps):
    global SAMPLING_MODE, NUM_SAMPLES, BOOTSTRAPS
    SAMPLING_MODE = SamplingMode.BOOTSTRAP
    BOOTSTRAPS = bootstraps
    NUM_SAMPLES = bootstraps.shape[0]

def get_sampling_mode():
    return SAMPLING_MODE

def get_num_samples():
    if get_sampling_mode() == SamplingMode.BOOTSTRAP:
        return BOOTSTRAPS.shape[0]
    elif get_sampling_mode() == SamplingMode.NONE:
        return 0
    else:
        return NUM_JACK

REBIN = 1
def set_rebin(rebin):
    global REBIN
    REBIN = rebin

def get_rebin():
    return REBIN

class ComplexArg(enum.Enum):
    REAL = 0
    IMAG = 1


class Data:

    def __init__(self, input_data, bins=False):
        if isinstance(input_data, gv.GVar):
            self._bins = None
            self._create_samples_from_gvar(input_data, bins)
        elif isinstance(input_data, (int, float, complex)) and not isinstance(input_data, bool):
            self._bins = np.array([input_data])
            self._samples = np.array([input_data])
            turn_off_resampling()
        elif bins:
            self._bins = input_data
            self._create_samples()
        else:
            self._bins = None
            # check input samples
            if len(input_data.shape) != 1:
                raise TypeError("Samplings must have one index")

            if get_sampling_mode() == SamplingMode.BOOTSTRAP and input_data.shape[0] != get_num_samples()+1:
                raise TypeError("Number of samples does not match current bootstrap settings")

            self._samples = input_data

    @property
    def bins(self):
        return self._bins

    @property
    def samples(self):
        return self._samples

    def _create_samples_from_gvar(self, gvar_data, num_jack):
        if SAMPLING_MODE == SamplingMode.JACKKNIFE:
            self._samples = np.zeros(num_jack+1)
            self._samples[0] = gvar_data.mean
            self._samples[1:] = np.random.normal(gvar_data.mean, gvar_data.sdev / (num_jack - 1)**0.5, num_jack)
        else:
            num_samples = get_num_samples()
            self._samples = np.zeros(num_samples+1)
            self._samples[0] = gvar_data.mean
            self._samples[1:] = np.random.normal(gvar_data.mean, gvar_data.sdev, num_samples)

    def _create_samples(self):
        num_bins = len(self._bins)
        rebin = get_rebin()
        num_rebin_bins = num_bins // rebin
        rebin_bins = np.zeros(num_rebin_bins, dtype=self._bins.dtype)
        for rebin_ci in range(num_rebin_bins):
            for ci in range(rebin_ci*rebin, rebin_ci*rebin + rebin):
                rebin_bins[rebin_ci] += self._bins[ci]

            rebin_bins[rebin_ci] /= rebin

        ensemble_sum = 0.
        for bin_value in rebin_bins:
            ensemble_sum += bin_value

        if SAMPLING_MODE == SamplingMode.JACKKNIFE:
            global NUM_JACK
            NUM_JACK = num_rebin_bins

            self._samples = np.zeros(num_rebin_bins+1, dtype=rebin_bins.dtype)
            self._samples[0] = ensemble_sum / num_rebin_bins

            for sample_i, bin_value in enumerate(rebin_bins, 1):
                self._samples[sample_i] = (ensemble_sum - bin_value) / (num_rebin_bins - 1)

        else:
            if num_rebin_bins != BOOTSTRAPS.shape[1]:
                print(num_rebin_bins)
                print(BOOTSTRAPS.shape)
                raise TypeError("Number of bins does not match number of configs")

            num_samples = get_num_samples()

            self._samples = np.zeros(num_samples+1, dtype=rebin_bins.dtype)
            self._samples[0] = ensemble_sum / num_rebin_bins
            for sample_i in range(num_samples):
                boots_map = BOOTSTRAPS[sample_i,:]
                sample_ave = 0.
                for rebin_i in range(num_rebin_bins):
                    sample_ave += rebin_bins[boots_map[rebin_i]]

                self._samples[sample_i+1] = sample_ave / num_rebin_bins


    @property
    def num_samples(self):
        return len(self._samples) - 1

    @property
    def mean(self):
        return self.ensemble_average()

    @property
    def sdev(self):
        return self.error()

    @property
    def var(self):
        return (self.error()**2)

    def ensemble_average(self):
        return self.samples[0]

    def sample_average(self):
        return np.mean(self.samples[1:])

    def error(self, asymmetric=False):
        if SAMPLING_MODE == SamplingMode.JACKKNIFE:
            return (self.num_samples - 1)**0.5 * np.std(self.samples[1:], ddof=0)

        elif SAMPLING_MODE == SamplingMode.NONE:
            return 0.

        elif asymmetric:
            sorted_samples = np.sort(self.samples[1:])
            percentile_index = int(round(self.num_samples * 0.16))
            error = (self.samples[0] - sorted_samples[percentile_index], sorted_samples[-percentile_index] - self.samples[0])
            return (error[0], error[1])

        else:
            return np.std(self.samples[1:], ddof=1)

    def invert_samples(self):
        temp_samples = np.zeros(self._samples.shape, dtype=self._samples.dtype)
        temp_samples[0] = self._samples[0]
        for i in range(self.num_samples):
            temp_samples[i+1] = self._samples[0] - (self._samples[i+1] - self._samples[0])

        self._samples = temp_samples


    def plot_samples(self, plot_file, num_bins=20):
        fig, ax = plt.subplots()
        ax.hist(self.samples[1:], bins=num_bins)

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def plot_bins(self, plot_file, num_bins=20):
        fig, ax = plt.subplots()
        ax.hist(self._bins[:], bins=num_bins)

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()


    def gvar(self):
        return gv.gvar(self.mean, self.sdev)

    def exp(self):
        return Data(np.exp(self.samples))

    def cosh(self):
        return Data(np.cosh(self.samples))

    def log(self):
        return Data(np.log(self.samples))

    def sqrt(self):
        return Data(np.sqrt(self.samples))

    def conj(self):
        return Data(np.conj(self.samples))

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Data(self.samples + other.samples)
        return Data(self.samples + other)

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            return Data(other.samples + self.samples)
        return Data(other + self.samples)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return Data(self.samples - other.samples)
        return Data(self.samples - other)

    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            return Data(other.samples - self.samples)
        return Data(other - self.samples)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return Data(self.samples * other.samples)
        return Data(self.samples * other)

    def __rmul__(self, other):
        if isinstance(other, self.__class__):
            return Data(other.samples * self.samples)
        return Data(other * self.samples)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return Data(self.samples / other.samples)
        return Data(self.samples / other)

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            return Data(other.samples / self.samples)
        return Data(other / self.samples)

    def __pow__(self, other):
        return Data(pow(self.samples, other))

    def __rpow__(self, other):
        return Data(pow(other, self.samples))

    def __neg__(self):
        return Data(-self.samples)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.samples, other.samples)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


    @property
    def real(self):
        return Data(self.samples.real)

    @property
    def imag(self):
        return Data(self.samples.imag)

    def __str__(self):
        return str(self.gvar())

    def __call__(self, samp_i):
        return self.samples[samp_i]


class DataType(metaclass=abc.ABCMeta):

    _cov = None
    _cov_chol_lower = None

    @property
    @abc.abstractmethod
    def data_name(self):
        pass

    @property
    @abc.abstractmethod
    def samples(self):
        """ gets samples of the data

        Returns: Must be of shape (num_samples+1, num_observables)
        """
        pass

    @property
    @abc.abstractmethod
    def num_samples(self):
        pass

    @property
    @abc.abstractmethod
    def independent_variables_values(self):
        pass

    @property
    @abc.abstractmethod
    def num_data(self):
        pass

    def set_covariance(self, uncorrelated=False, remove_correlations=[]):
        samples = self.samples

        if get_sampling_mode() == SamplingMode.JACKKNIFE:
            cov_factor = 1. - 1./self.num_samples
        else:
            cov_factor = 1./(self.num_samples - 1)

        diffs = samples[1:,:] - np.mean(samples[1:,:], axis=0)
        cov = cov_factor * np.tensordot(diffs.conj(), diffs, axes=(0,0))

        # remove correlations
        for i, j in remove_correlations:
            if i == j:
                continue

            cov[i,j] = 0.
            cov[j,i] = 0.

        if uncorrelated:
            for i in range(cov.shape[0]):
                for j in range(cov.shape[1]):
                    if i == j:
                        continue

                    cov[i,j] = 0.
                    cov[j,i] = 0.

        self._cov = cov
        self._cov_chol_lower = None


    @property
    def cov(self):
        if self._cov is None:
            self.set_covariance()
        return self._cov

    @property
    def cov_chol_lower(self):
        if self._cov_chol_lower is None:
            self._cov_chol_lower = scipy.linalg.cholesky(self.cov, lower=True, check_finite=False)
        return self._cov_chol_lower


    def get_residuals(self, samp_i, fit_func, params):
        residuals = fit_func(self.independent_variables_values, params) - self.samples[samp_i,:]
        if fit_func.num_priors:
            priored_residuals = fit_func.get_priored_residuals(params, samp_i)
            residuals = np.concatenate((residuals, priored_residuals))

        return residuals

    def get_weighted_residuals(self, samp_i, fit_func, params):
        cov_chol_lower = self.cov_chol_lower
        if fit_func.num_priors:
            priored_cov_chol_lower = np.zeros((cov_chol_lower.shape[0]+fit_func.num_priors,
                                              cov_chol_lower.shape[1]+fit_func.num_priors), dtype=np.float64)
            priored_cov_chol_lower[:cov_chol_lower.shape[0],:cov_chol_lower.shape[1]] = cov_chol_lower
            priored_sdevs = fit_func.get_priored_sdevs(params)
            for pi in range(fit_func.num_priors):
                priored_cov_chol_lower[cov_chol_lower.shape[0]+pi, cov_chol_lower.shape[1]+pi] = priored_sdevs[pi]

            cov_chol_lower = priored_cov_chol_lower

        residuals = self.get_residuals(samp_i, fit_func, params)

        return scipy.linalg.solve_triangular(cov_chol_lower, residuals, lower=True, check_finite=False)

    def get_weighted_residuals_func(self, samp_i, fit_func):
        def residuals_func(params):
            return self.get_weighted_residuals(samp_i, fit_func, params)
        return residuals_func



class MultiDataType(DataType):

    def __init__(self, data_types: list[DataType]):
        self._data_types = data_types

    @property
    def data_name(self):
        return [data_type.data_name for data_type in self.data_types]

    @property
    def num_data_types(self):
        return len(self._data_types)

    @property
    def data_types(self):
        return self._data_types

    @property
    def num_data(self):
        return sum([data_type.num_data for data_type in self._data_types])

    @property
    def samples(self):
        data_type_samps = [data_type.samples for data_type in self._data_types]
        return np.concatenate(data_type_samps, axis=1)

    @property
    def num_samples(self):
        return self._data_types[0].num_samples

    @property
    def independent_variables_values(self):
        indep_var_list = list()
        for data_type in self._data_types:
            indep_var_list.append(data_type.independent_variables_values)

        return indep_var_list

class Prior:
    def __init__(self, prior, scale=1):
        """
        Args:
          prior, either DataType or 2-tuple
        """
        if type(prior) is tuple:
            self._mean = prior[0]
            self._sdev = prior[1]
            self._samples = None
        else:
            self._samples = prior

        self._scale = scale

    @property
    def mean(self):
        if hasattr(self, '_mean'):
            return self._mean
        else:
            return prior.mean

    @property
    def sdev(self):
        if hasattr(self, '_sdev'):
            return self._scale * self._sdev
        else:
            return self._scale * self._samples.sdev

    def get_prior(self, samp_i):
        if hasattr(self, '_mean'):
            return self._mean
        else:
            return self._samples.samples[samp_i]

    def gvar(self):
        return gv.gvar(self.mean, self.sdev)

    def __str__(self):
        return str(self.gvar())



class FitFunction(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def fit_name(self):
        pass

    @property
    @abc.abstractmethod
    def params(self):
        pass

    @property
    def num_params(self):
        return len(self.params)

    @property
    def init_guesses(self):
        if hasattr(self, '_init_guesses'):
            return self._init_guesses
        return dict()

    @init_guesses.setter
    def init_guesses(self, in_init_guesses):
        self._init_guesses = in_init_guesses

    @property
    def priors(self):
        if hasattr(self, '_priors'):
            return self._priors
        return dict()

    @priors.setter
    def priors(self, in_priors):
        for param in in_priors.keys():
            if param not in self.params:
                raise ValueError(f"Not such param {param}")
        self._priors = in_priors

    def add_priors(self, priors):
        if not hasattr(self, '_priors'):
            self._priors = dict()

        for param, prior in priors.items():
            if param not in self.params:
                raise ValueError(f"Not such param {param}")
            self._priors[param] = prior

    @property
    def num_priors(self):
        return len(self.priors)

    @property
    def priored_params(self):
        return self.priors.keys()

    @abc.abstractmethod
    def function(self, x, p):
        pass

    '''
    def __call__(self, x, p):
        if type(p) is dict:
            return self.function(x, p)

        elif isinstance(p, np.ndarray) or type(p) is list:
            param_dict = dict()
            for pi, p_name in enumerate(self.params):
                param_dict[p_name] = p[pi]

            return self.function(x, param_dict)

        else:
            raise TypeError("Invalid parameters to FitFunction")
    '''
    def __call__(self, x, p):
        return self.function(x, p)


    def get_priored_residuals(self, p, samp_i=0):
        priored_residuals = np.zeros((self.num_priors,), dtype=np.float64)
        if type(p) is dict:
            for pi, (param, prior) in enumerate(self.priors.items()):
                priored_residuals[pi] = p[param] - prior.get_prior(samp_i)

        elif isinstance(p, np.ndarray) or type(p) is list:
            prior_i = 0
            for param_i, param in enumerate(self.params):
                if param in self.priors:
                    priored_residuals[prior_i] = p[param_i] - self.priors[param].get_prior(samp_i)
                    prior_i += 1

        else:
            raise TypeError("Invalid parameters to FitFunction")

        return priored_residuals

    def get_priored_sdevs(self, p):
        priored_sdevs = np.zeros((self.num_priors,), dtype=np.float64)
        if type(p) is dict:
            for pi, (param, prior) in enumerate(self.priors.items()):
                priored_sdevs[pi] = prior.sdev

        elif isinstance(p, np.ndarray) or type(p) is list:
            prior_i = 0
            for param_i, param in enumerate(self.params):
                if param in self.priors:
                    priored_sdevs[prior_i] = self.priors[param].sdev
                    prior_i += 1

        else:
            raise TypeError("Invalid parameters to FitFunction")

        return priored_sdevs


class MultiFitFunction(FitFunction):

    def __init__(self, fit_functions: list[FitFunction]):
        self._fit_functions = fit_functions

    @property
    def fit_name(self):
        _fit_name = "Multi_fit"
        for fit_func in self.fit_functions:
            _fit_name += f"-{fit_func.fit_name}"

        return _fit_name

    @property
    def params(self):
        _params = list()
        for fit_func in self.fit_functions:
            for param in fit_func.params:
                if param not in _params:
                    _params.append(param)

        return _params

    @property
    def init_guesses(self):
        '''
        if not self._init_guesses:
            for fit_func in self.fit_functions:
                for param, guess in fit_func.init_guesses.items():
                    if param in self._init_guesses and guess != self._init_guesses[param]:
                        print(f"Warning: conflicthing init guess for param {param}")

                    self._init_guesses[param] = guess

        return self._init_guesses
        '''

        _init_guesses = dict()
        for fit_func in self.fit_functions:
            for param, guess in fit_func.init_guesses.items():
                if param in _init_guesses and guess != _init_guesses[param]:
                    print(f"Warning: conflicthing init guess for param {param}")

                _init_guesses[param] = guess
        
        return _init_guesses

    '''
    @property
    def num_priors(self):
        return len(self.priors)
    '''

    @property
    def priors(self):
        '''
        if not self._priors:
            for fit_func in self.fit_functions:
                for param, prior in fit_func.priors.items():
                    if param in self._pirors and prior != self._priors[param]:
                        print(f"Warning: conflicthing prior for param {param}")

                    self._priors[param] = prior

        return self._priors
        '''

        _priors = dict()
        for fit_func in self.fit_functions:
            for param, prior in fit_func.priors.items():
                if param in _priors and prior != _priors[param]:
                    print(f"Warning: conflicthing prior for param {param}")

                _priors[param] = prior

        return _priors

    @property
    def fit_functions(self):
        return self._fit_functions

    @property
    def num_functions(self):
        return len(self.fit_functions)

    def function(self, x, p):
        """
        Args:
          x: list of size of self.fit_functions
          p: list of param values

        Return:
          list of size equal to total elements in x
        """
        x_start_i = 0
        if self.lsqfit_mode:
            function_values = np.empty(sum([d.size for d in x]), dtype=object)
        else:
            function_values = np.empty(sum([d.size for d in x]), dtype=np.float64)

        for data, fit_func in zip(x, self.fit_functions):
            param_values = list()
            for param in fit_func.params:
                param_values.append(p[self.params.index(param)])

            function_values[x_start_i:x_start_i+len(data)] = fit_func(data, param_values)

            x_start_i += len(data)

        return function_values

    def __call__(self, x, p):
        return self.function(x, p)

    def get_combined_fit_function(self, input_data):
        def combined_fit_function(x, p):
            x_start_i = 0
            function_values = np.zeros(input_data.num_data)
            for data_type, fit_func in zip(input_data.data_types, self.fit_functions):
                param_values = list()
                for param in fit_func.params:
                    param_values.append(p[self.params.index(param)])

                function_values[x_start_i:x_start_i+data_type.num_data] += fit_func(x[x_start_i:x_start_i+data_type.num_data], param_values)
                x_start_i += data_type.num_data

            return function_values

        return combined_fit_function
