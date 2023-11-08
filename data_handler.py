import enum
import abc

import random
import numpy as np
import gvar as gv

import scipy.linalg

import matplotlib.pyplot as plt
#plt.rc('font', family='serif', serif=['Times'], size=55)
#plt.rc('text', usetex=True)
plt.style.use('/home/ahanlon/code/lattice/qcd_analysis/plots.mplstyle')


class SamplingMode(enum.Enum):
    JACKKNIFE = 1
    BOOTSTRAP = 2

SAMPLING_MODE = SamplingMode.JACKKNIFE
NUM_JACK = None
BOOTSTRAPS = None

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
        for _ in range(num_bins):
            for skip_i in range(skip):
                random.randrange(num_bins)
            bin_i = random.randrange(num_bins)
            BOOTSTRAPS[samp_i,bin_i] += 1

    NUM_SAMPLES = BOOTSTRAPS.shape[0]

def get_sampling_mode():
    return SAMPLING_MODE

def get_num_samples():
    if get_sampling_mode() == SamplingMode.BOOTSTRAP:
        return BOOTSTRAPS.shape[0]
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

        if SAMPLING_MODE == SamplingMode.JACKKNIFE:
            global NUM_JACK
            NUM_JACK = num_rebin_bins
            ensemble_ave = 0.
            for bin_value in rebin_bins:
                ensemble_ave += bin_value

            self._samples = np.zeros(num_rebin_bins+1, dtype=rebin_bins.dtype)
            self._samples[0] = ensemble_ave / num_rebin_bins

            for sample_i, bin_value in enumerate(rebin_bins, 1):
                self._samples[sample_i] = (ensemble_ave - bin_value) / (num_rebin_bins - 1)

        else:
            if num_bins != BOOTSTRAPS.shape[1]:
                print(num_bins)
                print(BOOTSTRAPS.shape)
                raise TypeError("Number of bins does not match number of configs")

            num_samples = get_num_samples()

            ensemble_ave = 0.
            for bin_value in rebin_bins:
                ensemble_ave += bin_value

            self._samples = np.zeros(num_samples+1, dtype=rebin_bins.dtype)
            self._samples[0] = ensemble_ave / num_rebin_bins
            for sample_i in range(num_samples):
                sample_ave = 0.
                for rebin_i in range(num_rebin_bins):
                    sample_ave += BOOTSTRAPS[sample_i,rebin_i] * rebin_bins[rebin_i]

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

    def ensemble_average(self):
        return self.samples[0]

    def sample_average(self):
        ave = 0.
        for sample in self.samples[1:]:
            ave += sample

        return ave / self.num_samples

    def error(self):
        if SAMPLING_MODE == SamplingMode.JACKKNIFE:
            return (self.num_samples - 1)**0.5 * np.std(self.samples[1:])

        else:
            sorted_samples = np.sort(self.samples[1:])
            percentile_index = int(round(self.num_samples * 0.16))
            error = (self.samples[0] - sorted_samples[percentile_index], sorted_samples[-percentile_index] - self.samples[0])
            return 0.5*(error[0] + error[1])

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


    def gvar(self):
        return gv.gvar(self.mean, self.sdev)

    def exp(self):
        return Data(np.exp(self.samples))

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

    @property
    @abc.abstractmethod
    def data_name(self):
        pass

    @abc.abstractmethod
    def get_samples(self):
        """ gets samples of the data

        Returns: Must be of shape (num_samples+1, num_observables)
        """
        pass

    @property
    @abc.abstractmethod
    def num_samples(self):
        pass

    @abc.abstractmethod
    def get_independent_data(self):
        pass

    @abc.abstractmethod
    def get_organized_independent_data(self):
        pass

    @property
    def num_data(self):
        return len(self.get_organized_independent_data())

    def get_covariance_lsqfit(self, correlated=True):
        samples = self.get_samples()

        if get_sampling_mode() == SamplingMode.JACKKNIFE:
            cov_factor = 1. - 1./self.num_samples
        else:
            cov_factor = 1./(self.num_samples - 1)

        diffs = samples[1:,:] - samples[0,:]

        if correlated:
            return cov_factor * np.tensordot(diffs.conj(), diffs, axes=(0,0))
        else:
            return cov_factor * np.diag(np.einsum('ij,ji->i', diffs.conj().T, diffs))

    def set_covariance(self, remove_correlations=[]):
        samples = self.get_samples()

        if get_sampling_mode() == SamplingMode.JACKKNIFE:
            cov_factor = 1. - 1./self.num_samples
        else:
            cov_factor = 1./(self.num_samples - 1)

        diffs = samples[1:,:] - samples[0,:]
        cov = cov_factor * np.tensordot(diffs.conj(), diffs, axes=(0,0))

        # remove correlations
        for i, j in remove_correlations:
            if i == j:
                continue

            cov[i,j] = 0.
            cov[j,i] = 0.

        self._cov = cov
        self._cov_chol_lower = scipy.linalg.cholesky(cov, lower=True, check_finite=False)


    def get_covariance(self):
        if self._cov is None:
            self.set_covariance()
        return self._cov


    def get_residuals(self, samp_i, fit_func, params):
        residuals = fit_func(self.get_independent_data(), params) - self.get_samples()[samp_i,:]
        '''
        if fit_func.num_priors:
            priored_residuals = fit_func.get_priored_residuals(params)
            residuals = np.concatenate(residuals, priored_residuals)
        '''

        return residuals

    def get_weighted_residuals(self, samp_i, fit_func, params):
        cov_chol_lower = self._cov_chol_lower
        '''
        if fit_func.num_priors:
            priored_cov_chol_lower = np.zeros((cov_chol_lower.shape[0]+len(fit_func.num_priors),
                                              cov_chol_lower.shape[1]+len(fit_func.num_priors)), dtype=np.float64)
            priored_cov_chol_lower[:cov_chol_lower.shape[0],:cov_chol_lower.shape[1]] = cov_chol_lower
            for prior in priors:
                priored_sdevs = fit_func.get_priored_sdevs(params)
                for pi in range(fit_func.num_priors):
                    priored_cov_chol_lower[cov_chol_lower.shape[0]+pi, cov_chol_lower.shape[1]+pi] = priored_sdevs[pi]

            cov_chol_lower = priored_cov_chol_lower
        '''

        residuals = self.get_residuals(samp_i, fit_func, params)
        '''
        if fit_func.num_priors:
            priored_residuals = fit_func.get_priored_residuals(params)
            residuals = np.concatenate(residuals, priored_residuals)
        '''

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

    def get_samples(self):
        data_type_samps = [data_type.get_samples() for data_type in self._data_types]
        return np.concatenate(data_type_samps, axis=1)

    @property
    def num_samples(self):
        return self._data_types[0].num_samples

    def get_independent_data(self):
        indep_data_list = list()
        for data_type in self._data_types:
            indep_data_list.append(data_type.get_independent_data())

        return np.concatenate(indep_data_list)

    def get_organized_independent_data(self):
        indep_data_list = list()
        for data_type in self._data_types:
            indep_data_list.append(data_type.get_organized_independent_data())

        return np.concatenate(indep_data_list)


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
            return self._scale * prior.sdev

    def get_prior(self, samp_i):
        if hasattr(self, '_mean'):
            return self._mean
        else:
            return self._samples.samples[samp_i]



class FitFunction(metaclass=abc.ABCMeta):

    _priors = dict()

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

    def add_priors(self, priors):
        for param, prior in priors.items():
            _priors[param] = prior

    @property
    def priors(self):
        return self._priors

    @property
    def num_priors(self):
        return len(self._priors)

    @abc.abstractmethod
    def function(self, x, p):
        pass

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

    def get_priored_residuals(p, samp_i=0):
        priored_residuals = np.zeros((self.num_priors,), dtype=np.float64)
        if type(p) is dict:
            for pi, (param, prior) in enumerate(self.priors.items()):
                priored_residuals[pi] = p[param] - prior.get_prior(samp_i)

        elif isinstance(p, np.ndarray) or type(p) is list:
            prior_i = 0
            for parami, param in enumerate(self.params):
                if param in self.priors:
                    priored_residuals[prior_i] = p[parami] - self.priors[param].get_prior(samp_i)

        else:
            raise TypeError("Invalid parameters to FitFunction")

        return priored_residuals


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
            _params.extend(fit_func.params)

        return _params

    @property
    def num_params(self):
        return sum([func.num_params for func in self.fit_functions])

    @property
    def fit_functions(self):
        return self._fit_functions

    @property
    def num_functions(self):
        return len(self.fit_functions)

    @property
    def num_priors(self):
        return sum([func.num_priors for func in self.fit_functions])

    def add_priors(self, priors):
        raise TypeError("Add priors to individual fit functions")

    def function(self, x, p):
        """
        Args:
          x: list of size of self.fit_functions
          p: list of size of self.fit_functions

        Return:
          list of size of self.fit_functions
        """

        func_results = list()
        for funci, xi, pi in zip(self.fit_functions, x, p):
            func_results.append(funci(xi, pi))

        return func_results

    def __call__(self, x, p):
        return self.function(x, p)

    def get_combined_fit_function(self, input_data):
        def combined_fit_function(x, p):
            start_i = 0
            function_values = np.zeros(input_data.num_data)
            for data_type, fit_func in zip(input_data.data_types, self.fit_functions):
                function_values[start_i:start_i+data_type.num_data] += \
                    fit_func(x[start_i:start_i+data_type.num_data], p[start_i:start_i+data_type.num_data])
                start_i += data_type.num_data

            return function_values

        return combined_fit_function
