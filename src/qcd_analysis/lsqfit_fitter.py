import os
import numpy as np
import scipy
import scipy.linalg
import enum
import multiprocessing
import multiprocessing.shared_memory
import threadpoolctl
import types
import ctypes
import itertools

import gvar as gv
import lsqfit

from qcd_analysis import data_handler

DEBUG = False
NUM_PROCESSES = 2 if multiprocessing.cpu_count() <= 16 else multiprocessing.cpu_count() // 8

class FitInfo(enum.Enum):
    fitter: str
    fit_function: types.FunctionType
    independent_data: np.ndarray
    samples: np.ndarray
    cov: np.ndarray
    init_guesses: dict
    priors: dict
    svdcut: float
    fit_data: np.ndarray
    fit_chi2: np.ndarray
    fit_Q: np.ndarray

def get_shared_data(shape):
    d_size = np.dtype(np.float64).itemsize * np.prod(shape)
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=d_size)
    return shm

def release_shared_data(name):
    shm = multiprocessing.shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()


def fit_full(fit_info, params_map):
    dependent_data = gv.gvar(fit_info.samples[0,:], fit_info.cov)
    full_fit = lsqfit.nonlinear_fit(data=(fit_info.independent_data, dependent_data),
                                    fcn=fit_info.fit_function,
                                    p0=fit_info.init_guesses,
                                    prior=fit_info.priors[0],
                                    svdcut=fit_info.svdcut,
                                    fitter=fit_info.fitter,
                                    noise=(False,False),
                                    debug=DEBUG)

    shape = (len(fit_info.samples), len(params_map))
    fit_data = np.ndarray(shape=shape, dtype=np.float64, buffer=fit_info.fit_data.buf)
    for param_name, param_value in full_fit.pmean.items():
        fit_data[0,params_map[param_name]] = param_value

    shape = (len(fit_info.samples),)
    fit_chi2 = np.ndarray(shape=shape, dtype=np.float64, buffer=fit_info.fit_chi2.buf)
    fit_chi2[0] = full_fit.chi2

    fit_Q = np.ndarray(shape=shape, dtype=np.float64, buffer=fit_info.fit_Q.buf)
    fit_Q[0] = full_fit.Q

    return (full_fit.p, full_fit.chi2, full_fit.dof, full_fit.Q, full_fit.logGBF, full_fit.residuals, full_fit.format())


def fit_sample(sample_i, fit_info, params_map):
    dependent_data = gv.gvar(fit_info.samples[sample_i,:], fit_info.cov)
    sample_fit = lsqfit.nonlinear_fit(data=(fit_info.independent_data, dependent_data),
                                      fcn=fit_info.fit_function,
                                      p0=fit_info.init_guesses,
                                      prior=fit_info.priors[sample_i],
                                      svdcut=fit_info.svdcut,
                                      fitter=fit_info.fitter,
                                      noise=(False,False),
                                      debug=DEBUG)

    shape = (len(fit_info.samples), len(params_map))
    fit_data = np.ndarray(shape=shape, dtype=np.float64, buffer=fit_info.fit_data.buf)
    for param_name, param_value in sample_fit.pmean.items():
        fit_data[sample_i,params_map[param_name]] = param_value

    shape = (len(fit_info.samples),)
    fit_chi2 = np.ndarray(shape=shape, dtype=np.float64, buffer=fit_info.fit_chi2.buf)
    fit_chi2[sample_i] = sample_fit.chi2

    fit_Q = np.ndarray(shape=shape, dtype=np.float64, buffer=fit_info.fit_Q.buf)
    fit_Q[sample_i] = sample_fit.Q


def fit_name(fit_func_repr):
    def wrapper(fit_func):
        fit_func.name = fit_func_repr
        return fit_func
    return wrapper

def fit_num_params(num_params):
    def wrapper(fit_func):
        fit_func.num_params = num_params
        return fit_func
    return wrapper

def fit_params(params):
    def wrapper(fit_func):
        fit_func.params = params
        return fit_func
    return wrapper

class Fitter:

    def __init__(self, data_type, fit_function, fitter='scipy_least_squares'):
        self.data_type = data_type
        self.fit_function = fit_function
        self.fitter = fitter

    def do_fit(self, priors, log_params=[], correlated=True, prior_fit=True, resamplings=True, svdcut=1e-12):

        if self.num_data <= 0:
            print(f"ndat=0; fit failed")
            return False

        orig_param_names = list()
        params_map = dict()
        init_guesses = dict()
        actual_priors = dict()
        if prior_fit:
            for samp_i in range(self.data_type.num_samples+1):
                actual_priors[samp_i] = dict()
                for param_i, (param_name, prior) in enumerate(priors.items()):
                    if samp_i == 0:
                        orig_param_names.append(param_name)

                    if isinstance(prior[0], data_handler.Data):
                        prior = gv.gvar(prior[0](samp_i), prior[1]*prior[0].error())
                    else:
                        prior = gv.gvar(prior[0], prior[1])

                    if param_name in log_params:
                        param_name = f'log({param_name})'
                        prior = gv.log(prior)

                    actual_priors[samp_i][param_name] = prior
                    if samp_i == 0:
                        params_map[param_name] = param_i
                        init_guesses[param_name] = prior.mean

                    elif params_map[param_name] != param_i:
                        print(f"Param map not right")
                        return False

            dof = 0

        else:
            actual_priors = [None]*(self.data_type.num_samples+1)
            for param_i, (param_name, guess) in enumerate(priors.items()):
                orig_param_names.append(param_name)
                if param_name in log_params:
                    param_name = f'log({param_name})'
                    guess = np.log(guess)

                params_map[param_name] = param_i
                init_guesses[param_name] = guess

            dof = -len(init_guesses)

        fit_info = FitInfo
        fit_info.fitter = self.fitter
        fit_info.fit_function = self.fit_function
        fit_info.svdcut = svdcut

        fit_info.independent_data = self.data_type.get_independent_data()
        dof += self.data_type.num_data

        if dof < 0:
            print(f"invalid dof={dof}, ndat={len(fit_info.independent_data)}; fit failed")
            return False

        fit_info.samples = self.data_type.get_samples()
        fit_info.cov = self.data_type.get_covariance_lsqfit(correlated)
        self._cov = fit_info.cov

        fit_info.init_guesses = init_guesses
        fit_info.priors = actual_priors

        data_shape = (self.data_type.num_samples+1, len(params_map))
        fit_info.fit_data = get_shared_data(data_shape)
        fit_info.fit_chi2 = get_shared_data((self.data_type.num_samples+1,))
        fit_info.fit_Q = get_shared_data((self.data_type.num_samples+1,))

        '''
        if tune_priors:
          dependent_data = gv.gvar(fit_info.samples[0,:], fit_info.cov)
          def fitargs(z):
            dp = z
            current_priors = dict()
            for param_name, param_prior in priors.items():
              current_priors[param_name] = gv.gvar(param_prior.mean, dp*param_prior.mean)

            return dict(prior=gv.gvar(current_priors), fcn=self.fit_function, data=(fit_info.independent_data, dependent_data))

          fit, z = lsqfit.empbayes_fit(.1, fitargs)
          priors = fit.prior
        '''

        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            fit_result = fit_full(fit_info, params_map)
            fit_params = fit_result[0]
            self._chi2 = fit_result[1]
            self._dof = fit_result[2]
            self._Q = fit_result[3]
            self._logGBF = fit_result[4]
            self._residuals = fit_result[5]
            self._output = fit_result[6]
            if fit_info.init_guesses is not None:
                for param, fit_val in fit_params.items():
                    fit_info.init_guesses[param] = fit_val.mean


            if resamplings:
                with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                    pool.starmap(fit_sample, zip(range(1, self.data_type.num_samples+1), itertools.repeat(fit_info), itertools.repeat(params_map)))

                fit_data = np.copy(np.ndarray(shape=data_shape, dtype=np.float64, buffer=fit_info.fit_data.buf))
                self._params = dict()
                for param_name in orig_param_names:
                    if f'log({param_name})' in params_map:
                        param_i = params_map[f'log({param_name})']
                        self._params[param_name] = np.exp(data_handler.Data(fit_data[:,param_i]))

                    else:
                        param_i = params_map[param_name]
                        self._params[param_name] = data_handler.Data(fit_data[:,param_i])

                fit_chi2 = np.copy(np.ndarray(shape=(self.data_type.num_samples+1,), dtype=np.float64, buffer=fit_info.fit_chi2.buf))
                fit_Q = np.copy(np.ndarray(shape=(self.data_type.num_samples+1,), dtype=np.float64, buffer=fit_info.fit_Q.buf))
                self._resampling_chi2 = fit_chi2
                self._resampling_Q = fit_Q

            else:
                self._params = dict()
                for param_name in orig_param_names:
                    if f'log({param_name})' in params_map:
                        self._params[param_name] = gv.exp(fit_params[f'log({param_name})'])

                    else:
                        self._params[param_name] = fit_params[param_name]

        '''
        chol_cov = scipy.linalg.cholesky(fit_info.cov, lower=True)
        residues = np.empty(self.num_data, dtype=fit_info.samples.dtype)
        mean_params = self.params_sample(0)
        for d_i, independent_data_point in enumerate(self.data_type.get_organized_independent_data()):
          residues[d_i] = self.fit_function(independent_data_point, mean_params) - fit_info.samples[0,d_i]

        self._chi2_data = scipy.linalg.norm(scipy.linalg.solve_triangular(chol_cov, residues, lower=True, check_finite=False))**2

        if resamplings:
          self._resampling_chi2_data = np.empty(self.data_type.num_samples+1, dtype=np.float64)
          self._resampling_chi2_data[0] = self._chi2_data
          for sample_i in range(1, self.data_type.num_samples+1):
            sample_params = self.params_sample(sample_i)
            for d_i, independent_data_point in enumerate(self.data_type.get_organized_independent_data()):
              residues[d_i] = self.fit_function(independent_data_point, sample_params) - fit_info.samples[sample_i,d_i]

            self._resampling_chi2_data[sample_i] = scipy.linalg.norm(scipy.linalg.solve_triangular(chol_cov, residues, lower=True, check_finite=False))**2
        '''

        release_shared_data(fit_info.fit_data.name)
        release_shared_data(fit_info.fit_chi2.name)
        release_shared_data(fit_info.fit_Q.name)

        return True


    def tune_priors(self):
        ...

    @property
    def cov(self):
        if hasattr(self, '_cov'):
            return self._cov
        return None

    @property
    def num_params(self):
        return self.fit_function.num_params

    @property
    def chi2(self):
        if hasattr(self, '_chi2'):
            return self._chi2
        return None

    @property
    def chi2_data(self):
        if hasattr(self, '_chi2_data'):
            return self._chi2_data
        return None

    @property
    def chi2_dof_data(self):
        if hasattr(self, '_chi2_data'):
            try:
                return self._chi2_data / self.true_dof
            except ZeroDivisionError:
                return np.Inf
        return None

    @property
    def dof(self):
        if hasattr(self, '_dof'):
            return self._dof
        return None

    @property
    def num_data(self):
        return self.data_type.num_data

    @property
    def true_dof(self):
        return self.num_data - self.num_params


    @property
    def chi2_dof(self):
        if hasattr(self, '_chi2') and hasattr(self, '_dof'):
            return self._chi2 / self._dof
        return None

    @property
    def resampling_chi2(self):
        if hasattr(self, '_resampling_chi2'):
            return self._resampling_chi2
        return None

    @property
    def resampling_chi2_dof(self):
        if hasattr(self, '_resampling_chi2') and hasattr(self, '_dof'):
            return self._resampling_chi2 / self._dof
        return None

    @property
    def resampling_chi2_data(self):
        if hasattr(self, '_resampling_chi2_data'):
            return self._resampling_chi2_data
        return None

    @property
    def resampling_chi2_dof_data(self):
        if hasattr(self, '_resampling_chi2'):
            return self._resampling_chi2_data / self.true_dof
        return None

    @property
    def Q(self):
        if hasattr(self, '_Q'):
            return self._Q
        return None

    @property
    def resampling_Q(self):
        if hasattr(self, '_resampling_Q'):
            return self._resampling_Q
        return None

    @property
    def Q_data(self):
        if hasattr(self, '_chi2_data'):
            Q = scipy.special.gammaincc(self.true_dof/2., self.chi2_data/2.)
            return Q
        return None

    @property
    def resampling_Q_data(self):
        if hasattr(self, '_resampling_chi2_data'):
            _resampling_Q_data = np.empty(self.data_type.num_samples+1, dtype=np.float64)
            for si, chi2_data in enumerate(self.resampling_chi2_data):
                _resampling_Q_data[si] = scipy.special.gammaincc(self.true_dof/2., chi2_data/2.)
            return _resampling_Q_data
        return None

    @property
    def AIC(self):
        if hasattr(self, '_Q'):
            return 2.*self.num_params - 2.*np.log(self.Q)
        return None

    def adjusted_AIC(self, num_priors):
        dof = self.num_data - self.num_params + num_priors
        Q = scipy.special.gammaincc(dof/2., self.chi2/2.)
        return 2.*(self.num_params - num_priors) - 2.*np.log(Q)

    def resampled_AIC(self, sample_i):
        if hasattr(self, '_resampling_chi2'):
            Q = scipy.special.gammaincc(self.dof/2., self._resampling_chi2[sample_i]/2.)
            return 2.*self.num_params - 2.*np.log(Q)
        return None

    def resampled_adjusted_AIC(self, sample_i, num_priors):
        if hasattr(self, '_resampling_chi2'):
            dof = self.num_data - self.num_params + num_priors
            Q = scipy.special.gammaincc(dof/2., self._resampling_chi2[sample_i]/2.)
            return 2.*(self.num_params - num_priors) - 2.*np.log(Q)
        return None

    @property
    def logGBF(self):
        if hasattr(self, '_logGBF'):
            return self._logGBF
        return None

    @property
    def residuals(self):
        if hasattr(self, '_residuals'):
            return self._residuals
        return None

    @property
    def w(self):
        if hasattr(self, '_logGBF') and self.logGBF is not None:
            return np.exp(self._logGBF)
        return None

    @property
    def params(self):
        if hasattr(self, '_params'):
            return self._params
        return None

    def param(self, param_name):
        if hasattr(self, '_params') and param_name in self._params:
            return self._params[param_name]
        return None

    def params_sample(self, sample_i):
        params_dict = dict()
        for param_name, param_data in self.params.items():
            params_dict[param_name] = param_data.samples[sample_i]

        return params_dict

    @property
    def params_samples(self):
        params_dict = dict()
        for param_name, param_data in self.params.items():
            params_dict[param_name] = param_data.samples

        return params_dict

    @property
    def name(self):
        return self.fit_function.name

    @property
    def independent_data(self):
        return self.data_type.get_independent_data()

    def output(self):
        '''
        _output = f"Fit results: {self.name}\n"
        _output += f"  chi2/dof [dof] = {round(self.chi2_dof, 2)} [{self.dof}]      Q = {round(self.Q, 3)}"
        if self.logGBF is None:
          _output += "\n\n"
        else:
          _output += f"      logGBF = {round(self.logGBF, 2)}      w = {round(self.w, 2)}\n\n"

        _output += "Parameters:\n"
        for param_name, param in self._params.items():
          _output += f"  {param_name} = {param!s}\n"

        return _output
        '''
        return self._output
