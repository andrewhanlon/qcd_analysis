import sys
import h5py
import numpy as np
import scipy.optimize
import scipy.special
import itertools
import multiprocessing
import multiprocessing.shared_memory
import threadpoolctl

import lsqfit
import gvar as gv

from qcd_analysis.data_handling import data_handler

NUM_PROCESSES = multiprocessing.cpu_count() // 2
def set_num_processes(num_processes):
    global NUM_PROCESSES
    NUM_PROCESSES = num_processes

PARALLEL = True
def set_to_serial():
    global PARALLEL
    PARALLEL = False

def set_to_parallel():
    global PARALLEL
    PARALLEL = True

def get_shared_data(shape):
    d_size = int(np.dtype(np.float64).itemsize * np.prod(shape))
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=d_size)
    return shm

def release_shared_data(name):
    shm = multiprocessing.shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()

def fit_sample(samp_i, fit_data_sh, input_data, fit_function, init_guesses, **kwargs):
    data_shape = (input_data.num_samples+1, len(init_guesses))
    residuals_func = input_data.get_weighted_residuals_func(samp_i, fit_function)
    fit_result = scipy.optimize.least_squares(residuals_func, init_guesses, **kwargs)
    fit_data = np.ndarray(shape=data_shape, dtype=np.float64, buffer=fit_data_sh.buf)
    fit_data[samp_i,:] = fit_result.x
    if samp_i == 0:
        return fit_result

def fit_sample_lsqfit(samp_i, fit_data_sh, input_data, fit_function, init_guesses, **kwargs):
    dependent_data = gv.gvar(input_data.samples[0,:], input_data.cov)
    full_fit = lsqfit.nonlinear_fit(data=(input_data.independent_variables_values, dependent_data),
                                    fcn=fit_function,
                                    p0=init_guesses,
                                    prior=fit_function.priors,
                                    svdcut=fit_info.svdcut,
                                    fitter=fit_info.fitter,
                                    noise=(False,False),
                                    debug=DEBUG)

    data_shape = (input_data.num_samples+1, len(init_guesses))
    residuals_func = input_data.get_weighted_residuals_func(samp_i, fit_function)
    fit_result = scipy.optimize.least_squares(residuals_func, init_guesses, **kwargs)
    fit_data = np.ndarray(shape=data_shape, dtype=np.float64, buffer=fit_data_sh.buf)
    fit_data[samp_i,:] = fit_result.x
    if samp_i == 0:
        return fit_result

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

class Fitter:

    def __init__(self, input_data, fit_function):
        """
        Args:
          input_data - list[data_handler.DataType] or data_handler.DataType:
              The data to be fit
          fit_function - list[data_handler.FitFunction] or data_handler.FitFunction:
              The functions to fit the data to. The number of input functions must match
              the number of input data types.
        """

        if type(input_data) is list:
            self.input_data = data_handler.MultiDataType(input_data)
        elif isinstance(input_data, data_handler.MultiDataType):
            self.input_data = input_data
        elif isinstance(input_data, data_handler.DataType):
            self.input_data = data_handler.MultiDataType([input_data])
        else:
            raise TypeError("Invalid input_data passed to fitter.Fitter")

        if type(fit_function) is list:
            self.fit_function = data_handler.MultiFitFunction(fit_function)
        elif isinstance(fit_function, data_handler.MultiFitFunction):
            self.fit_function = fit_function
        elif isinstance(fit_function, data_handler.FitFunction):
            self.fit_function = data_handler.MultiFitFunction([fit_function])
        else:
            raise TypeError("Invalid fit_function passed to fitter.Fitter")


        if self.fit_function.num_functions != self.input_data.num_data_types:
            raise ValueError("Mismatch in size of input_data and fit_function passed to fitter.Fitter")


    def do_fit(self, uncorrelated=False, method='trf', **kwargs):
        """
        Args:
            uncorrelated (bool): self explanatory, default is false
            method (str): Possible methods are
                          - 'trf' (default)
                          - 'dogbox'
                          - 'lm'
                          - 'lsqft_scipy'
                          - 'lsqft_gsl'
            kwargs: to be passed to the fitter
        Returns:
            bool: True if fit is successful, otherwise False
        """

        #try:
        if self.input_data.num_data <= 0:
            print(f"ndat <= 0; fit failed")
            return False

        self._dof = self.input_data.num_data - self.fit_function.num_params + self.fit_function.num_priors

        if self._dof <= 0:
            print(f"invalid dof={self._dof}, ndat={self.input_data.num_data}, nparams={self.fit_function.num_params}, npriors={self.fit_function.num_priors}; fit failed")
            return False

        self.input_data.set_covariance(uncorrelated)

        init_guesses_flat = list()
        for param in self.fit_function.params:
            init_guesses_flat.append(self.init_guesses[param])

        # construct result data structures
        data_shape = (self.input_data.num_samples+1, len(init_guesses_flat))
        fit_data_sh = get_shared_data(data_shape)

        # do fit
        if PARALLEL:
            with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
                fit_results_mean = fit_sample(0, fit_data_sh, self.input_data, self.fit_function, init_guesses_flat, **kwargs)

                self._chi2 = 2.*fit_results_mean.cost
                init_guesses_flat = fit_results_mean.x

                with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                    # Thanks be to https://stackoverflow.com/a/53173433
                    samples_i_list = list(range(1, self.input_data.num_samples+1))
                    args_iter = zip(samples_i_list,
                                    itertools.repeat(fit_data_sh),
                                    itertools.repeat(self.input_data),
                                    itertools.repeat(self.fit_function),
                                    itertools.repeat(init_guesses_flat))
                    kwargs_iter = itertools.repeat(kwargs)
                    args_for_starmap = zip(itertools.repeat(fit_sample), args_iter, kwargs_iter)
                    pool.starmap(apply_args_and_kwargs, args_for_starmap)


        else:
            fit_results_mean = fit_sample(0, fit_data_sh, self.input_data, self.fit_function, init_guesses_flat, **kwargs)

            self._chi2 = 2.*fit_results_mean.cost
            init_guesses_flat = fit_results_mean.x

            for sample_i in range(1, self.input_data.num_samples+1):
                fit_sample(sample_i, fit_data_sh, self.input_data, self.fit_function, init_guesses_flat, **kwargs)

        # get results
        fit_data = np.copy(np.ndarray(shape=data_shape, dtype=np.float64, buffer=fit_data_sh.buf))
        self._params = dict()
        for param_i, param in enumerate(self.fit_function.params):
            self._params[param] = data_handler.Data(fit_data[:,param_i])

        release_shared_data(fit_data_sh.name)

        '''
        except Exception as e:
            print(f"Fit failed: {e}")
            return False
        '''

        return True

    def output(self, spacing=0):
        _output = spacing*" " + f"Fit results:\n"
        _output += spacing*" " + f"    chi2/dof [dof] = {round(self.chi2_dof, 2)} [{self.dof}]      Q = {round(self.Q, 3)}     AIC = {round(self.AIC, 3)}\n\n"
        '''
        if self.logGBF is None:
            _output += "\n\n"
        else:
            _output += f"      logGBF = {round(self.logGBF, 2)}      w = {round(self.w, 2)}\n\n"
        '''

        _output += spacing*" " + "Parameters:\n"

        for data_type, fit_func in zip(self.input_data.data_types, self.fit_function.fit_functions):
            _output += spacing*" " + f"    Data: {data_type.data_name}, Fit function: {fit_func.fit_name}\n"
            for param in fit_func.params:
                param_value = self.params[param]
                init_guess = self.init_guesses[param]
                _output += spacing*" " + f"      {param : >10} {str(param_value):>15}    [ {init_guess:>12.6f} "
                if param in self.fit_function.priors:
                    _output += f", {str(self.fit_function.priors[param]):>15} "
                _output += "]\n"
            _output += "\n"

        return _output

    def write_to_hdf5(self, filename, fit_name, params_to_write, overwrite=True, additional_params={}, additional_attrs={}):
        fh = h5py.File(filename, 'a')

        if fit_name in fh and not overwrite:
            print(f"Group '{fit_name}' already exists in file '{filename}'")
            sys.exit()

        fit_group = fh.create_group(fit_name)
        for param in params_to_write:
            fit_group.create_dataset(param, data=self.params[param].samples)

        for param, param_data in additional_params.items():
            if param not in params_to_write:
                fit_group.create_dataset(param, data=param_data.samples)

        fit_group.attrs['chi2'] = self.chi2
        fit_group.attrs['dof'] = self.dof
        fit_group.attrs['num_params'] = self.num_params
        fit_group.attrs['num_priors'] = self.num_priors

        params = list()
        priored_params = list()
        init_guesses = list()
        param_results = list()
        priors = list()

        for fit_func in self.fit_function.fit_functions:
            for param in fit_func.params:
                params.append(param)
                param_results.append(str(self.params[param]))
                init_guesses.append(self.init_guesses[param])
                if param in self.fit_function.priors:
                    priored_params.append(param)
                    priors.append(str(self.fit_function.priors[param]))

        fit_group.attrs['params'] = params
        fit_group.attrs['param_results'] = param_results
        fit_group.attrs['init_guesses'] = init_guesses
        if priored_params:
            fit_group.attrs['priored_params'] = priored_params
            fit_group.attrs['priors'] = priors

        for k, v in additional_attrs.items():
            fit_group.attrs[k] = v

        fh.close()


    @property
    def num_params(self):
        return self.fit_function.num_params

    @property
    def num_priors(self):
        return self.fit_function.num_priors

    @property
    def Q(self):
        if hasattr(self, '_Q'):
            return self._Q
        elif hasattr(self, '_chi2') and hasattr(self, '_dof'):
            self._Q = scipy.special.gammaincc(self.dof/2., self.chi2/2.)
            return self._Q
        else:
            raise AttributeError("Cannot access Q: fit not done yet!")

    @property
    def chi2(self):
        if hasattr(self, '_chi2'):
            return self._chi2
        else:
            raise AttributeError("Cannot access chi2: fit not done yet!")

    @property
    def dof(self):
        if hasattr(self, '_dof'):
            return self._dof
        else:
            raise AttributeError("Cannot access dof: fit not done yet!")

    @property
    def chi2_dof(self):
        if hasattr(self, '_chi2') and hasattr(self, '_dof'):
            return self._chi2 / self._dof
        else:
            raise AttributeError("Cannot access chi2_dof: fit not done yet!")

    @property
    def AIC(self):
        if hasattr(self, '_Q'):
            return 2.*self.num_params - 2.*np.log(self.Q)
        else:
            raise AttributeError("Cannot access AIC: fit not done yet!")

    @property
    def logGBF(self):
        if hasattr(self, '_logGBF'):
            return self._logGBF
        else:
            raise AttributeError("Cannot access logGBF: fit not done yet!")

    @property
    def params(self):
        if hasattr(self, '_params'):
            return self._params
        else:
            raise AttributeError("Cannot access params: fit not done yet!")

    @property
    def init_guesses(self):
        return self.fit_function.init_guesses
