import numpy as np
import scipy.optimize
import itertools
import multiprocessing
import multiprocessing.shared_memory
import threadpoolctl

import data_handler

NUM_PROCESSES = 2 if multiprocessing.cpu_count() <= 16 else multiprocessing.cpu_count() // 8

def get_shared_data(shape):
    d_size = np.dtype(np.float64).itemsize * np.prod(shape)
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


    def do_fit(self, **kwargs):
        """
        Returns:
          bool: True if fit is successful, otherwise False
        """

        if self.input_data.num_data <= 0:
            print(f"ndat <= 0; fit failed")
            return False

        self._dof = self.input_data.num_data - self.fit_function.num_params + self.fit_function.num_priors

        if self._dof < 0:
            print(f"invalid dof={self._dof}, ndat={self.input_data.num_data}; fit failed")
            return False

        self.input_data.set_covariance()

        init_guesses_flat = list()
        for param in self.fit_function.params:
            init_guesses_flat.append(self.init_guesses[param])

        # construct result data structures
        data_shape = (self.input_data.num_samples+1, len(init_guesses_flat))
        fit_data_sh = get_shared_data(data_shape)

        # do fit
        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            fit_results_mean = fit_sample(0, fit_data_sh, self.input_data, self.fit_function, init_guesses_flat, **kwargs)

            self._chi2 = 2.*fit_results_mean.cost
            init_guesses_flat = fit_results_mean.x

            with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                pool.starmap(fit_sample, zip(list(range(1, self.input_data.num_samples+1)),
                                             itertools.repeat(fit_data_sh),
                                             itertools.repeat(self.input_data),
                                             itertools.repeat(self.fit_function),
                                             itertools.repeat(init_guesses_flat)))


        # get results
        fit_data = np.copy(np.ndarray(shape=data_shape, dtype=np.float64, buffer=fit_data_sh.buf))
        self._params = dict()
        for param_i, param in enumerate(self.fit_function.params):
            self._params[param] = data_handler.Data(fit_data[:,param_i])

        release_shared_data(fit_data_sh.name)

        return True

    def output(self):
        _output = f"Fit results:\n"
        _output += f"    chi2/dof [dof] = {round(self.chi2_dof, 2)} [{self.dof}]      Q = {round(self.Q, 3)}"
        if self.logGBF is None:
            _output += "\n\n"
        else:
            _output += f"      logGBF = {round(self.logGBF, 2)}      w = {round(self.w, 2)}\n\n"

        _output += "Parameters:\n"

        for data_type, fit_func in zip(self.input_data.data_types, self.fit_function.fit_functions):
            _output += f"    Data: {data_type.data_name}, Fit function: {fit_func.fit_name}\n"
            for param in fit_func.params:
                param_value = self.params[param]
                init_guess = self.init_guesses[param]
                _output += f"      {param : >10} {str(param_value):>15}    [ {init_guess:>12.6f} "
                if param in self.fit_function.priors:
                    _output += f", {str(self.fit_function.priors[param]):>15} "
                _output += "]\n"
            _output += "\n"

        return _output

    @property
    def chi2(self):
        if hasattr(self, '_chi2'):
            return self._chi2
        return None

    @property
    def dof(self):
        if hasattr(self, '_dof'):
            return self._dof
        return None

    @property
    def chi2_dof(self):
        if hasattr(self, '_chi2') and hasattr(self, '_dof'):
            return self._chi2 / self._dof
        return None

    @property
    def logGBF(self):
        if hasattr(self, '_logGBF'):
            return self._logGBF
        return None

    @property
    def Q(self):
        if hasattr(self, '_chi2') and hasattr(self, '_dof'):
            Q = scipy.special.gammaincc(self.dof/2., self.chi2/2.)
            return Q
        return None

    @property
    def params(self):
        if hasattr(self, '_params'):
            return self._params
        return None

    @property
    def init_guesses(self):
        return self.fit_function.init_guesses
