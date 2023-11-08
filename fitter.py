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

def fit_sample(samp_i, fit_data_sh, input_data, fit_function, init_guesses):
    comb_fit_function = fit_function.get_combined_fit_function(input_data)
    data_shape = (input_data.num_samples+1, len(init_guesses))
    residuals_func = input_data.get_weighted_residuals_func(samp_i, comb_fit_function)
    fit_result = scipy.optimize.least_squares(residuals_func, init_guesses)
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
        elif isinstance(input_data, data_handler.DataType):
            self.input_data = data_handler.MultiDataType([input_data])
        elif type(input_data) is data_handler.MultiDataType:
            self.input_data = input_data
        else:
            raise TypeError("Invalid input_data passed to fitter.Fitter")

        if type(fit_function) is list:
            self.fit_function = data_handler.MultiFitFunction(fit_function)
        elif isinstance(fit_function, data_handler.FitFunction):
            self.fit_function = data_handler.MultiFitFunction([fit_function])
        elif type(fit_function) is data_handler.MultiFitFunction:
            self.fit_function = fit_function
        else:
            raise TypeError("Invalid fit_function passed to fitter.Fitter")


        if self.fit_function.num_functions != self.input_data.num_data_types:
            raise ValueError("Mismatch in size of input_data and fit_function passed to fitter.Fitter")


    def do_fit(self, init_guesses):
        """
        Args:
          init_guesses - list[dict{str: float}] or dict{str: float}:
              If a list, each element corresponds to one of the fit functions. The number of
              fit_functions (and therefore data types) must match the size of the list.
              If init_guesses is a dict, then it will be treated like a one element list

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

        # construct init_guesses
        if type(init_guesses) is dict:
            init_guesses = [init_guesses]

        if len(init_guesses) != self.input_data.num_data_types:
            print("Mismatch in size of init_guesses and input_data")
            return False

        init_guesses_flat = list()
        for init_guess, fit_func in zip(init_guesses, self.fit_function.fit_functions):
            for param in fit_func.params:
                init_guesses_flat.append(init_guess[param])

        # construct result data structures
        data_shape = (self.input_data.num_samples+1, len(init_guesses_flat))
        fit_data_sh = get_shared_data(data_shape)

        # do fit
        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            fit_results_mean = fit_sample(0, fit_data_sh, self.input_data, self.fit_function, init_guesses_flat)

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
        self._params = list()
        param_i = 0
        for fit_func in self.fit_function.fit_functions:
            fit_params = dict()
            for param in fit_func.params:
                fit_params[param] = data_handler.Data(fit_data[:,param_i])
                param_i += 1
            self._params.append(fit_params)

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
        for data_type, fit_func, params in zip(self.input_data.data_types, self.fit_function.fit_functions, self.params):
            _output += f"    Data: {data_type.data_name}, Fit function: {fit_func.fit_name}\n"
            for param_name, param in params.items():
                _output += f"        {param_name} = {param!s}\n"

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
    def params(self):
        if hasattr(self, '_params'):
            return self._params
        return None

    @property
    def Q(self):
        if hasattr(self, '_chi2') and hasattr(self, '_dof'):
            Q = scipy.special.gammaincc(self.dof/2., self.chi2/2.)
            return Q
        return None
