import sys

import numpy as np

from qcd_analysis.data_handling import data_handler


class ContinuumDispersion(data_handler.FitFunction):

    def __init__(self, Ns):
        self.Ns = Ns

    @property
    def fit_name(self):
        return f"continuum_dispersion"

    @property
    def params(self):
        return ['m0', 'c']

    def function(self, x, p):
        return p['m0']**2 + p['c']*(2.*np.pi/self.Ns)**2*x


    def get_init_guesses(self, m0):
        init_guesses_dict = {
                'm0': m0,
                'c': 1.,
        }

        return init_guesses_dict
                
