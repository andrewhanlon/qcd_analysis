import os

import enum
import h5py
import numpy as np
import scipy
import mpmath
import warnings
import copy
import tqdm

import gvar as gv

import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt

from qcd_analysis import data_handler
from qcd_analysis import lsqfit_fitter as fitter


class General1d(data_handler.DataType):

    def __init__(self, data):
        self._data = data

    def get_samples(self):
        _samples = list()
        for z_data in self._data.values():
            _samples.append(z_data.samples)

        return np.array(_samples).T

    @property
    def num_samples(self):
        return list(self._data.values())[0].num_samples

    @property
    def num_data(self):
        return len(self.get_independent_data())

    def get_independent_data(self):
        return np.array(list(self._data.keys()))

    def get_organized_independent_data(self):
        return self.get_independent_data()

    def remove_data(self, data_to_remove):
        new_data = dict()
        for x, data in sorted(self._data.items()):
            if x in data_to_remove:
                continue

            new_data[x] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def __call__(self, z):
        return self._data[z]

    def plot(self, xlabel, ylabel, plot_file):
        fig, ax = plt.subplots()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        x_vals = list()
        y_vals = list()
        y_errs = list()

        for x, y in self._data.keys():
            y_vals.append(y.mean)
            y_errs.append(y.sdev)
            x_vals.append(x)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()


    def plot_with_fit(self, xlabel, ylabel, plot_file, fitter, fitted_data, print_params={}):
        fig, ax = plt.subplots()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        x_vals = list()
        y_vals = list()
        y_errs = list()

        for x, y in self._data.keys():
            y_vals.append(y.mean)
            y_errs.append(y.sdev)
            x_vals.append(x)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        x_vals = list(np.linspace(min(fitted_data.get_independent_data()), max(fitted_data.get_independent_data()), 500))
        mean_vals = list()
        upper_vals = list()
        lower_vals = list()
        for x in x_vals:
            y = fitter.fit_function(x., fitter.params)

            mean = y.mean
            mean_vals.append(mean)
            err = y.sdev
            upper_vals.append(mean+err)
            lower_vals.append(mean-err)

        ax.plot(x_vals, mean_vals, '-', lw=1., color='blue')
        ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color='blue')

        num_params = len(print_params)

        plt.text(0.05, 0.07 + num_params*.075, rf"$\chi^2_{{\rm dof}} = {round(fitter.chi2_dof,2)}$", transform=ax.transAxes)
        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
            plt.text(0.05, 0.07 + (num_params - height)*.075, rf"${param_text} = {fitter.param(param_name)!s}$", transform=ax.transAxes)

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()
