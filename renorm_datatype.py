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
#plt.rc('font', family='serif', serif=['Times'], size=55)
#plt.rc('text', usetex=True)
plt.style.use('/home/ahanlon/code/lattice/qcd_analysis/plots.mplstyle')

import data_handler
import lsqfit_fitter as fitter

hc = 6.62607015e-34 / 1.602176634e-19 / (2*np.pi) * 299792458 * 1e6 # exact: new SI

class RenormalizationData(data_handler.DataType):

    def __init__(self, data, Ns):
        self._data = data
        self.Ns = Ns

    @property
    def data_name(self):
        return "renorm"

    def get_samples(self):
        _samples = list()
        for renorm_data in self._data.values():
            _samples.append(renorm_data.samples)

        return np.array(_samples).T

    @property
    def num_samples(self):
        return list(self._data.values())[0].num_samples

    @property
    def num_data(self):
        return len(self.get_independent_data()[0])

    def get_psqs(self):
        pfactor = 2.*np.pi / self.Ns
        psq_vals = list()
        for puRs in self._data.keys():
            puRsq = 0.
            for p_i, puR in enumerate(puRs):
                if p_i == 3:
                    puRsq += (np.sin(pfactor*(puR + .5)))**2
                else:
                    puRsq += (np.sin(pfactor*puR))**2

            psq_vals.append(puRsq)

        return sorted(psq_vals)

    def get_independent_data(self):
        px_list = list()
        py_list = list()
        pz_list = list()
        pt_list = list()
        for pu in self._data.keys():
            px_list.append(pu[0])
            py_list.append(pu[1])
            pz_list.append(pu[2])
            pt_list.append(pu[3])

        return (np.array(px_list), np.array(py_list), np.array(pz_list), np.array(pt_list))

    def get_organized_independent_data(self):
        return np.array(self._data.keys())

    def remove_data(self, data_to_remove):
        new_data = dict()
        for x, data in sorted(self._data.items()):
            if x in data_to_remove:
                continue

            new_data[x] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def convert_to(self, func, a):
        pfactor = 2.*np.pi / self.Ns
        new_data = dict()
        for puRs, data in sorted(self._data.items()):

            puRsq = 0.
            for p_i, puR in enumerate(puRs):
                if p_i == 3:
                    puRsq += (np.sin(pfactor*(puR + .5)))**2
                else:
                    puRsq += (np.sin(pfactor*puR))**2

            puRsq *= (hc/a)**2

            mu = np.sqrt(puRsq)

            new_data[puRs] = func(mu)*data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def run_to(self, mu, func, a):
        pfactor = 2.*np.pi / self.Ns
        new_data = dict()
        for puRs, data in sorted(self._data.items()):

            puRsq = 0.
            for p_i, puR in enumerate(puRs):
                if p_i == 3:
                    puRsq += (np.sin(pfactor*(puR + .5)))**2
                else:
                    puRsq += (np.sin(pfactor*puR))**2

            puRsq *= (hc/a)**2

            mup = np.sqrt(puRsq)

            new_data[puRs] = (func(mu)/func(mup))*data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def __call__(self, z):
        return self._data[z]

    def plot(self, plot_file, ylabel, a=None):
        fig, ax = plt.subplots()
        if a is not None:
            plt.xlabel(r'$p_R^2 [{\rm GeV}^2]$')
        else:
            plt.xlabel(r'$(a p_R)^2$')
        plt.ylabel(ylabel)

        x_vals = list()
        y_vals = list()
        y_errs = list()

        pfactor = 2.*np.pi / self.Ns
        for puRs, y in self._data.items():
            y_vals.append(y.mean)
            y_errs.append(y.sdev)

            puRsq = 0.
            for p_i, puR in enumerate(puRs):
                if p_i == 3:
                    puRsq += (np.sin(pfactor*(puR + .5)))**2
                else:
                    puRsq += (np.sin(pfactor*puR))**2

            if a is not None:
                x_vals.append(puRsq*(hc/a)**2)
            else:
                x_vals.append(puRsq)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()


    def plot_with_fit(self, plot_file, ylabel, fitter, fitted_data, fit_function, print_params={}, a=None):
        fig, ax = plt.subplots()
        if a is not None:
            plt.xlabel(r'$p_R^2 [{\rm GeV}^2]$')
        else:
            plt.xlabel(r'$(a p_R)^2$')
        plt.ylabel(ylabel)

        x_vals = list()
        y_vals = list()
        y_errs = list()

        pfactor = 2.*np.pi / self.Ns
        for puRs, y in self._data.items():
            y_vals.append(y.mean)
            y_errs.append(y.sdev)

            puRsq = 0.
            for p_i, puR in enumerate(puRs):
                if p_i == 3:
                    puRsq += (np.sin(pfactor*(puR + .5)))**2
                else:
                    puRsq += (np.sin(pfactor*puR))**2

            if a is not None:
                x_vals.append(puRsq*(hc/a)**2)
            else:
                x_vals.append(puRsq)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        x_vals = list(np.linspace(min(fitted_data.get_psqs()), max(fitted_data.get_psqs()), 500))
        mean_vals = list()
        upper_vals = list()
        lower_vals = list()
        final_x_vals = list()
        for x in x_vals:
            y = fit_function(x, fitter.params)

            mean = y.mean
            mean_vals.append(mean)
            err = y.sdev
            upper_vals.append(mean+err)
            lower_vals.append(mean-err)

            if a is not None:
                final_x_vals.append(x*(hc/a)**2)
            else:
                final_x_vals.append(x)


        ax.plot(final_x_vals, mean_vals, '-', lw=1., color='blue')
        ax.fill_between(final_x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color='blue')

        num_params = len(print_params)

        plt.text(0.95, 0.07 + num_params*.075, rf"$\chi^2_{{\rm dof}} = {round(fitter.chi2_dof,2)}$", transform=ax.transAxes, ha='right', va='center')
        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
            plt.text(0.95, 0.07 + (num_params - height)*.075, rf"${param_text} = {fitter.param(param_name)!s}$", transform=ax.transAxes, ha='right', va='center')

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def has_data(self, puR):
        return puR in self._data

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            if self.Ns != other.Ns:
                raise ValueError("Ns does not match")

            new_data = dict()
            for puR, data in self._data.items():
                if other.has_data(puR):
                    new_data[puR] = data / other(puR)

            new_obj = self.__class__(new_data, self.Ns)

            return new_obj

        else:
            new_data = dict()
            for z, data in self._data.items():
                new_data[z] = data / other

            new_obj = self.__class__(new_data, self.Ns)

            return new_obj


def get_renormalization_model(k, Ns):
    @fitter.fit_name(f"renormalization_model")
    @fitter.fit_num_params(6)
    @fitter.fit_params(['Z', 'B', 'C', 'C4', 'C6', 'C8'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        D4 = 0.
        D6 = 0.
        D8 = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
                D4 += (np.sin(pfactor*(pRu + .5)))**4
                D6 += (np.sin(pfactor*(pRu + .5)))**6
                D8 += (np.sin(pfactor*(pRu + .5)))**8
            else:
                apRsq += (np.sin(pfactor*pRu))**2
                D4 += (np.sin(pfactor*pRu))**4
                D6 += (np.sin(pfactor*pRu))**6
                D8 += (np.sin(pfactor*pRu))**8

        D4 /= apRsq**2
        D6 /= apRsq**3
        D8 /= apRsq**4

        return p['Z'] + p['B']/apRsq + p['C']*apRsq**(k/2) * (1. + p['C4']*D4 + p['C6']*D6 + p['C8']*D8)

    return fit_function

def get_min_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_min_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'B'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['B']/apRsq

    return fit_function

def get_min4_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_min_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'B'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['B']/(apRsq**2)

    return fit_function

def get_min_lin_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_min_lin_model")
    @fitter.fit_num_params(3)
    @fitter.fit_params(['Z', 'B', 'D'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['B']/apRsq + p['D']*np.sqrt(apRsq)

    return fit_function

def get_min_quad_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_min_quad_model")
    @fitter.fit_num_params(3)
    @fitter.fit_params(['Z', 'B', 'D2'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['B']/apRsq + p['D2']*apRsq

    return fit_function

def get_min_lin_quad_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_min_lin_quad_model")
    @fitter.fit_num_params(4)
    @fitter.fit_params(['Z', 'B', 'D', 'D2'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['B']/apRsq + p['D']*np.sqrt(apRsq) + p['D2']*apRsq

    return fit_function

def get_lin_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_lin_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'D'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['D']*np.sqrt(apRsq)

    return fit_function

def get_quad_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_quad_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'D2'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['D2']*apRsq

    return fit_function

def get_lin_quad_renormalization_model(Ns):
    @fitter.fit_name(f"renormalization_lin_quad_model")
    @fitter.fit_num_params(3)
    @fitter.fit_params(['Z', 'D', 'D2'])
    def fit_function(x, p):
        pfactor = 2.*np.pi / Ns

        apRsq = 0.
        for p_i, pRu in enumerate(x):
            if p_i == 3:
                apRsq += (np.sin(pfactor*(pRu + .5)))**2
            else:
                apRsq += (np.sin(pfactor*pRu))**2

        return p['Z'] + p['D']*np.sqrt(apRsq) + p['D2']*apRsq

    return fit_function

# psq models

def get_renormalization_psq_model():
    @fitter.fit_name(f"renormalization_psq_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'B'])
    def fit_function(x, p):
        return p['Z'] + p['B']/x

    return fit_function

def get_renormalization_psq4_model():
    @fitter.fit_name(f"renormalization_psq4_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'B'])
    def fit_function(x, p):
        return p['Z'] + p['B']/(x**2)

    return fit_function


def get_renormalization_psq_lin_model():
    @fitter.fit_name(f"renormalization_psq_lin_model")
    @fitter.fit_num_params(3)
    @fitter.fit_params(['Z', 'B', 'D'])
    def fit_function(x, p):
        return p['Z'] + p['B']/x + p['D']*np.sqrt(x)

    return fit_function

def get_renormalization_psq_quad_model():
    @fitter.fit_name(f"renormalization_psq_quad_model")
    @fitter.fit_num_params(3)
    @fitter.fit_params(['Z', 'B', 'D2'])
    def fit_function(x, p):
        return p['Z'] + p['B']/x + p['D2']*x

    return fit_function

def get_renormalization_psq_lin_quad_model():
    @fitter.fit_name(f"renormalization_psq_lin_quad_model")
    @fitter.fit_num_params(4)
    @fitter.fit_params(['Z', 'B', 'D', 'D2'])
    def fit_function(x, p):
        return p['Z'] + p['B']/x + p['D']*np.sqrt(x) + p['D2']*x

    return fit_function

def get_renormalization_lin_model():
    @fitter.fit_name(f"renormalization_lin_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'D'])
    def fit_function(x, p):
        return p['Z'] + p['D']*np.sqrt(x)

    return fit_function

def get_renormalization_quad_model():
    @fitter.fit_name(f"renormalization_quad_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['Z', 'D2'])
    def fit_function(x, p):
        return p['Z'] + p['D2']*x

    return fit_function

def get_renormalization_lin_quad_model():
    @fitter.fit_name(f"renormalization_lin_quad_model")
    @fitter.fit_num_params(3)
    @fitter.fit_params(['Z', 'D', 'D2'])
    def fit_function(x, p):
        return p['Z'] + p['D']*np.sqrt(x) + p['D2']*x

    return fit_function
