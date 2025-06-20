import os
import copy

import enum
import h5py
import numpy as np

import gvar as gv
import matplotlib.pyplot as plt

from qcd_analysis import data_handler
from qcd_analysis import lsqfit_fitter as fitter


class C3ptData(data_handler.DataType):

    def __init__(self, data):
        self._data = data

    '''
    def get_summed_data(self, removals):
      summed_data = dict()
      for tsep, tsep_data in self._data.items():
        summed_data[tsep] = 0.
        texc = removals.get(tsep, 0)
        for tins in range(1+texc, tsep-texc):
          if tins in tsep_data:
            summed_data[tsep] += tsep_data[tins]

      return summed_data
    '''
    def get_summed_data(self, texc):
        summed_data = dict()
        for tsep, tsep_data in self._data.items():
            summed_data[tsep] = 0.
            for tins in range(1+texc, tsep-texc):
                if tins in tsep_data:
                    summed_data[tsep] += tsep_data[tins]

        return summed_data

    def get_samples(self):
        _samples = list()
        for tsep, tsep_data in self._data.items():
            for tins, tins_data in tsep_data.items():
                _samples.append(tins_data.samples)

        return np.array(_samples).T

    @property
    def num_samples(self):
        return list(list(self._data.values())[0].values())[0].num_samples

    def get_independent_data(self):
        tsep_list = list()
        tins_list = list()
        for tsep, tsep_data in self._data.items():
            tsep_list.extend(len(tsep_data)*[tsep])
            tins_list.extend(tsep_data.keys())

        return (np.array(tsep_list), np.array(tins_list))

    def get_organized_independent_data(self):
        independent_data_list = list()
        for tsep, tsep_data in self._data.items():
            for tins in tsep_data.keys():
                independent_data_list.append((tsep, tins))

        return np.array(independent_data_list)


    '''
    @property
    def num_data(self):
      return len(self.get_independent_data()[0])
    '''

    def remove_tseps(self, tseps):
        new_data = dict()
        for tsep, tsep_data in sorted(self._data.items()):
            if tsep not in tseps:
                new_data[tsep] = tsep_data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def remove_tins(self, texc):
        new_data = dict()
        for tsep, tsep_data in sorted(self._data.items()):
            new_tsep_data = dict()
            for tins in range(1+texc, tsep-texc):
                new_tsep_data[tins] = tsep_data[tins]

            new_data[tsep] = new_tsep_data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def remove_data(self, times_dict, tmax_rel_error=0.):
        new_data = dict()
        for tsep, tsep_data in sorted(self._data.items()):
            new_tsep_data = dict()
            for tins in range(tsep//2, -1, -1):
                if tins not in tsep_data:
                    continue

                if tsep in times_dict and tins in times_dict[tsep]:
                    continue

                corr_val = tsep_data[tins].mean
                corr_err = tsep_data[tins].sdev
                if (abs(corr_val) / corr_err) < tmax_rel_error:
                    break

                new_tsep_data[tins] = tsep_data[tins]

            # stupid I know, but I wanted the tmax_rel_error thing to be symmetric
            for tins in range(tsep//2 + 1, tsep + 1):
                if tins not in tsep_data:
                    continue

                if tsep in times_dict and tins in times_dict[tsep]:
                    continue

                corr_val = tsep_data[tins].mean
                corr_err = tsep_data[tins].sdev
                if (abs(corr_val) / corr_err) < tmax_rel_error:
                    break

                new_tsep_data[tins] = tsep_data[tins]

            if new_tsep_data:
                new_data[tsep] = new_tsep_data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def plot_data(self, plot_dir):
        fig, ax = plt.subplots()
        plt.xlabel(r"$t_{\rm ins} - t_{\rm sep}/2$")
        if self.complex_arg == data_handler.ComplexArg.REAL:
            plt.ylabel(r"$\Re R$")
        else:
            plt.ylabel(r"$\Im R$")

        num_tseps = len(self._data)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_tseps))
        disp = .33 * (num_tseps - 1) / 10
        displacements = np.linspace(-disp, disp, num_tseps)

        for color_i, (tsep, tsep_data) in enumerate(self._data.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()

            for tins, data in tsep_data.items():
                x_vals.append(tins - tsep/2 + displacements[color_i])
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$t_{{\rm sep}} = {tsep}$")

        plt.legend()
        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{self.name}.pdf")
        plt.savefig(plot_file)
        plt.close()

    def print_data(self):
        for tsep, tsep_data in self._data.items():
            print(f"tsep = {tsep}")
            for tins, data in tsep_data.items():
                x_val = tins - tsep/2
                y_val = data.mean
                y_err = data.sdev
                y_gvar = gv.gvar(y_val, y_err)
                print(f"  {x_val}: {y_gvar}", end='')
            print()

    def plot_fit(self, plot_dir, name, fit, fit_data, print_params={}, plot_params=list()):
        fig, ax = plt.subplots()
        plt.xlabel(r"$t_{\rm ins} - t_{\rm sep}/2$")
        if self.complex_arg == data_handler.ComplexArg.REAL:
            plt.ylabel(r"$\Re R$")
        else:
            plt.ylabel(r"$\Im R$")

        num_tseps = len(self._data) + len(plot_params)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_tseps))
        disp = .33 * (num_tseps - 1) / 10
        displacements = np.linspace(-disp, disp, num_tseps)

        # plot data
        for color_i, (tsep, tsep_data) in enumerate(self._data.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()

            for tins, data in tsep_data.items():
                x_vals.append(tins - tsep/2 + displacements[color_i])
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$t_{{\rm sep}} = {tsep}$")

        # now the fit lines
        for color_i, tsep in enumerate(self._data.keys()):
            if tsep not in fit_data._data:
                continue

            tsep_data = fit_data._data[tsep]
            tinss = np.linspace(min(tsep_data.keys()) - .1, max(tsep_data.keys()) + .1, 500)
            tseps = len(tinss)*[float(tsep)]
            mean_vals = list()
            upper_vals = list()
            lower_vals = list()
            for x in zip(tseps, tinss):
                y = fit.fit_function(x, fit.params)
                mean = y.mean
                mean_vals.append(mean)
                err = y.sdev
                upper_vals.append(mean+err)
                lower_vals.append(mean-err)

            x = tinss - tsep/2

            ax.plot(x, mean_vals, '-', lw=1., color=colors[color_i])
            ax.fill_between(x, lower_vals, upper_vals, alpha=0.2, lw=0, color=colors[color_i])

        '''
        if fit.chi2_dof_data != fit.chi2_dof:
          plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof_data,2)}, {round(fit.chi2_dof,2)}$", transform=ax.transAxes)
        else:
          plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof_data,2)}$", transform=ax.transAxes)
        '''
        plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof,2)}$", transform=ax.transAxes)

        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
            plt.text(0.05, 0.93 - height*.075, rf"${param_text} = {fit.param(param_name)!s}$", transform=ax.transAxes)

        if plot_params:
            for color_i, plot_param in enumerate(plot_params, len(self._data)):
                param_val = plot_param.mean
                param_err = plot_param.sdev

                plt.axhline(y=param_val, color=colors[color_i], linestyle='-', lw=1.)
                ax.axhspan(param_val - param_err, param_val + param_err, color=colors[color_i], alpha=0.2, lw=0)

        plt.legend(loc='upper right')
        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{self.name}_{name}.pdf")
        plt.savefig(plot_file)
        plt.close()

    def plot_tsep_dependence_of_fit(self, plot_dir, name, fit, fit_data, tsep_range, tsep_fits=dict(), print_params={}, plot_params=list()):
        fig, ax = plt.subplots()
        plt.xlabel(r"$t_{\rm sep}$")
        if self.complex_arg == data_handler.ComplexArg.REAL:
            plt.ylabel(r"$\Re R$")
        else:
            plt.ylabel(r"$\Im R$")

        plt.xlim(tsep_range[0], tsep_range[1])

        num_colors = 1 + len(tsep_fits) + len(plot_params)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
        disp = .33 * (num_colors - 1) / 10
        displacements = np.linspace(-disp, disp, num_colors)

        # get tsep dependence band
        tseps = np.linspace(tsep_range[0], tsep_range[1], 500)
        x_vals = list()
        mean_vals = list()
        upper_vals = list()
        lower_vals = list()
        for ts in tseps:
            ti = ts/2.
            x_vals.append(ts)
            y = fit.fit_function((ts,ti), fit.params)
            mean = y.mean
            mean_vals.append(mean)
            err = y.sdev
            upper_vals.append(mean+err)
            lower_vals.append(mean-err)

        ax.plot(x_vals, mean_vals, '-', lw=1., color=colors[0])
        ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color=colors[0])

        # plot tsep fits
        for color_i, (tsep, tsep_fit) in enumerate(tsep_fits.items(), 1):
            plt.errorbar(tsep, tsep_fit.mean, yerr=tsep_fit.sdev, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i])

        '''
        if fit.chi2_dof_data != fit.chi2_dof:
          plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof_data,2)}, {round(fit.chi2_dof,2)}$", transform=ax.transAxes)
        else:
          plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof_data,2)}$", transform=ax.transAxes)

        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
          plt.text(0.05, 0.93 - height*.075, rf"${param_text} = {fit.param(param_name)!s}$", transform=ax.transAxes)
        '''

        if plot_params:
            for color_i, plot_param in enumerate(plot_params, 1 + len(tsep_fits)):
                param_val = plot_param.mean
                param_err = plot_param.sdev

                plt.axhline(y=param_val, color=colors[color_i], linestyle='-', lw=1.)
                ax.axhspan(param_val - param_err, param_val + param_err, color=colors[color_i], alpha=0.2, lw=0)

        #plt.legend(loc='upper right')
        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{self.name}_{name}.pdf")
        plt.savefig(plot_file)
        plt.close()


    def __call__(self, tsep, tins):
        return self._data[tsep][tins]

###################################################################################################
#     HELPER FUNCTIONS
###################################################################################################

def get_summed_fit_function(fit, nexc):
    Nsum = 1000
    def func(tsep):
        N = (tsep - (1 + 2*nexc))/Nsum
        S = 0.
        for tins in np.linspace(1+nexc, tsep - (1+nexc), Nsum):
            S += fit.fit_function((tsep, tins), fit.params)

        S *= N

        return S

    return func


###################################################################################################
#     MODELS
###################################################################################################

# x is [tseps, tinss]

'''
@fitter.fit_name(f"one_state_symm")
def one_state_symm(x, p):
  return p['M_0,0'] * gv.exp(-p['E0']*x[0])

@fitter.fit_name(f"two_state_symm")
def two_state_symm(x, p):
  return p['M_0,0'] * gv.exp(-p['E0']*x[0]) + \
         p['M_0,1'] * (gv.exp(-p['E0']*(x[0] - x[1])) * gv.exp(-p['E1']*x[1])) + p['M_0,1'] * (gv.exp(-p['E1']*(x[0] - x[1])) * gv.exp(-p['E0']*x[1])) + \
         p['M_1,1'] * gv.exp(-p['E1']*x[0])

@fitter.fit_name(f"three_state_symm")
def three_state_symm(x, p):
  return p['M_0,0'] * gv.exp(-p['E0']*x[0]) + \
         p['M_0,1'] * (gv.exp(-p['E0']*(x[0] - x[1])) * gv.exp(-p['E1']*x[1])) + p['M_0,1'] * (gv.exp(-p['E1']*(x[0] - x[1])) * gv.exp(-p['E0']*x[1])) + \
         p['M_1,1'] * gv.exp(-p['E1']*x[0]) + \
         p['M_0,2'] * (gv.exp(-p['E0']*(x[0] - x[1])) * gv.exp(-p['E2']*x[1])) + p['M_0,2'] * (gv.exp(-p['E2']*(x[0] - x[1])) * gv.exp(-p['E0']*x[1])) + \
         p['M_1,2'] * (gv.exp(-p['E1']*(x[0] - x[1])) * gv.exp(-p['E2']*x[1])) + p['M_1,2'] * (gv.exp(-p['E2']*(x[0] - x[1])) * gv.exp(-p['E1']*x[1])) + \
         p['M_2,2'] * gv.exp(-p['E2']*x[0])
'''

'''
@fitter.fit_name(f"two_state_symmetric")
@fitter.fit_num_params(5)
def two_state_symmetric(x, p):
  return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[0]/2.) * (gv.exp(-p['dE1,0']*(x[0]/2. - x[1])) + gv.exp(p['dE1,0']*(x[0]/2. - x[1]))) + p['M_1,1']*gv.exp(-p['dE1,0']*x[0])) / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]))

@fitter.fit_name(f"two_state_symmetric_no_denom")
@fitter.fit_num_params(4)
def two_state_symmetric_no_denom(x, p):
  return p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[0]/2.) * (gv.exp(-p['dE1,0']*(x[0]/2. - x[1])) + gv.exp(p['dE1,0']*(x[0]/2. - x[1]))) + p['M_1,1']*gv.exp(-p['dE1,0']*x[0])

@fitter.fit_name(f"two_state_symmetric_no_denom_no_tsep")
@fitter.fit_num_params(3)
def two_state_symmetric_no_denom_no_tsep(x, p):
  return p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[0]/2.) * (gv.exp(-p['dE1,0']*(x[0]/2. - x[1])) + gv.exp(p['dE1,0']*(x[0]/2. - x[1])))

@fitter.fit_name(f"three_state_symmetric")
@fitter.fit_num_params(10)
def three_state_symmetric(x, p):
  return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[0]/2.) * (gv.exp(-p['dE1,0']*(x[0]/2. - x[1])) + gv.exp(p['dE1,0']*(x[0]/2. - x[1]))) + p['M_1,1']*gv.exp(-p['dE1,0']*x[0])
         + p['M_0,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0]/2.) * (gv.exp(-(p['dE2,1'] + p['dE1,0'])*(x[0]/2. - x[1])) + gv.exp(-(p['dE2,1'] + p['dE1,0'])*(x[0]/2. - x[1])))
         + p['M_1,2']*gv.exp(-p['dE1,0']*x[0])*gv.exp(-p['dE2,1']*x[0]/2.)*(gv.exp(-p['dE2,1']*(x[0]/2. - x[1])) + gv.exp(p['dE2,1']*(x[0]/2. - x[1])))
         + p['M_2,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0])
         ) / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]) + p['R2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0]))
'''

@fitter.fit_name(f"one_state")
@fitter.fit_num_params(1)
def one_state(x, p):
    return p['M_0,0']

@fitter.fit_name(f"two_state")
@fitter.fit_num_params(6)
def two_state(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_1,0']*gv.exp(-p['dE1,0']*(x[0] - x[1])) + p['M_1,1']*gv.exp(-p['dE1,0']*x[0])
           )  / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]))

@fitter.fit_name(f"two_state_nodenom")
@fitter.fit_num_params(5)
def two_state_nodenom(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_1,0']*gv.exp(-p['dE1,0']*(x[0] - x[1])) + p['M_1,1']*gv.exp(-p['dE1,0']*x[0]))

@fitter.fit_name(f"two_state_notsep_nodenom")
@fitter.fit_num_params(4)
def two_state_notsep_nodenom(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_1,0']*gv.exp(-p['dE1,0']*(x[0] - x[1])))

@fitter.fit_name(f"two_state_notsep")
@fitter.fit_num_params(5)
def two_state_notsep(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_1,0']*gv.exp(-p['dE1,0']*(x[0] - x[1]))
           )  / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]))

@fitter.fit_name(f"two_state_symmetric")
@fitter.fit_num_params(5)
def two_state_symmetric(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_0,1']*gv.exp(-p['dE1,0']*(x[0] - x[1])) + p['M_1,1']*gv.exp(-p['dE1,0']*x[0])
           ) / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]))

@fitter.fit_name(f"two_state_symmetric_nodenom")
@fitter.fit_num_params(4)
def two_state_symmetric_nodenom(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_0,1']*gv.exp(-p['dE1,0']*(x[0] - x[1])) + p['M_1,1']*gv.exp(-p['dE1,0']*x[0]))

@fitter.fit_name(f"two_state_symmetric_notsep_nodenom")
@fitter.fit_num_params(3)
def two_state_symmetric_notsep_nodenom(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_0,1']*gv.exp(-p['dE1,0']*(x[0] - x[1])))

@fitter.fit_name(f"two_state_symmetric_notsep")
@fitter.fit_num_params(4)
def two_state_symmetric_notsep(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_0,1']*gv.exp(-p['dE1,0']*(x[0] - x[1]))
           ) / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]))

@fitter.fit_name(f"three_state")
@fitter.fit_num_params(13)
def three_state(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_1,0']*gv.exp(-p['dE1,0']*(x[0] - x[1]))
                       + p['M_0,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[1]) + p['M_2,0']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*(x[0] - x[1]))
                       + p['M_1,2']*gv.exp(-p['dE2,1']*x[1] - p['dE1,0']*x[0]) + p['M_2,1']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0] + p['dE2,1']*x[1])
                       + p['M_1,1']*gv.exp(-p['dE1,0']*x[0]) + p['M_2,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0])
           ) / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]) + p['R2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0]))

@fitter.fit_name(f"three_state_symmetric")
@fitter.fit_num_params(10)
def three_state_symmetric(x, p):
    return (p['M_0,0'] + p['M_0,1']*gv.exp(-p['dE1,0']*x[1]) + p['M_0,1']*gv.exp(-p['dE1,0']*(x[0] - x[1]))
                       + p['M_0,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[1]) + p['M_0,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*(x[0] - x[1]))
                       + p['M_1,2']*gv.exp(-p['dE2,1']*x[1] - p['dE1,0']*x[0]) + p['M_1,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0] + p['dE2,1']*x[1])
                       + p['M_1,1']*gv.exp(-p['dE1,0']*x[0]) + p['M_2,2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0])
           ) / (1. + p['R1']*gv.exp(-p['dE1,0']*x[0]) + p['R2']*gv.exp(-(p['dE2,1'] + p['dE1,0'])*x[0]))

###################################################################################################
#     C3ptSummedData
###################################################################################################

class C3ptSummedData(data_handler.DataType):

    def __init__(self, data):
        self._data = data

    def get_samples(self):
        _samples = list()
        for tsep, tsep_data in self._data.items():
            _samples.append(tsep_data.samples)

        return np.array(_samples).T

    @property
    def num_samples(self):
        return list(self._data.values())[0].num_samples

    def get_independent_data(self):
        return np.array(list(self._data.keys()))

    def get_organized_independent_data(self):
        return self.get_independent_data()

    '''
    @property
    def num_data(self):
      return len(self.get_independent_data())
    '''

    def remove_data(self, tmin=0, tmax=-1, tmax_rel_error=0.):
        new_data = dict()
        for tsep, data in sorted(self._data.items()):
            if tsep < tmin:
                continue

            if np.sign(tmax) > 0 and tsep > tmax:
                break

            corr_val = data.mean
            corr_err = data.sdev
            if (abs(corr_val) / corr_err) < tmax_rel_error:
                break

            new_data[tsep] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def remove_tseps(self, tseps):
        new_data = dict()
        for tsep, data in sorted(self._data.items()):
            if tsep not in tseps:
                new_data[tsep] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def plot_data(self, plot_dir):
        fig, ax = plt.subplots()
        plt.xlabel(r"$t_{\rm sep}$")
        if self.complex_arg == data_handler.ComplexArg.REAL:
            plt.ylabel(r"$\Sigma \Re R$")
        else:
            plt.ylabel(r"$\Sigma \Im R$")

        x_vals = list()
        y_vals = list()
        y_errs = list()
        for tsep, tsep_data in self._data.items():
            x_vals.append(tsep)
            y_vals.append(tsep_data.mean)
            y_errs.append(tsep_data.sdev)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color='k')

        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{self.name}.pdf")
        plt.savefig(plot_file)
        plt.close()

    def plot_fit(self, plot_dir, name, fit, fit_data, print_params={}, comparison_fits={}):
        fig, ax = plt.subplots()
        plt.xlabel(r"$t_{\rm sep}$")
        if self.complex_arg == data_handler.ComplexArg.REAL:
            plt.ylabel(r"$\Sigma \Re R$")
        else:
            plt.ylabel(r"$\Sigma \Im R$")

        x_vals = list()
        y_vals = list()
        y_errs = list()
        for tsep, tsep_data in self._data.items():
            x_vals.append(tsep)
            y_vals.append(tsep_data.mean)
            y_errs.append(tsep_data.sdev)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color='k')

        # now the fit line
        mean_vals = list()
        upper_vals = list()
        lower_vals = list()
        x_vals = np.linspace(min(fit_data._data.keys()), max(fit_data._data.keys()), 500)
        for x in x_vals:
            y = fit.fit_function(x, fit.params)
            mean = y.mean
            mean_vals.append(mean)
            err = y.sdev
            upper_vals.append(mean+err)
            lower_vals.append(mean-err)

        ax.plot(x_vals, mean_vals, '-', lw=1., color='k')
        ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color='k')

        '''
        if fit.chi2_dof_data != fit.chi2_dof:
          plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof_data,2)}, {round(fit.chi2_dof,2)}$", transform=ax.transAxes)
        else:
          plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof_data,2)}$", transform=ax.transAxes)
        '''
        plt.text(0.05, 0.93, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof,2)}$", transform=ax.transAxes)

        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
            plt.text(0.05, 0.93 - height*.075, rf"${param_text} = {fit.param(param_name)!s}$", transform=ax.transAxes)

        # comparison_fits
        num_fits = len(comparison_fits)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_fits))
        for color_i, (fit_name, comp_func) in enumerate(comparison_fits.items()):
            mean_vals = list()
            upper_vals = list()
            lower_vals = list()
            for x in x_vals:
                y = comp_func(x)
                mean = y.mean
                mean_vals.append(mean)
                err = y.sdev
                upper_vals.append(mean+err)
                lower_vals.append(mean-err)

            ax.plot(x_vals, mean_vals, '-', lw=1., color=colors[color_i], label=fit_name)
            ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color=colors[color_i])

        if comparison_fits:
            plt.legend()

        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{self.name}_{name}.pdf")
        plt.savefig(plot_file)
        plt.close()



    def __call__(self, tsep):
        return self._data[tsep]


###################################################################################################
#     MODELS
###################################################################################################

# x is tsep

'''
@fitter.fit_name('one-state_summation')
@fitter.fit_num_params(2)
def one_state_summation(x, p):
  return p['C0'] + p['M_0,0']*x

@fitter.fit_name('two-state_summation')
@fitter.fit_num_params(4)
def two_state_summation(x, p):
  return p['C0'] + p['M_0,0']*x + p['C1']*gv.exp(-p['dE1,0']*x)
'''


def get_summation_fit_function(num_states, n_exc):
    if num_states == 1:
        @fitter.fit_name(f"1-state_summation-exc{n_exc}")
        @fitter.fit_num_params(2)
        def fit_function(x, p):
            return (x - 2*n_exc)*p['M_0,0'] + p['C0']

    elif num_states == 2:
        @fitter.fit_name(f"2-state_summation-exc{n_exc}")
        @fitter.fit_num_params(4)
        def fit_function(x, p):
            return (x - 2*n_exc)*p['M_0,0'] + p['C0'] + p['C1']*gv.exp(-p['dE1,0']*x)

    else:
        raise NotImplementedError

    return fit_function
