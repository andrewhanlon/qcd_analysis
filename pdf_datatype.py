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


class PDFMatrixElement(data_handler.DataType):

    def __init__(self, data):
        self.normalization = 1.
        self._data = data

    def normalize(self, normalization=None):
        old_normalization = self.normalization

        if normalization is not None:
            self.normalization = normalization
        else:
            normalize_z = min(abs(self.get_independent_data()))
            self.normalization = old_normalization / np.abs(self._data[normalize_z].mean)

        for z in self.get_independent_data():
            self._data[z] = (self.normalization/old_normalization) * self._data[z]

    def symmetrize(self, neg_sign):
        symm_data = dict()

        max_z = max([abs(z) for z in self.get_independent_data()])

        if 0 in self._data:
            symm_data[0] = self._data[0]

        for z in range(1, max_z+1):
            if z in self._data and -z in self._data:
                symm_data[z] = 0.5*(self._data[z] + np.sign(neg_sign)*self._data[-z])

            elif z in self._data:
                symm_data[z] = self._data[z]

            elif -z in self._data:
                symm_data[z] = np.sign(neg_sign)*self._data[-z]

        self._data = symm_data

    def make_gvar(self):
        gvar_data = dict()
        for z, data in self._data.items():
            gvar_data[z] = data.gvar()

        self._data = gvar_data

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

    def remove_data(self, zs_to_remove):
        new_data = dict()
        for z, data in sorted(self._data.items()):
            if z in zs_to_remove:
                continue

            new_data[z] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def __call__(self, z, normalized=True):
        if normalized:
            return self._data[z]
        else:
            return self.normalization * self._data[z]

    def has_z(self, z):
        return z in self._data


    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for z, data in self._data.items():
                if other.has_z(z):
                    new_data[z] = data / other(z)

            new_obj = self.__class__(new_data)
            new_obj.normalization = self.normalization / other.normalization

            return new_obj

        elif isinstance(other, dict):
            new_data = dict()
            for z, data in self._data.items():
                if z in other:
                    new_data[z] = data / other[z]

            new_obj = self.__class__(new_data)
            new_obj.normalization = self.normalization

            return new_obj

        else:
            new_data = dict()
            for z, data in self._data.items():
                new_data[z] = data / other

            new_obj = self.__class__(new_data)
            new_obj.normalization = self.normalization

            return new_obj

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for z, data in self._data.items():
                if other.has_z(z):
                    new_data[z] = data * other(z)

            new_obj = self.__class__(new_data)
            new_obj.normalization = self.normalization * other.normalization

            return new_obj

        elif isinstance(other, dict):
            new_data = dict()
            for z, data in self._data.items():
                if z in other:
                    new_data[z] = data * other[z]

            new_obj = self.__class__(new_data)
            new_obj.normalization = self.normalization

            return new_obj

        else:
            new_data = dict()
            for z, data in self._data.items():
                new_data[z] = data * other

            new_obj = self.__class__(new_data)
            new_obj.normalization = self.normalization

            return new_obj

    def __rmul__(self, other):
        return self.__mul__(other)



###################################################################################################
#     HELPER FUNCTIONS
###################################################################################################

def plot_pdf_matrix_element(plot_file, to_plot_map, a, ylabel, *text_adds):
    fig, ax = plt.subplots()
    plt.xlabel(r"$x$ [fm]")
    plt.ylabel(ylabel)

    num_data = len(to_plot_map)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_data))
    disp = .33 * (num_data - 1) / 10
    displacements = np.linspace(-disp, disp, num_data)

    min_z = None
    max_z = None
    for color_i, (plot_label, plot_data) in enumerate(to_plot_map.items()):
        x_vals = list()
        y_vals = list()
        y_errs = list()

        if min_z is None:
            min_z = min(plot_data.get_independent_data())
        elif min_z > min(plot_data.get_independent_data()):
            min_z = min(plot_data.get_independent_data())

        if max_z is None:
            max_z = max(plot_data.get_independent_data())
        elif max_z < max(plot_data.get_independent_data()):
            max_z = max(plot_data.get_independent_data())

        for z in plot_data.get_independent_data():
            x_vals.append((z + displacements[color_i])*a)
            y_vals.append(plot_data(z).mean)
            y_errs.append(plot_data(z).sdev)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', ms=2, capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=plot_label)

    for height, text_add in enumerate(text_adds):
        text_height = 0.93 - height*.075
        plt.text(0.65, text_height, text_add, transform=ax.transAxes)

    if num_data > 1:
        plt.legend()

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)
    plt.close()


def plot_pdf_matrix_element_with_systematics(plot_file, to_plot_map, a, ylabel, *text_adds):
    fig, ax = plt.subplots()
    plt.xlabel(r"$x$ [fm]")
    plt.ylabel(ylabel)

    num_data = len(to_plot_map)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_data))
    disp = .33 * (num_data - 1) / 10
    displacements = np.linspace(-disp, disp, num_data)

    for color_i, (plot_label, plot_data) in enumerate(to_plot_map.items()):
        stat_plot_data = plot_data[0]
        tot_plot_data = plot_data[1]

        x_vals = list()
        y_vals = list()
        y_errs = list()

        for z in stat_plot_data.get_independent_data():
            x_vals.append((z + displacements[color_i])*a)
            y_vals.append(stat_plot_data(z).mean)
            y_errs.append(stat_plot_data(z).sdev)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=plot_label)

        x_vals = list()
        y_vals = list()
        y_errs = list()

        for z in tot_plot_data.get_independent_data():
            x_vals.append((z + displacements[color_i])*a)
            y_vals.append(tot_plot_data(z).mean)
            y_errs.append(tot_plot_data(z).sdev)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=plot_label)

    for height, text_add in enumerate(text_adds):
        text_height = 0.93 - height*.075
        plt.text(0.65, text_height, text_add, transform=ax.transAxes)

    if num_data > 1:
        plt.legend()

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)
    plt.close()


###############################################################################
# Ioffe Time Distribution
###############################################################################


class IoffeTimeDistribution(data_handler.DataType):

    def __init__(self, data, ref_mom, Ns, complex_arg, denom_data=None, include_px0=False):
        self.moms = sorted(list(data[0].keys()))
        self.ref_mom = ref_mom
        self.Ns = Ns
        self.complex_arg = complex_arg
        self.include_px0 = include_px0

        self._form_ratio(data, denom_data)

    def _form_ratio(self, data, denom_data):
        if denom_data is None:
            denom_data = data

        self._data = dict()
        if data[0].keys() != data[1].keys():
            print("Numerator Real and Imaginary data don't have same Px")
            sys.exit()

        if self.ref_mom not in denom_data[0] or self.ref_mom not in denom_data[1]:
            print(f"Denominator doesn't have ref mom = {self.ref_mom}")

        data0_z0 = denom_data[0][self.ref_mom][0] + 1.j*denom_data[1][self.ref_mom][0]

        for mom in self.moms:
            if (not self.include_px0 and mom <= self.ref_mom) or mom < self.ref_mom:
                continue

            if data[0][mom].keys() != data[1][mom].keys():
                print("invalid")
                sys.exit()

            data_z0 = data[0][mom][0] + 1.j*data[1][mom][0]

            self._data[mom] = dict()
            for z in data[0][mom].keys():
                data_z = data[0][mom][z] + 1.j*data[1][mom][z]
                data0_z = denom_data[0][self.ref_mom][z] + 1.j*denom_data[1][self.ref_mom][z]

                self._data[mom][z] = (data_z / data0_z) * (data0_z0 / data_z0)
                if self.complex_arg == data_handler.ComplexArg.REAL:
                    self._data[mom][z] = self._data[mom][z].real
                else:
                    self._data[mom][z] = self._data[mom][z].imag

    def get_samples(self):
        _samples = list()
        for mom, mom_data in self._data.items():
            for z, z_data in mom_data.items():
                _samples.append(z_data.samples)

        return np.array(_samples).T

    def symmetrize(self, sym, keep_neg=False):
        new_data = dict()
        for mom, data_z in self._data.items():
            new_data[mom] = dict()
            zs = set([abs(z) for z in data_z.keys()])
            for z in zs:
                if z == 0 or -z not in data_z:
                    new_data[mom][z] = data_z[z]
                    if keep_neg:
                        new_data[mom][-z] = sym*data_z[z]

                else:
                    new_data[mom][z] = 0.5*(data_z[z] + sym*data_z[-z])
                    if keep_neg:
                        new_data[mom][-z] = 0.5*(sym*data_z[z] + data_z[-z])

        self._data = new_data

    @property
    def num_samples(self):
        return list(list(self._data.values())[0].values())[0].num_samples

    @property
    def num_data(self):
        return len(self.get_organized_independent_data())

    def get_independent_data(self):
        ioffe_list = list()
        zsq_list = list()
        for mom, mom_data in self._data.items():
            for z in mom_data.keys():
                ioffe = (2.*np.pi / self.Ns) * z * mom
                ioffe_list.append(ioffe)
                zsq_list.append(z**2)

        return (np.array(ioffe_list), np.array(zsq_list))

    def get_organized_independent_data(self):
        data_list = list()
        for mom, mom_data in self._data.items():
            for z in mom_data.keys():
                ioffe = (2.*np.pi / self.Ns) * z * mom
                data_list.append((ioffe, z**2))

        return np.array(data_list)

    def remove_data(self, mom_range, z_range):
        new_data = dict()
        for mom in mom_range:
            if mom not in self._data:
                continue

            new_data[mom] = dict()
            for z in z_range:
                if z not in self._data[mom]:
                    continue

                new_data[mom][z] = self._data[mom][z]

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def plot(self, plot_file, zmax, a, ope_funcs, ylabel):
        line_styles = ['--', '-', '-.', ':', '-', '--']
        fig, ax = plt.subplots()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(ylabel)

        num_tseps = len(self._data)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_tseps))

        for color_i, (mom, data_z) in enumerate(self._data.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()

            min_z = None
            max_z = None
            for z, data in data_z.items():
                if abs(z) > zmax:
                    continue
                if min_z is None or min_z > z:
                    min_z = z
                if max_z is None or max_z < z:
                    max_z = z

                ioffe = (2.*np.pi / self.Ns) * z * mom
                x_vals.append(ioffe)
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$n_x = {mom}$")

            z_vals = np.linspace(min_z, max_z, 1000)
            for line_style_i, (plot_label, ope_func) in enumerate(ope_funcs.items()):
                x_vals = list()
                y_vals = list()

                for z in z_vals:
                    ioffe = (2.*np.pi / self.Ns) * z * mom
                    x_vals.append(ioffe)
                    y_vals.append(ope_func(ioffe, z**2))

                ax.plot(x_vals, y_vals, line_styles[line_style_i], lw=1., color=colors[color_i])

        '''
        if ope_funcs:
          ax2 = ax.twinx()
          for line_style_i, plot_label in enumerate(ope_funcs.keys()):
            ax2.plot(np.NaN, np.NaN, ls=line_styles[line_style_i], color='k', label=plot_label)
          ax2.get_yaxis().set_visible(False)

          ax2.legend()
        '''

        plt.legend()
        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def plot_fit(self, plot_file, fit, fitted_data, a, ylabel, print_params, *text_adds):
        fig, ax = plt.subplots()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(ylabel)

        num_tseps = len(self._data)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_tseps))

        # plot data
        for color_i, (mom, data_z) in enumerate(self._data.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()

            for z, data in data_z.items():
                ioffe = (2.*np.pi / self.Ns) * z * mom
                x_vals.append(ioffe)
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$n_x = {mom}$")


        # plot fit
        for color_i, mom in enumerate(self._data.keys()):
            if mom not in fitted_data._data:
                continue

            data_z = fitted_data._data[mom]

            z_negs = [z for z in data_z.keys() if z < 0]
            z_poss = [z for z in data_z.keys() if z >= 0]

            if z_negs and z_poss:
                zs = np.append(np.linspace(min(z_negs), max(z_negs), 500), np.linspace(min(z_poss), max(z_poss), 500))
            elif z_poss:
                zs = np.linspace(min(z_poss), max(z_poss), 500)
            else:
                zs = np.linspace(min(z_negs), max(z_negs), 500)

            ioffes = list()
            mean_vals = list()
            upper_vals = list()
            lower_vals = list()
            for z in zs:
                ioffe = (2.*np.pi / self.Ns) * z * mom
                ioffes.append(ioffe)
                x_vals.append(ioffe)
                y = fit.fit_function((ioffe, z**2), fit.params)
                mean = y.mean
                mean_vals.append(mean)
                err = y.sdev
                upper_vals.append(mean+err)
                lower_vals.append(mean-err)

            ax.plot(ioffes, mean_vals, '-', lw=1., color=colors[color_i])
            ax.fill_between(ioffes, lower_vals, upper_vals, alpha=0.2, lw=0, color=colors[color_i])


        num_params = len(print_params)

        starting_height = .25
        plt.text(0.05, starting_height + num_params*.075, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof,2)}$", transform=ax.transAxes)
        '''
        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
          plt.text(0.05, starting_height + (num_params - height)*.075, rf"${param_text} = {fit.param(param_name)!s}$", transform=ax.transAxes)
        '''
        for height, (param_text, param_data) in enumerate(print_params.items(), 1):
            plt.text(0.05, starting_height + (num_params - height)*.075, rf"${param_text} = {param_data!s}$", transform=ax.transAxes)

        for height, text_add in enumerate(text_adds):
            text_height = 0.93 - height*.075
            plt.text(0.65, text_height, text_add, transform=ax.transAxes)

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def plot_vs_z(self, plot_file, a, ope_funcs, ylabel):
        #line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (3, 10, 1, 10))]
        line_styles = ['--', '-', '-.', ':', '-', '--']
        fig, ax = plt.subplots()
        plt.xlabel(r"$z$ [fm]")
        plt.ylabel(ylabel)

        num_tseps = len(self._data)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_tseps))

        for color_i, (mom, data_z) in enumerate(self._data.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()

            min_z = None
            max_z = None
            for z, data in data_z.items():
                if min_z is None or min_z > z:
                    min_z = z
                if max_z is None or max_z < z:
                    max_z = z

                x_vals.append(z*a)
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$n_x = {mom}$")

            z_vals = np.linspace(min_z, max_z, 1000)
            for line_style_i, (plot_label, ope_func) in enumerate(ope_funcs.items()):
                x_vals = list()
                y_vals = list()

                for z in z_vals:
                    ioffe = (2.*np.pi / self.Ns) * z * mom
                    x_vals.append(z*a)
                    y_vals.append(ope_func(ioffe, z**2))

                ax.plot(x_vals, y_vals, line_styles[line_style_i], lw=1., color=colors[color_i])

        '''
        if ope_funcs:
          ax2 = ax.twinx()
          for line_style_i, plot_label in enumerate(ope_funcs.keys()):
            ax2.plot(np.NaN, np.NaN, ls=line_styles[line_style_i], color='k', label=plot_label)
          ax2.get_yaxis().set_visible(False)

          ax2.legend()
        '''

        plt.legend(loc='lower left')
        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()


    def plot_fit_vs_z(self, plot_file, fit, fitted_data, a, ylabel, print_params, *text_adds):
        fig, ax = plt.subplots()
        plt.xlabel(r"$z$ [fm]")
        plt.ylabel(ylabel)

        num_tseps = len(self._data)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_tseps))

        # plot data
        for color_i, (mom, data_z) in enumerate(self._data.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()

            for z, data in data_z.items():
                x_vals.append(z*a)
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$n_x = {mom}$")


        # plot fit
        for color_i, mom in enumerate(self._data.keys()):
            if mom not in fitted_data._data:
                continue

            data_z = fitted_data._data[mom]

            z_negs = [z for z in data_z.keys() if z < 0]
            z_poss = [z for z in data_z.keys() if z >= 0]

            if z_negs and z_poss:
                zs = np.append(np.linspace(min(z_negs), max(z_negs), 500), np.linspace(min(z_poss), max(z_poss), 500))
            elif z_poss:
                zs = np.linspace(min(z_poss), max(z_poss), 500)
            else:
                zs = np.linspace(min(z_negs), max(z_negs), 500)

            x_vals = list()
            mean_vals = list()
            upper_vals = list()
            lower_vals = list()
            for z in zs:
                ioffe = (2.*np.pi / self.Ns) * z * mom
                x_vals.append(z*a)
                y = fit.fit_function((ioffe, z**2), fit.params)
                mean = y.mean
                mean_vals.append(mean)
                err = y.sdev
                upper_vals.append(mean+err)
                lower_vals.append(mean-err)

            ax.plot(x_vals, mean_vals, '-', lw=1., color=colors[color_i])
            ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color=colors[color_i])


        num_params = len(print_params)

        starting_height = .25
        plt.text(0.05, starting_height + num_params*.075, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof,2)}$", transform=ax.transAxes)
        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
            plt.text(0.05, starting_height + (num_params - height)*.075, rf"${param_text} = {fit.param(param_name)!s}$", transform=ax.transAxes)

        for height, text_add in enumerate(text_adds):
            text_height = 0.93 - height*.075
            plt.text(0.65, text_height, text_add, transform=ax.transAxes)

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def __call__(self, mom, z):
        return self._data[mom][z]


###################################################################################################
#     HELPER FUNCTIONS
###################################################################################################

markers = ['<', '^', '>', 'v', 'o', 'x', 's', '*', 'D', 'p', '3', '4', 'H']
line_styles = ['-', '--']
def plot_ioffe_time_dist_comp(plot_file, to_plot_map, a, Pxs, Ns, ope_funcs, ylabel, *text_adds):
    fig, ax = plt.subplots()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(ylabel)

    num_data = len(to_plot_map)
    disp = .33 * (num_data - 1) / 10
    displacements = np.linspace(-disp, disp, num_data)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(Pxs)))
    for color_i, px in enumerate(Pxs):
        min_z = None
        max_z = None
        for marker_i, (plot_label, plot_data) in enumerate(to_plot_map.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()
            for z, data in plot_data._data[px].items():
                ioffe = (2.*np.pi / Ns) * z * px
                if min_z is None or min_z > z:
                    min_z = z
                if max_z is None or max_z < z:
                    max_z = z

                x_vals.append(ioffe + displacements[marker_i])
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker=markers[marker_i], ms=2, capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i])

        z_vals = np.linspace(min_z, max_z, 1000)
        for line_style_i, (plot_label, ope_func) in enumerate(ope_funcs.items()):
            x_vals = list()
            y_vals = list()

            for z in z_vals:
                ioffe = (2.*np.pi / Ns) * z * px
                x_vals.append(ioffe)
                y_vals.append(ope_func(ioffe, z**2))

            ax.plot(x_vals, y_vals, line_styles[line_style_i], lw=1., color=colors[color_i])

    for color_i, px in enumerate(Pxs):
        ax.plot(np.NaN, np.NaN, c=colors[color_i], label=rf"$P_x = {px}$")

    ax2 = ax.twinx()
    for marker_i, plot_label in enumerate(to_plot_map.keys()):
        ax2.plot(np.NaN, np.NaN, marker=markers[marker_i], color='k', label=plot_label)
    ax2.get_yaxis().set_visible(False)

    if ope_funcs:
        ax3 = ax.twinx()
        for line_style_i, plot_label in enumerate(ope_funcs.keys()):
            ax3.plot(np.NaN, np.NaN, ls=line_styles[line_style_i], color='k', label=plot_label)
        ax3.get_yaxis().set_visible(False)

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    if ope_funcs:
        ax3.legend(loc='lower right')

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)
    plt.close()


line_styles = ['-', '--']
def plot_ioffe_time_dist(plot_file, ioffe_time_dist, a, Pxs, Ns, ope_funcs, ylabel, *text_adds):
    fig, ax = plt.subplots()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(ylabel)

    num_data = len(Pxs)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(Pxs)))
    for color_i, px in enumerate(Pxs):
        min_z = min(ioffe_time_dist._data[px].keys())
        max_z = max(ioffe_time_dist._data[px].keys())

        x_vals = list()
        y_vals = list()
        y_errs = list()
        for z, data in ioffe_time_dist._data[px].items():
            ioffe = (2.*np.pi / Ns) * z * px
            x_vals.append(ioffe)
            y_vals.append(data.mean)
            y_errs.append(data.sdev)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', ms=2, capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$P_x = {px}$")

        z_vals = np.linspace(min_z, max_z, 1000)
        for line_style_i, (plot_label, ope_func) in enumerate(ope_funcs.items()):
            x_vals = list()
            y_vals = list()

            for z in z_vals:
                ioffe = (2.*np.pi / Ns) * z * px
                x_vals.append(ioffe)
                y_vals.append(ope_func(ioffe, z**2))

            ax.plot(x_vals, y_vals, line_styles[line_style_i], lw=1., color=colors[color_i])

    if len(Pxs) > 1:
        plt.legend()

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)
    plt.close()

line_styles = ['-', '--']
def plot_ioffe_time_dist_with_systematic(plot_file, ioffe_time_dist, a, Pxs, Ns, ope_funcs, ylabel, *text_adds):
    fig, ax = plt.subplots()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(ylabel)

    num_data = len(Pxs)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(Pxs)))
    for color_i, px in enumerate(Pxs):
        min_z = min(ioffe_time_dist[px].keys())
        max_z = max(ioffe_time_dist[px].keys())

        x_vals = list()
        y_vals = list()
        y_stat_errs = list()
        y_tot_errs = list()
        for z, (mean_val, stat_err, sys_err) in ioffe_time_dist[px].items():
            ioffe = (2.*np.pi / Ns) * z * px
            x_vals.append(ioffe)
            y_vals.append(mean_val)
            y_stat_errs.append(stat_err)
            y_tot_errs.append(np.sqrt(stat_err**2 + sys_err**2))

        plt.errorbar(x_vals, y_vals, yerr=y_stat_errs, marker='.', ms=2, capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$P_x = {px}$")
        plt.errorbar(x_vals, y_vals, yerr=y_tot_errs, marker='.', ms=2, capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i])

        z_vals = np.linspace(min_z, max_z, 1000)
        for line_style_i, (plot_label, ope_func) in enumerate(ope_funcs.items()):
            x_vals = list()
            y_vals = list()

            for z in z_vals:
                ioffe = (2.*np.pi / Ns) * z * px
                x_vals.append(ioffe)
                y_vals.append(ope_func(ioffe, z**2))

            ax.plot(x_vals, y_vals, line_styles[line_style_i], lw=1., color=colors[color_i])

    if len(Pxs) > 1:
        plt.legend()

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)
    plt.close()


###############################################################################
# RatioZ0
###############################################################################

class RatioZ0(data_handler.DataType):

    def __init__(self, data, z0, dma, symmetrize=True):
        self._form_ratio(data, z0, dma, symmetrize)

    def _form_ratio(self, data, z0, dma, symmetrize):
        z0_data = data[z0]
        min_z = abs(min(data.keys()))
        max_z = max(data.keys())
        ZMAX = max_z if max_z > min_z else min_z

        self._data = dict()
        for z in range(z0+1, ZMAX+1):
            self._data[z] = np.exp(dma*(z - z0)) * data[z] / z0_data
            if symmetrize and -z in data:
                neg_res = np.exp(-dma*(z - z0)) * data[-z] / z0_data
                self._data[z] = 0.5*(neg_res + self._data[z])

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

    def plot(self, plot_file, a):
        fig, ax = plt.subplots()
        plt.xlabel(r"$z$ [fm]")
        plt.ylabel(r'$e^{\delta m (a) (z - z_0)} \frac{h(z, 0, a)}{h(z_0, 0, a)}$')

        for z, data in self._data.items():
            x_vals = list()
            y_vals = list()
            y_errs = list()

            x_vals.append(z*a)
            y_vals.append(data.mean)
            y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color='k')

        plt.legend()
        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()


    def plot_fit(self, plot_file, fit, fitted_data, a, print_params, *text_adds):
        fig, ax = plt.subplots()
        plt.xlabel(r"$z$ [fm]")
        plt.ylabel(r'$e^{\delta m (a) (z - z_0)} \frac{h(z, 0, a)}{h(z_0, 0, a)}$')

        for z, data in self._data.items():
            x_vals = list()
            y_vals = list()
            y_errs = list()

            x_vals.append(z*a)
            y_vals.append(data.mean)
            y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color='k')

        # plot fit
        zs = np.linspace(min(fitted_data._data.keys()), max(fitted_data._data.keys()), 500)

        x_vals = list()
        mean_vals = list()
        upper_vals = list()
        lower_vals = list()
        for z in zs:
            x_vals.append(z*a)
            y = fit.fit_function(z, fit.params)
            mean = y.mean
            mean_vals.append(mean)
            err = y.sdev
            upper_vals.append(mean+err)
            lower_vals.append(mean-err)

        ax.plot(x_vals, mean_vals, '-', lw=1., color='k')
        ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color='k')

        num_params = len(print_params)

        starting_height = .25
        plt.text(0.05, starting_height + num_params*.075, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof,2)}$", transform=ax.transAxes)
        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
            plt.text(0.05, starting_height + (num_params - height)*.075, rf"${param_text} = {fit.param(param_name)!s}$", transform=ax.transAxes)

        for height, text_add in enumerate(text_adds):
            text_height = 0.93 - height*.075
            plt.text(0.65, text_height, text_add, transform=ax.transAxes)

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def remove_data(self, zs_to_remove):
        new_data = dict()
        for z, data in sorted(self._data.items()):
            if z in zs_to_remove:
                continue

            new_data[z] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def __call__(self, z):
        return self._data[z]



###############################################################################
# Hybrid
###############################################################################


class Hybrid(data_handler.DataType):

    def __init__(self, data, ref_mom, dmap, L, C0, zs, Ns, complex_arg=None, test_data=None, renorm=True):
        self.moms = sorted(list(data[0].keys()))
        self.ref_mom = ref_mom
        self.dmap = dmap
        self.L = L
        self.C0 = C0
        self.zs = zs
        self.Ns = Ns
        self.complex_arg = complex_arg
        self.test_data = test_data

        if renorm:
            self._form_ratio(data)
        else:
            self._use_exact(data)

    def _use_exact(self, data):
        self._data = dict()
        for mom in self.moms:
            if mom <=self.ref_mom:
                continue

            self._data[mom] = dict()

            if data[0][mom].keys() != data[1][mom].keys():
                print("invalid")
                sys.exit()

            for z in data[0][mom].keys():
                self._data[mom][z] = data[0][mom][z] + 1.j*data[1][mom][z]

                if self.complex_arg == data_handler.ComplexArg.REAL:
                    self._data[mom][z] = self._data[mom][z].real
                elif self.complex_arg == data_handler.ComplexArg.IMAG:
                    self._data[mom][z] = self._data[mom][z].imag

    def _form_ratio(self, data):
        self._data = dict()
        if data[0].keys() != data[1].keys():
            print("Numerator Real and Imaginary data don't have same Px")
            sys.exit()

        if self.ref_mom not in data[0] or self.ref_mom not in data[1]:
            print(f"Data doesn't have ref mom = {self.ref_mom}")

        data0_z0 = data[0][self.ref_mom][0] + 1.j*data[1][self.ref_mom][0]
        data0_zs = data[0][self.ref_mom][self.zs] + 1.j*data[1][self.ref_mom][self.zs]

        for mom in self.moms:
            if mom <=self.ref_mom:
                continue

            if data[0][mom].keys() != data[1][mom].keys():
                print("invalid")
                sys.exit()

            data_z0 = data[0][mom][0] + 1.j*data[1][mom][0]

            norm = data0_z0 / data_z0

            self._data[mom] = dict()
            for z in data[0][mom].keys():
                data_z = data[0][mom][z] + 1.j*data[1][mom][z]
                data0_z = data[0][self.ref_mom][z] + 1.j*data[1][self.ref_mom][z]

                C0_z = self.C0(z**2) if z != 0 else self.C0(.00000000000001**2)
                C0_zs = self.C0(self.zs**2)

                if abs(z) <= self.zs:
                    if self.test_data is not None:
                        test_data_z = self.test_data[0][mom][z] + 1.j*self.test_data[1][mom][z]
                        self._data[mom][z] = test_data_z * (C0_z - self.L*z**2) / C0_z
                    else:
                        self._data[mom][z] = norm * (data_z/data0_z) * (C0_z - self.L*z**2) / C0_z
                else:
                    self._data[mom][z] = norm * np.exp(self.dmap*abs(z - self.zs)) * (data_z/data0_zs) * (C0_zs - self.L*self.zs**2) / C0_zs

                if self.complex_arg == data_handler.ComplexArg.REAL:
                    self._data[mom][z] = self._data[mom][z].real
                elif self.complex_arg == data_handler.ComplexArg.IMAG:
                    self._data[mom][z] = self._data[mom][z].imag

    def get_samples(self):
        _samples = list()
        for mom, mom_data in self._data.items():
            for z, z_data in mom_data.items():
                _samples.append(z_data.samples)

        return np.array(_samples).T

    def symmetrize(self, sym, keep_neg=False):
        new_data = dict()
        for mom, data_z in self._data.items():
            new_data[mom] = dict()
            zs = set([abs(z) for z in data_z.keys()])
            for z in zs:
                if z == 0 or -z not in data_z:
                    new_data[mom][z] = data_z[z]
                    if keep_neg:
                        new_data[mom][-z] = sym*data_z[z]

                else:
                    new_data[mom][z] = 0.5*(data_z[z] + sym*data_z[-z])
                    if keep_neg:
                        new_data[mom][-z] = 0.5*(sym*data_z[z] + data_z[-z])

        self._data = new_data

    @property
    def num_samples(self):
        return list(list(self._data.values())[0].values())[0].num_samples

    @property
    def num_data(self):
        return len(self.get_organized_independent_data())

    @property
    def real(self):
        if self.complex_arg == data_handler.ComplexArg.REAL:
            return self
        elif self.complex_arg == data_handler.ComplexArg.IMAG:
            raise ValueError("Can't make imaginary data real")
        else:
            new_data = dict()
            for mom, data_z in self._data.items():
                new_data[mom] = dict()
                for z, data in data_z.items():
                    new_data[mom][z] = data.real

            new_obj = copy.deepcopy(self)
            new_obj._data = new_data
            new_obj.complex_arg = data_handler.ComplexArg.REAL

            return new_obj

    @property
    def imag(self):
        if self.complex_arg == data_handler.ComplexArg.IMAG:
            return self
        elif self.complex_arg == data_handler.ComplexArg.REAL:
            raise ValueError("Can't make real data imag")
        else:
            new_data = dict()
            for mom, data_z in self._data.items():
                new_data[mom] = dict()
                for z, data in data_z.items():
                    new_data[mom][z] = data.imag

            new_obj = copy.deepcopy(self)
            new_obj._data = new_data
            new_obj.complex_arg = data_handler.ComplexArg.IMAG

            return new_obj


    def get_independent_data(self):
        ioffe_list = list()
        zsq_list = list()
        for mom, mom_data in self._data.items():
            for z in mom_data.keys():
                ioffe = (2.*np.pi / self.Ns) * z * mom
                ioffe_list.append(ioffe)
                zsq_list.append(z**2)

        return (np.array(ioffe_list), np.array(zsq_list))

    def get_organized_independent_data(self):
        data_list = list()
        for mom, mom_data in self._data.items():
            for z in mom_data.keys():
                ioffe = (2.*np.pi / self.Ns) * z * mom
                data_list.append((ioffe, z**2))

        return np.array(data_list)

    def remove_data(self, mom_range, z_range):
        new_data = dict()
        for mom in mom_range:
            if mom not in self._data:
                continue

            new_data[mom] = dict()
            for z in z_range:
                if z not in self._data[mom]:
                    continue

                new_data[mom][z] = self._data[mom][z]

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj


    def do_fourier(self, extrap_fit_name, extrap_fit, px, zLreal, zLimag, num_data=4000, y_range=(-1.99951, 1.99949)):
        if self.complex_arg != None:
            raise ValueError("Need complex data for fourier")


        lLreal = (2.*np.pi / self.Ns) * zLreal * px
        lLimag = (2.*np.pi / self.Ns) * zLimag * px
        if not np.isclose(lLreal/zLreal, lLimag/zLimag):
            print(lLreal/zLreal)
            print(lLimag/zLimag)
            raise ValueError("Pz not matching between real and imaginary")

        int_func = extrap_integral_map[extrap_fit_name](lLreal/zLreal, lLreal, lLimag)

        yvals = np.linspace(y_range[0], y_range[1], num_data)

        fourier_result = np.zeros(num_data, dtype=object)
        for y_i, y in tqdm.tqdm(enumerate(yvals)):

            # do DFT from 0 to zL
            dft_int = 0.
            for z in range(0, zLreal+1):
                ioffe = (2.*np.pi / self.Ns) * z * px
                data = self._data[px][z]
                dft_int += (lLreal/(np.pi*(zLreal+1.)))*data.real*np.cos(y*ioffe)

            for z in range(0, zLimag+1):
                ioffe = (2.*np.pi / self.Ns) * z * px
                data = self._data[px][z]
                dft_int -= (lLimag/(np.pi*(zLimag+1.)))*data.imag*np.sin(y*ioffe)

            # do int from zL to +inf
            ana_int = int_func(y, extrap_fit[0], extrap_fit[1])

            fourier_result[y_i] = dft_int + data_handler.Data(ana_int)

        return yvals, fourier_result


    def plot(self, plot_file, moms, zmax, a, ylabel, ylim=None):
        fig, ax = plt.subplots()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(ylabel)

        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

        num_tseps = len(self._data)
        colors = plt.cm.tab10(list(range(num_tseps)))

        for color_i, (mom, data_z) in enumerate(self._data.items()):
            if mom not in moms:
                continue

            x_vals = list()
            y_vals = list()
            y_errs = list()

            min_z = None
            max_z = None
            for z, data in data_z.items():
                if abs(z) > zmax:
                    continue
                if min_z is None or min_z > z:
                    min_z = z
                if max_z is None or max_z < z:
                    max_z = z

                ioffe = (2.*np.pi / self.Ns) * z * mom
                x_vals.append(ioffe)
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$n_x = {mom}$")
        plt.legend()
        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def plot_fit(self, plot_file, fit, fitted_data, a, ylabel, print_params, extrap_zmax):
        fig, ax = plt.subplots()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(ylabel)

        num_momenta = len(self._data)
        colors = plt.cm.tab10(list(range(num_momenta)))

        # plot data
        for color_i, (mom, data_z) in enumerate(self._data.items()):
            x_vals = list()
            y_vals = list()
            y_errs = list()

            for z, data in data_z.items():
                ioffe = (2.*np.pi / self.Ns) * z * mom
                x_vals.append(ioffe)
                y_vals.append(data.mean)
                y_errs.append(data.sdev)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, ls='none', color=colors[color_i], label=rf"$n_x = {mom}$")
        # plot extrapolation
        for color_i, mom in enumerate(self._data.keys()):
            if mom not in fitted_data._data:
                continue

            fitdata_z = fitted_data._data[mom]
            fitzs = np.linspace(min(fitdata_z.keys()), max(fitdata_z.keys()), 500)
            extrapzs = np.linspace(max(fitdata_z), extrap_zmax, 500)

            ioffes = list()
            mean_vals = list()
            upper_vals = list()
            lower_vals = list()
            for z in fitzs:
                ioffe = (2.*np.pi / self.Ns) * z * mom
                ioffes.append(ioffe)
                x_vals.append(ioffe)
                y = fit.fit_function((ioffe, z**2), fit.params)
                mean = y.mean
                mean_vals.append(mean)
                err = y.sdev
                upper_vals.append(mean+err)
                lower_vals.append(mean-err)

            ax.plot(ioffes, mean_vals, '-', lw=1., color=colors[color_i])
            ax.fill_between(ioffes, lower_vals, upper_vals, alpha=0.4, lw=0, color=colors[color_i])

            ioffes = list()
            mean_vals = list()
            upper_vals = list()
            lower_vals = list()
            for z in extrapzs:
                ioffe = (2.*np.pi / self.Ns) * z * mom
                ioffes.append(ioffe)
                x_vals.append(ioffe)
                y = fit.fit_function((ioffe, z**2), fit.params)
                mean = y.mean
                mean_vals.append(mean)
                err = y.sdev
                upper_vals.append(mean+err)
                lower_vals.append(mean-err)

            ax.plot(ioffes, mean_vals, '-', lw=1., color=colors[color_i])
            ax.fill_between(ioffes, lower_vals, upper_vals, alpha=0.2, lw=0, color=colors[color_i])

        num_params = len(print_params)

        starting_height = .25
        plt.text(0.05, starting_height + num_params*.075, rf"$\chi^2_{{\rm dof}} = {round(fit.chi2_dof,2)}$", transform=ax.transAxes)
        for height, (param_text, param_data) in enumerate(print_params.items(), 1):
            plt.text(0.05, starting_height + (num_params - height)*.075, rf"${param_text} = {param_data!s}$", transform=ax.transAxes)

        if num_momenta > 1:
            plt.legend()

        plt.tight_layout(pad=0.80)
        plt.savefig(plot_file)
        plt.close()

    def __call__(self, mom, z):
        return self._data[mom][z]


###################################################################################################
#     HELPER FUNCTIONS
###################################################################################################

def get_exp_decay_model_func(m_min, d_min):
    @fitter.fit_name(f"exp_decay_model")
    @fitter.fit_num_params(3)
    @fitter.fit_params(['A', 'm', 'd'])
    def fit_function(x, p):
        return p['A']*gv.exp(-(p['m'] + m_min)*gv.sqrt(x[1])) / (gv.abs(x[0])**(p['d'] + d_min))

    return fit_function

def expint(n, z):
    return complex(mpmath.expint(n, z))

expint_np = np.frompyfunc(expint, 2, 1)

def get_exp_decay_model_int_func(Pz, lLreal, lLimag):
    def int_function(x, preal, pimag):
        re_part = (preal['A']*lLreal**(1.-preal['d']))*(expint_np(preal['d'], lLreal*(preal['m'] - 1.j*Pz*x)/Pz)
                                                      + expint_np(preal['d'], lLreal*(preal['m'] + 1.j*Pz*x)/Pz))

        im_part = (pimag['A']*lLimag**(1.-pimag['d']))*(expint_np(pimag['d'], lLimag*(pimag['m'] - 1.j*Pz*x)/Pz)
                                                      - expint_np(pimag['d'], lLimag*(pimag['m'] + 1.j*Pz*x)/Pz))

        int_val = 1./(2.*np.pi)*(re_part + 1.j*im_part)

        if not np.all(np.isreal(int_val)):
            raise ValueError("Integral is imaginary!")

        int_val = int_val.astype(np.complex128)

        return int_val.real

    return int_function

'''
def get_exp_decay_model_int_lMax_func(Pz, lL, lMax):
  def int_function(x, preal, pimag):
    expint_real_L = expint_np(preal['d'], lL*(preal['m'] - 1.j*Pz*x)/Pz) + expint_np(preal['d'], lL*(preal['m'] + 1.j*Pz*x)/Pz)
    expint_real_max = expint_np(preal['d'], lMax*(preal['m'] - 1.j*Pz*x)/Pz) + expint_np(preal['d'], lMax*(preal['m'] + 1.j*Pz*x)/Pz)
    expint_imag_L = expint_np(pimag['d'], lL*(pimag['m'] - 1.j*Pz*x)/Pz) - expint_np(pimag['d'], lL*(pimag['m'] + 1.j*Pz*x)/Pz)
    expint_imag_max = expint_np(pimag['d'], lMax*(pimag['m'] - 1.j*Pz*x)/Pz) - expint_np(pimag['d'], lMax*(pimag['m'] + 1.j*Pz*x)/Pz)

    re_part = preal['A']/(lL*lMax)**preal['d'] * (lL*lMax**preal['d']*expint_real_L - lL**preal['d']*lMax*expint_real_max)
    im_part = pimag['A']/(lL*lMax)**pimag['d'] * (lL*lMax**pimag['d']*expint_imag_L - lL**pimag['d']*lMax*expint_imag_max)

    int_val = 1./(2.*np.pi)*(re_part + 1.j*im_part)

    #if not np.all(np.isreal(int_val)):
    #  raise ValueError("Integral is imaginary!")

    int_val = int_val.astype(np.complex128)

    return int_val.real

  return int_function
'''


def get_pow_decay_model_func(d_min):
    @fitter.fit_name(f"pow_decay_model")
    @fitter.fit_num_params(2)
    @fitter.fit_params(['A', 'd'])
    def fit_function(x, p):
        return p['A'] / (gv.abs(x[0])**(p['d'] + d_min))

    return fit_function

def hyp1f2(a1, b1, b2, z):
    return complex(mpmath.hyp1f2(a1, b1, b2, z))

hyp1f2_np = np.frompyfunc(hyp1f2, 4, 1)

def get_pow_decay_model_int_func(Pz, lLreal, lLimag):
    def int_function(x, preal, pimag):
        re_part1 = (preal['A']*lLreal**(1.-preal['d'])*hyp1f2_np(0.5-preal['d']/2., 0.5, 1.5-preal['d']/2., -.25*(lLreal*x)**2))/(preal['d']-1.)
        re_part2 = preal['A']*abs(x)**(preal['d']-1.)*scipy.special.gamma(1.-preal['d'])*np.sin(preal['d']*np.pi/2.)

        im_part1 = (pimag['A']*lLimag**(2.-pimag['d'])*x*hyp1f2_np(1.-pimag['d']/2., 1.5, 2.-pimag['d']/2., -.25*(lLimag*x)**2))/(pimag['d']-2.)
        im_part2 = pimag['A']*abs(x)**(pimag['d'])*scipy.special.gamma(1.-pimag['d'])*np.cos(pimag['d']*np.pi/2.)/x

        int_val = 1./(np.pi)*(re_part1 + real_part2 - 1.j*(im_part1 + im_part2))

        if not np.all(np.isreal(int_val)):
            raise ValueError("Integral is imaginary!")

        int_val = int_val.astype(np.complex128)

        return int_val.real

    return int_function

extrap_integral_map = {
    'exp_decay_model': get_exp_decay_model_int_func,
    'pow_decay_model': get_pow_decay_model_int_func,
}
