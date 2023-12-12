import os

import numpy as np
import scipy.linalg
import warnings
import copy

import gvar as gv

import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
#plt.rc('font', family='serif', serif=['Times'], size=55)
#plt.rc('text', usetex=True)
plt.style.use('/home/ahanlon/code/lattice/qcd_analysis/plots.mplstyle')

import data_handler
#import lsqfit_fitter as fitter
import fitter


q_colors = {
    'red':   [(0.0,  1.0, 1.0),
              (0.1,  0.3, 0.0),
              (1.0,  0.0, 0.0)],
    'green': [(0.0,  0.0, 0.0),
              (0.1, 0.0, 0.3),
              (0.5, 1.0, 1.0),
              (0.9, 0.3, 0.0),
              (1.0,  0.0, 0.0)],
    'blue':  [(0.0,  0.0, 0.0),
              (0.9,  0.0, 0.3),
              (1.0,  1.0, 1.0)]
}
q_color_map = matplotlib.colors.LinearSegmentedColormap('q_color_map', q_colors, 256)

hc = 6.62607015e-34 / 1.602176634e-19 / (2*np.pi) * 299792458 * 1e6 # exact: new SI

class C2ptData(data_handler.DataType):

    def __init__(self, data, normalize=False):

        self.ratio = False

        self.normalization = 1.

        if normalize:
            normalize_tsep = min(data.keys()) + 1
            normalization = np.abs(data[normalize_tsep].mean)
            for tsep in data.keys():
                data[tsep] = (1./normalization) * data[tsep]

            self.normalization = normalization

        self._data = data
    
    @property
    def data_name(self):
        return "C2pt"

    def get_samples(self):
        _samples = list()
        for tsep_data in self._data.values():
            _samples.append(tsep_data.samples)

        return np.array(_samples).T

    def get_data(self):
        return self._data

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

            if tmax > 0 and tsep > tmax:
                break

            corr_val = data.mean
            corr_err = data.sdev
            if (abs(corr_val) / corr_err) < tmax_rel_error:
                break

            new_data[tsep] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    @property
    def real(self):
        new_data = dict()
        for tsep, data in sorted(self._data.items()):
            new_data[tsep] = data.real

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj


    def get_effective_energy(self, dt=1):
        eff_energy = dict()
        for tsep in self._data.keys():
            tsep_dt = tsep + dt
            if tsep_dt not in self._data:
                continue

            x_val = (tsep + tsep_dt)/2

            data = self._data[tsep]
            data_dt = self._data[tsep_dt]

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    data_eff_energy = (-1./dt)*np.log(data_dt/data)
                except Warning as e:
                    #print(f"Warning for tsep={tsep}: {e}")
                    continue

            eff_energy[x_val] = data_eff_energy

        return eff_energy

    def print_data(self):
        for tsep, tsep_data in self._data.items():
            print(f"C({tsep}) = {tsep_data}")

    '''
    def plot_tmin(self, plot_dir, name, init_guess, priors, num_exps, shifted, tmin_min, tmin_max, tmax, tmax_rel_error=0.):
      num_params = 2*num_exps
      fit_function = get_shift_fit_function2(num_exps) if shifted else get_fit_function(num_exps)

      x_vals = dict()
      y_vals = dict()
      y_errs = dict()
      for exp in range(num_exps):
        x_vals[exp] = list()
        y_vals[exp] = list()
        y_errs[exp] = list()

      ps = dict()

      for tmin in range(tmin_min, tmin_max+1):
        fit_data = self.remove_data(tmin, tmax, tmax_rel_error)
        if len(fit_data.get_independent_data()) - num_params <= 0:
          continue

        fit = fitter.Fitter(fit_data, fit_function)
        fit.do_fit(init_guess, priors)

        ps[tmin] = fit.Q

        x_vals[0].append(tmin)
        y_vals[0].append(fit.param('E0').mean)
        y_errs[0].append(fit.param('E0').sdev)

        for exp in range(1, num_exps):
          x_vals[exp].append(tmin)
          if shifted:
            E_data = fit.param(f'D{exp}0') + fit.param('E0')
            y_vals[exp].append(E_data.mean)
            y_errs[exp].append(E_data.sdev)
          else:
            y_vals[exp].append(fit.param(f'E{exp}').mean)
            y_errs[exp].append(fit.param(f'E{exp}').sdev)

      for exp in range(num_exps):
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.xlabel(r"$t_{\rm min}$")
        plt.ylabel(r"$a E_{\rm fit}$")

        for fit_i, tmin in enumerate(x_vals[exp]):
          plt.errorbar(tmin, y_vals[exp][fit_i], yerr=y_errs[exp][fit_i], color=q_color_map(ps[tmin]), marker='.', capsize=2, capthick=.5, lw=.5, ls='none')

        plt.tight_layout(pad=0.80)
        plotname = f"Nexp{num_exps}_E{exp}_{name}"
        plot_file = os.path.join(plot_dir, f"{self.name}_{plotname}.pdf")
        plt.savefig(plot_file)
        plt.close()
    '''

    def do_fit(self, num_exps, log_amps, tmin, tmax, tmax_rel_error, resamplings=True):
        eff_energy = self.get_effective_energy(1)[tmin + 0.5]
        eff_energy_val = eff_energy.mean

        eff_amp = np.exp(eff_energy*tmin) * self(tmin).real
        eff_amp_val = eff_amp.mean

        guesses = {
            'E0': eff_energy_val,
            'A0': eff_amp_val,
        }

        if self.ratio:
            log_params = ['A0']
        else:
            log_params = ['E0', 'A0']

        for n in range(1, num_exps):
            guesses[f'dE{n},{n-1}'] = .1
            guesses[f'R{n}'] = .1

            if not self.ratio:
                log_params.append(f'dE{n},{n-1}')

            if log_amps:
                log_params.append(f'R{n}')

        fit_function = get_fit_function(num_exps)
        corr_data = self.remove_data(tmin, tmax, tmax_rel_error)
        fit = fitter.Fitter(corr_data, fit_function)
        if fit.do_fit(guesses, log_params, True, False, resamplings):
            print(fit.output())

            return fit

        return None


    def plot_effective_energy(self, plot_dir, name, dt=1):
        fig, ax = plt.subplots()
        plt.xlabel(r"$t_{\rm sep}$")
        plt.ylabel(r"$a E_{\rm eff} (t_{\rm sep})$")

        x_vals = list()
        y_vals = list()
        y_errs = list()

        eff_energy = self.get_effective_energy(dt)

        for tsep in self._data.keys():
            tsep_dt = tsep + dt
            if tsep_dt not in self._data:
                continue

            x_val = (tsep + tsep_dt)/2
            if x_val not in eff_energy:
                continue

            data_eff_energy = eff_energy[x_val]
            y_vals.append(data_eff_energy.mean)
            y_errs.append(data_eff_energy.sdev)
            x_vals.append(x_val)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{name}.pdf")
        plt.savefig(plot_file)
        plt.close()


    def plot_effective_energy_with_fit(self, plot_dir, name, fitter, fitted_data, print_params={}, dt=1):
        fig, ax = plt.subplots()
        plt.xlabel(r"$t_{\rm sep}$")
        plt.ylabel(r"$a E_{\rm eff} (t_{\rm sep})$")

        x_vals = list()
        y_vals = list()
        y_errs = list()

        eff_energy = self.get_effective_energy(dt)

        for tsep in self._data.keys():
            tsep_dt = tsep + dt
            if tsep_dt not in self._data:
                continue

            x_val = (tsep + tsep_dt)/2
            if x_val not in eff_energy:
                continue

            data_eff_energy = eff_energy[x_val]
            y_vals.append(data_eff_energy.mean)
            y_errs.append(data_eff_energy.sdev)
            x_vals.append(x_val)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        x_vals = list(np.linspace(min(fitted_data.get_independent_data()), max(fitted_data.get_independent_data()), 500))
        mean_vals = list()
        upper_vals = list()
        lower_vals = list()
        for x in x_vals:
            y = fitter.fit_function(x - dt/2., fitter.params)
            y_dt = fitter.fit_function(x + dt/2., fitter.params)
            data_eff_energy = (-1./dt)*np.log(y_dt/y)

            mean = data_eff_energy.mean
            mean_vals.append(mean)
            err = data_eff_energy.sdev
            upper_vals.append(mean+err)
            lower_vals.append(mean-err)

        ax.plot(x_vals, mean_vals, '-', lw=1., color='blue')
        ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color='blue')

        num_params = len(print_params)

        plt.text(0.05, 0.07 + num_params*.075, rf"$\chi^2_{{\rm dof}} = {round(fitter.chi2_dof,2)}$", transform=ax.transAxes)
        for height, (param_name, param_text) in enumerate(print_params.items(), 1):
            plt.text(0.05, 0.07 + (num_params - height)*.075, rf"${param_text} = {fitter.param(param_name)!s}$", transform=ax.transAxes)

        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{name}.pdf")
        plt.savefig(plot_file)
        plt.close()


    def plot_data_to_fit_ratio(self, plot_dir, name, fitters):
        fig, ax = plt.subplots()

        plt.xlabel(r"$t_{\rm sep}$")
        plt.ylabel(r"$C^{\rm dat} / C^{\rm fit}$")

        colors = plt.cm.rainbow(np.linspace(0, 1, len(fitters)))

        plt.axhline(y=1., color='red', linestyle='dotted')

        for color_i, (fit_name, fitter) in enumerate(fitters.items()):

            x_vals = list()
            y_vals = list()
            y_errs = list()
            for tsep, data in self._data.items():
                data_ratio = data / fitter.fit_function(tsep, fitter.params)
                ratio_mean = data_ratio.mean
                ratio_err = data_ratio.sdev

                x_vals.append(tsep)
                y_vals.append(ratio_mean)
                y_errs.append(ratio_err)

            plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', capsize=2, capthick=.5, lw=.5, color=colors[color_i], label=fit_name, ls='none')

        #plt.legend(loc='upper right')
        plt.legend()
        plt.tight_layout(pad=0.80)
        plot_file = os.path.join(plot_dir, f"{self.name}_{name}.pdf")
        plt.savefig(plot_file)
        plt.close()

    def get_shifted_c2pt(self, shift):
        new_data = dict()
        for tsep, tsep_data in self._data.items():
            tsep_prime = tsep - shift
            if tsep_prime < 0:
                continue
            new_data[tsep_prime] = tsep_data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data
        new_obj._shift = shift

        return new_obj

    def get_prony_matrix(self, N, delta=1):
        mat = np.empty((N,N,), dtype=object)
        for i in range(N):
            for j in range(N):
                mat[i,j] = self.get_shifted_c2pt(i+j)

        return C2ptMatrixData(mat)


    def __call__(self, tsep, normalize=True):
        if normalize:
            return self._data[tsep]
        else:
            return self.normalization * self._data[tsep]

    def get_ratio_correlator(self, corr_list):
        new_data = dict()
        for tsep, tsep_data in self._data.items():
            denom_tsep_data = list()
            all_tseps = True
            for corr in corr_list:
                if tsep not in corr._data:
                    all_tseps = False
                    break

                denom_tsep_data.append(corr._data[tsep])

            if not all_tseps:
                continue

            new_data[tsep] = tsep_data / np.prod(denom_tsep_data)

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data
        new_obj.ratio = True

        return new_obj

    '''
      TODO: what should be returned for num_exps > 1 ?
    '''
    def get_amplitude(self, non_int_corrs, tmin, tmax, tmax_rel_error, num_exps):
        if non_int_corrs:
            self = self.get_ratio_correlator(non_int_corrs)
            non_int_amps = list()
            for non_int_corr in non_int_corrs:
                non_int_amps.append(non_int_corr.get_amplitude([], tmin, tmax, tmax_rel_error, num_exps))

        log_amps = not non_int_corrs

        fit_result = self.do_fit(num_exps, log_amps, tmin, tmax, tmax_rel_error)

        if non_int_corrs:
            return fit_result.param('A0') * np.prod(non_int_amps)
        else:
            return fit_result.param('A0')



###################################################################################################
#     MODELS
###################################################################################################

def get_fit_function(num_states):
    @fitter.fit_name(f"{num_states}-exp")
    @fitter.fit_num_params(2*num_states)
    def fit_function(x, p):
        f = 1.
        for i in range(1, num_states):
            fi = p[f'R{i}']
            for j in range(i):
                fi *= gv.exp(-p[f'dE{j+1},{j}']*x)
            f += fi
        f *= p['A0']*gv.exp(-p['E0']*x)
        return f

    return fit_function

'''
def get_fit_function(num_states):
  @fitter.fit_name(f"{num_states}-exp")
  @fitter.fit_num_params(2*num_states)
  def fit_function(x, p):
    f = 1.
    for i in range(1, num_states):
      fi = p[f'R{i}']*p[f'R{i}']
      for j in range(1, i+1):
        fi *= gv.exp(-p[f'dE{j},{j-1}']*x)
      f += fi
    f *= p['A0']*p['A0']*gv.exp(-p['E0']*x)
    return f

  return fit_function
'''

# TODO: currently assumes degenerate particles
def get_ratio_fit_function(num_states, single_hadron_psqs):
    @fitter.fit_name(f"ratio-{num_states}-exp")
    @fitter.fit_num_params(2*num_states + 2*(num_states-1)*len(set(single_hadron_psqs)))
    def fit_function(x, p):
        f_num = 1.
        f_den = [1. for x in single_hadron_psqs]
        for i in range(1, num_states):
            f_num += p[f'R{i}']**2 * gv.exp(-p[f'dE{i},0']*x)
            f_den_new = list()
            for f_den_i, psq in zip(f_den, single_hadron_psqs):
                f_den_i += p[f'r{psq},{i}']**2 * gv.exp(-p[f'dE{psq},{i},0']*x)
                f_den_new.append(f_den_i)

            f_den = f_den_new

        f = p['R0']**2 * gv.exp(-p[f'dE0']*x) * f_num / np.prod(f_den)

        return f

    return fit_function


class C2ptModel(data_handler.FitFunction):

    def __init__(self, num_states):
        self.num_states = num_states

    @property
    def fit_name(self):
        return f"{self.num_states}-exp"

    @property
    def params(self):
        _params = ['E0']
        for i in range(1, self.num_states):
            _params.append(f'dE{i},{i-1}')

        _params.append('A0')
        for i in range(1, self.num_states):
            _params.append(f'R{i}')

        return _params


    def function(self, x, p):
        f = 1.
        for i in range(1, self.num_states):
            fi = p[f'R{i}']
            for j in range(i):
                fi *= np.exp(-p[f'dE{j+1},{j}']*x)
            f += fi
        f *= p['A0']*np.exp(-p['E0']*x)
        return f


    def get_init_guesses(self, c2pt_data, tsep):
        init_guesses_dict = dict()

        eff_energy = c2pt_data.get_effective_energy(1)[tsep + 0.5]
        init_guesses_dict['E0'] = eff_energy.mean

        eff_amp = np.exp(eff_energy*tsep) * c2pt_data(tsep).real
        init_guesses_dict['A0'] = eff_amp.mean

        for param in self.params:
            if param == 'E0' or param == 'A0':
                continue

            init_guesses_dict[param] = .1

        return init_guesses_dict
                



class C2ptSingleHadronConspiracyModel(data_handler.FitFunction):

    def __init__(self, num_single_hadron_states, particle=''):
        self.num_single_hadron_states = num_single_hadron_states
        self.particle = particle

    @property
    def fit_name(self):
        return f"single-hadron-conspiracy_{self.num_single_hadron_states}-exp-single-hadron{self.particle}"

    @property
    def particle_str(self):
        if self.particle:
            return f"({self.particle})"
        else:
            return self.particle

    @property
    def params(self):
        _params = [f'E{self.particle_str}0']
        for i in range(1, self.num_single_hadron_states):
            _params.append(f'DE{self.particle_str}{i},{i-1}')

        _params.append(f'A{self.particle_str}0')
        for i in range(1, self.num_single_hadron_states):
            _params.append(f'r{self.particle_str}{i}')

        return _params


    def function(self, x, p):
        f = 1.
        for i in range(1, self.num_single_hadron_states):
            fi = p[f'r{self.particle_str}{i}']
            for j in range(i):
                fi *= np.exp(-p[f'DE{self.particle_str}{j+1},{j}']*x)
            f += fi
        f *= p[f'A{self.particle_str}0']*np.exp(-p[f'E{self.particle_str}0']*x)
        return f


    def get_init_guesses(self, c2pt_data, tsep):
        init_guesses_dict = dict()

        eff_energy = c2pt_data.get_effective_energy(1)[tsep + 0.5]
        init_guesses_dict[f'E{self.particle_str}0'] = eff_energy.mean

        eff_amp = np.exp(eff_energy*tsep) * c2pt_data(tsep).real
        init_guesses_dict[f'A{self.particle_str}0'] = eff_amp.mean

        for param in self.params:
            if param == f'E{self.particle_str}0' or param == f'A{self.particle_str}0':
                continue

            init_guesses_dict[param] = .1

        return init_guesses_dict


class C2ptTwoHadronConspiracyModel(data_handler.FitFunction):

    def __init__(self, num_single_hadron_states, degenerate=False):
        self.num_single_hadron_states = num_single_hadron_states
        self.degenerate = degenerate

    @property
    def fit_name(self):
        if self.degenerate:
            return f"two-hadron-conspiracy_{self.num_single_hadron_states}-exp-deg-single-hadron"
        else:
            return f"two-hadron-conspiracy_{self.num_single_hadron_states}-exp-single-hadron"

    @property
    def params(self):
        if self.degenerate:
            _params = []
            for i in range(self.num_single_hadron_states):
                for j in range(i+1):
                    _params.append(f'dE{i},{j}')

            _params.append('E0')
            for i in range(1, self.num_single_hadron_states):
                _params.append(f'DE{i},{i-1}')

            _params.append('A0,0')
            for i in range(1, self.num_single_hadron_states):
                for j in range(i+1):
                    _params.append(f'r{i},{j}')

        else:
            _params = []
            for i in range(self.num_single_hadron_states):
                for j in range(self.num_single_hadron_states):
                    _params.append(f'dE{i},{j}')

            _params.append('E(1)0')
            _params.append('E(2)0')
            for i in range(1, self.num_single_hadron_states):
                _params.append(f'DE(1){i},{i-1}')
                _params.append(f'DE(2){i},{i-1}')

            _params.append('A0,0')
            for i in range(self.num_single_hadron_states):
                for j in range(self.num_single_hadron_states):
                    if i == 0 and j == 0:
                        continue
                    _params.append(f'r{i},{j}')

        return _params


    def function(self, x, p):
        f = 0.

        for i1 in range(self.num_single_hadron_states):
            for i2 in range(self.num_single_hadron_states):
                if i1 == 0 and i2 == 0:
                    f += 1.
                else:
                    if self.degenerate:
                        if i1 > i2:
                            fi = p[f'r{i1},{i2}']*np.exp(-p[f'dE{i1},{i2}'])
                        else:
                            fi = p[f'r{i2},{i1}']*np.exp(-p[f'dE{i2},{i1}'])
                    else:
                        fi = p[f'r{i1},{i2}']*np.exp(-p[f'dE{i1},{i2}'])

                    for i in range(i1):
                        if self.degenerate:
                            fi *= np.exp(-p[f'DE{i+1},{i}']*x)
                        else:
                            fi *= np.exp(-p[f'DE(1){i+1},{i}']*x)
                    for i in range(i2):
                        if self.degenerate:
                            fi *= np.exp(-p[f'DE{i+1},{i}']*x)
                        else:
                            fi *= np.exp(-p[f'DE(2){i+1},{i}']*x)
                    f += fi

        if self.degenerate:
            f *= p['A0,0']*np.exp(-(2.*p['E0'] + p['dE0,0'])*x)
        else:
            f *= p['A0,0']*np.exp(-(p['E(1)0'] + p['E(2)0'] + p['dE0,0'])*x)

        return f

    def get_init_guesses(self, c2pt_interacting, c2pt_noninteracting1, c2pt_noninteracting2):
        """
        Args: c2pt_interacting - 2-tuple: First element is data for interacting correlators.
                                          Second element is tsep value to use
              c2pt_noninteracting1 - 2-tuple: Same as above but for first particle
              c2pt_noninteracting2 - 2-tuple: Same as above but for second particle
        """
        init_guesses_dict = dict()
        ignore_params = list()

        # get amplitude of interacting correlators
        eff_energy = c2pt_interacting[0].get_effective_energy(1)[c2pt_interacting[1] + 0.5]
        eff_amp = np.exp(eff_energy*c2pt_interacting[1]) * c2pt_interacting[0](c2pt_interacting[1]).real
        init_guesses_dict['A0,0'] = eff_amp.mean
        ignore_params.append('A0,0')

        # get effecitve energy from particles
        eff_energy1 = c2pt_noninteracting1[0].get_effective_energy(1)[c2pt_noninteracting1[1] + 0.5]
        if self.degenerate:
            init_guesses_dict['E0'] = eff_energy1.mean
            ignore_params.append('E0')
        else:
            eff_energy2 = c2pt_noninteracting2[0].get_effective_energy(1)[c2pt_noninteracting2[1] + 0.5]
            init_guesses_dict['E(1)0'] = eff_energy1.mean
            init_guesses_dict['E(2)0'] = eff_energy2.mean
            ignore_params.append('E(1)0')
            ignore_params.append('E(2)0')

        # get interacting shift
        c2pt_ratio = c2pt_interacting[0].get_ratio_correlator([c2pt_noninteracting1[0], c2pt_noninteracting2[0]])
        eff_energy = c2pt_ratio.get_effective_energy(1)[c2pt_interacting[1] + 0.5]
        init_guesses_dict['dE0,0'] = eff_energy.mean
        ignore_params.append('dE0,0')

        for param in self.params:
            if param in ignore_params:
                continue

            init_guesses_dict[param] = .1

        return init_guesses_dict


###################################################################################################
#     HELPER FUNCTIONS
###################################################################################################

def plot_dispersion(energies, plot_file, Ns):
    fit, ax = plt.subplots()
    plt.xlabel(r"$n_z$")
    plt.ylabel(r'$E(P_z)$')

    m0 = energies[0].mean
    p_range = np.linspace(0, max(energies.keys()), 500)
    cont_energies = list()
    for p in p_range:
        cont_energies.append(np.sqrt(m0**2 + (2*np.pi*p/Ns)**2))
    cont_energies = np.array(cont_energies)

    x_vals = list()
    y_vals = list()
    y_errs = list()
    for px, energy in energies.items():
        x_vals.append(px)
        y_vals.append(energy.mean)
        y_errs.append(energy.sdev)

    plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color='k', capsize=2, capthick=.5, lw=.5, ls='none')
    plt.plot(p_range, cont_energies, color='k', lw=.5)

    plt.tight_layout(pad=0.80)

    plt.savefig(plot_file)
    plt.close()

def plot_multiple_dispersion(energies_dict, plot_file, Ns, a):
    fit, ax = plt.subplots()
    plt.xlabel(r"$n_z$")
    plt.ylabel(r'$E(P_z)$ [GeV]')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(energies_dict)))

    for color_i, (energy_label, energies) in enumerate(energies_dict.items()):
        m0 = energies[0].mean
        p_range = np.linspace(0, max(energies.keys()), 500)
        cont_energies = list()
        for p in p_range:
            cont_energies.append(np.sqrt(m0**2 + (2*np.pi*p/Ns)**2)*hc/a)
        cont_energies = np.array(cont_energies)

        x_vals = list()
        y_vals = list()
        y_errs = list()
        for px, energy in energies.items():
            x_vals.append(px)
            y_vals.append(energy.mean*hc/a)
            y_errs.append(energy.sdev*hc/a)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='.', color=colors[color_i], capsize=2, capthick=.5, lw=.5, ls='none', label=energy_label)
        plt.plot(p_range, cont_energies, color=colors[color_i], lw=.5)

    plt.legend()
    plt.tight_layout(pad=0.80)

    plt.savefig(plot_file)
    plt.close()


def plot_tmins(fit_results, param_name, ylabel, plot_file, yrange=None, include_w=False):
    if include_w:
        fig, [energy_ax, q_ax, w_ax] = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1, 1]})
    else:
        fig, [energy_ax, q_ax] = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]})

    if yrange is not None:
        energy_ax.set_ylim(yrange[0], yrange[1])

    fig.subplots_adjust(hspace=0.)
    plt.xlabel(r"$t_{\rm min} / a$")
    energy_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    q_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if include_w:
        w_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    energy_ax.set_ylabel(ylabel)
    q_ax.set_ylabel(r"$Q$")
    if include_w:
        w_ax.set_ylabel(r"$w$")

    num_fit_models = len(fit_results)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_fit_models))
    disp = .33 * (num_fit_models - 1) / 10
    displacements = np.linspace(-disp, disp, num_fit_models)

    for plot_i, (fit_label, fits) in enumerate(fit_results.items()):
        tmin_vals = list()
        q_vals = list()
        w_vals = list()
        fit_vals = list()
        fit_errs = list()

        for fit in fits:
            fit_param = fit.param(param_name)
            if fit_param is None:
                continue

            tmin = min(fit.independent_data)
            tmin_vals.append(tmin + displacements[plot_i])
            q_vals.append(fit.Q)
            w_vals.append(fit.w)
            fit_vals.append(fit_param.mean)
            fit_errs.append(fit_param.sdev)

        if len(fit_vals) == 0:
            continue

        energy_ax.errorbar(tmin_vals, fit_vals, yerr=fit_errs, marker='.', capsize=2, capthick=.5, elinewidth=.5, lw=.1, color=colors[plot_i], markerfacecolor='none', label=fit_label, ls='none')
        q_ax.plot(tmin_vals, q_vals, marker='.', lw=.1, color=colors[plot_i], markerfacecolor='none', ls='none')
        if include_w:
            w_ax.plot(tmin_vals, w_vals, marker='.', lw=.1, color=colors[plot_i], markerfacecolor='none', ls='none')

    energy_ax.legend()
    plt.tight_layout(pad=0.80)

    plt.savefig(plot_file)
    plt.close()



def plot_effective_energies(c2pt_datas, dt, plot_file, tmin=0, tmax=-1, tmax_rel_error=0.):
    fig, ax = plt.subplots()
    plt.xlabel(r"$t_{\rm sep}$")
    plt.ylabel(r"$a E_{\rm eff} (t_{\rm sep})$")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(c2pt_datas)))

    disps = np.linspace(-.2, .2, len(c2pt_datas))

    for color_i, (label, c2pt_data) in enumerate(c2pt_datas.items()):
        x_vals = list()
        y_vals = list()
        y_errs = list()

        eff_energy = c2pt_data.get_effective_energy(dt)
        removed_data = c2pt_data.remove_data(tmin, tmax, tmax_rel_error)

        for tsep in remove_data.get_included_data():
            tsep_dt = tsep + dt
            if tsep_dt not in remove_data.get_included_data():
                continue

            x_val = (tsep + tsep_dt)/2
            data_eff_energy = eff_energy[x_val]
            y_vals.append(data_eff_energy.mean)
            y_errs.append(data_eff_energy.sdev)
            x_vals.append(x_val + disps[color_i])

        plt.errorbar(x_vals, y_vals, yerr=y_errs, label=label, marker='.', color=colors[color_i], capsize=2, capthick=.5, lw=.5, ls='none')

    plt.legend()
    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)
    plt.close()


def boost_energy(m0, psq, Ns):
    boosted_energy = np.sqrt(m0**2 + (2*np.pi/Ns)**2*psq)
    return boosted_energy


###################################################################################################
#     C2ptMatrixData
###################################################################################################

class C2ptMatrixData:

    def __init__(self, corr_mat, norm_time=None, hermitian=False):
        self._corr_mat = corr_mat
        self._norm_time = norm_time
        self._hermitian = hermitian
        self._setup()

    def _setup(self):
        self.N = self._corr_mat.shape[0]
        tsep_min = None
        tsep_max = None

        if self._hermitian:
            for i in range(self.N):
                for j in range(i+1):
                    tseps_ij = set(self._corr_mat[i,j].get_independent_data())
                    tseps_ji = set(self._corr_mat[j,i].get_independent_data())
                    tseps = tseps_ij | tseps_ji
                    if tsep_min is None or tsep_min < min(tseps):
                        tsep_min = min(tseps)
                    if tsep_max is None or tsep_max > max(tseps):
                        tsep_max = max(tseps)

                    # make herm
                    new_data_ij_dict = dict()
                    new_data_ji_dict = dict()
                    for tsep in tseps:
                        if tsep in tseps_ij and tsep in tseps_ji:
                            new_data_ij_dict[tsep] = 0.5*(self._corr_mat[i,j](tsep) + (self._corr_mat[j,i](tsep)).conj())
                            new_data_ji_dict[tsep] = new_data_ij_dict[tsep].conj()

                        elif tsep in tseps_ij:
                            new_data_ij_dict[tsep] = self._corr_mat[i,j](tsep)
                            new_data_ji_dict[tsep] = (self._corr_mat[i,j](tsep)).conj()

                        elif tsep in tseps_ji:
                            new_data_ji_dict[tsep] = self._corr_mat[j,i](tsep)
                            new_data_ij_dict[tsep] = (self._corr_mat[j,i](tsep)).conj()

                        else:
                            raise TypeError("huh?")

                    self._corr_mat[i,j] = C2ptData(new_data_ij_dict)
                    self._corr_mat[j,i] = C2ptData(new_data_ji_dict)


        else:
            for i in range(self.N):
                for j in range(self.N):
                    tseps = self._corr_mat[i,j].get_independent_data()
                    if tsep_min is None or tsep_min < min(tseps):
                        tsep_min = min(tseps)
                    if tsep_max is None or tsep_max > max(tseps):
                        tsep_max = max(tseps)

        self.tseps = list(range(tsep_min, tsep_max+1))

        # normalize
        if self._norm_time is not None:
            normalizations = dict()
            for i in range(self.N):
                normalizations[i] = self._corr_mat[i, i](self._norm_time).samples[0]

            for i in range(self.N):
                for j in range(self.N):
                    new_data_ij_dict = dict()
                    for tsep in self.tseps:
                        new_data_ij_dict[tsep] = self._corr_mat[i, j](tsep) / np.sqrt(normalizations[i]*normalizations[j])

                    self._corr_mat[i, j] = C2ptData(new_data_ij_dict)

        # setup raw correlator matrix
        raw_corr_mat = np.empty((self.N, self.N, len(self.tseps), data_handler.get_num_samples()+1), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):
                for ti, tsep in enumerate(self.tseps):
                    raw_corr_mat[i, j, ti, :] = self._corr_mat[i, j](tsep).samples[:]

        self._raw_corr_mat = raw_corr_mat

    def get_principal_correlators(self, t0, td=None, mean=True):
        """
        Args:
            t0 (int) - required, the metric time
            td (int) - optional, the diagonalization time. If missing, diagonalize at all times
            mean (bool) - optional. If True, only gevp on the mean is done. If False, gevp on
                          all resamplings is done

        Returns:
            C2ptMatrixData - contains the rotated correlators

        TODO:
            No eigenvector pinning is implemented, so one should only use the defaults
        """
        self._t0 = t0
        self._td = td
        self._mean = mean

        if mean and td is None:
            eigvecs = np.empty((self.N, self.N, len(self.tseps)), dtype=np.complex128)
            t0_i = self.tseps.index(t0)
            for td_i, td in enumerate(self.tseps):
                _, eigvecs[:, :, td_i] = scipy.linalg.eigh(self._raw_corr_mat[:, :, td_i, 0], self._raw_corr_mat[:, :, t0_i, 0])

            eigvecs = np.repeat(eigvecs[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

        elif mean:
            t0_i = self.tseps.index(t0)
            td_i = self.tseps.index(td)
            Ct0_mean = self._raw_corr_mat[:, :, t0_i, 0]
            Ctd_mean = self._raw_corr_mat[:, :, td_i, 0]

            if not scipy.linalg.ishermitian(Ct0_mean) or not scipy.linalg.ishermitian(Ctd_mean):
                raise ValueError("Matrices must be Hermitian")

            _, eigvecs = scipy.linalg.eigh(Ctd_mean, Ct0_mean)
            eigvecs_mean = np.repeat(eigvecs[:, :, np.newaxis], len(self.tseps), axis=2)
            eigvecs = np.repeat(eigvecs_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

        elif td is None:
            eigvecs = np.empty((self.N, self.N, len(self.tseps), data_handler.get_num_samples()+1), dtype=np.complex128)
            t0_i = self.tseps.index(t0)
            for td_i, td in enumerate(self.tseps):
                for s_i in range(data_handler.get_num_samples()+1):
                    _, eigvecs[:, :, td_i, s_i] = scipy.linalg.eigh(self._raw_corr_mat[:, :, td_i, s_i], self._raw_corr_mat[:, :, t0_i, s_i])

        else:
            eigvecs = np.empty((self.N, self.N, data_handler.get_num_samples()+1), dtype=np.complex128)
            t0_i = self.tseps.index(t0)
            td_i = self.tseps.index(td)
            for s_i in range(data_handler.get_num_samples()+1):
                _, eigvecs[:, :, s_i] = scipy.linalg.eigh(self._raw_corr_mat[:, :, td_i, s_i], self._raw_corr_mat[:, :, t0_i, s_i])

            eigvecs = np.repeat(eigvecs[:, :, np.newaxis, :], len(self.tseps), axis=2)

        eigvecs = np.flip(eigvecs, axis=1)  # orders eigenvectors by energy
        self._eigen_vecs = eigvecs

        principal_corrs_raw = self.rotate_raw(eigvecs)
        principal_corrs = np.empty((self.N, self.N), dtype=object)
        for n_i in range(self.N):
            for n_j in range(self.N):
                c2pt_data_dict = dict()
                for ts_i, ts in enumerate(self.tseps):
                    c2pt_data_dict[ts] = data_handler.Data(principal_corrs_raw[n_i, n_j, ts_i, :])

                principal_corrs[n_i, n_j] = C2ptData(c2pt_data_dict)

        return C2ptMatrixData(principal_corrs)


    def rotate_raw(self, eigvecs):
        return np.einsum('ijlm,jklm,knlm->inlm', np.transpose(eigvecs.conj(), (1,0,2,3)) , self._raw_corr_mat, eigvecs)

    @property
    def eigenvectors(self):
        if hasattr(self, '_eigen_vecs'):
            return self._eigen_vecs
        return None

    def get_energy_overlaps(self, t, energy_i, amplitude):
        t0_i = self.tseps.index(self._t0)
        t_i = self.tseps.index(t)

        overlaps = np.einsum('jks,ks,s->js', self._raw_corr_mat_t0_sqrt, self.eigenvectors[:, energy_i, t_i, :], np.sqrt(amplitude.samples))
        overlap_list = list()
        overlap_sum = 0.
        for op_i in range(self.N):
            overlap = overlaps[op_i,:]*np.conj(overlaps[op_i,:])
            overlap = data_handler.Data(overlap.real)
            overlap_list.append(overlap)
            overlap_sum += overlap.samples[0]

        for op_i in range(self.N):
            overlap_list[op_i] /= overlap_sum

        return np.array(overlap_list)

    def get_overlaps(self, t, amplitudes):
        """
        Args:
            t - (int), specifies time for eigenvectors
            amplitudes - (np.array, float64)[energy_id, sample_i]

        Returns:
            overlaps - (np.array, Data)[op_id, level_id]
        """
        t0_i = self.tseps.index(self._t0)
        t_i = self.tseps.index(t)

        overlaps = np.einsum('jks,kns,ns->jns', self._raw_corr_mat[:, :, t0_i, :] , self.eigenvectors[:, :, t_i, :], np.sqrt(amplitudes))
        overlap_list = list()
        for op_i in range(self.N):
            op_overlap_list = list()
            op_overlap_sum = 0.
            for level_i in range(overlaps.shape[1]):
                overlap = overlaps[op_i, level_i, :]*np.conj(overlaps[op_i, level_i, :])
                overlap = data_handler.Data(overlap.real)
                op_overlap_list.append(overlap)
                op_overlap_sum += overlap.samples[0]

            for level_i in range(overlaps.shape[1]):
                op_overlap_list[level_i] /= op_overlap_sum

            overlap_list.append(op_overlap_list)

        return np.array(overlap_list)

    def get_principal_correlators_from_ev(self, t0, td, mean):
        """
        Args:
            t0 (int) - required, the metric time
            td (int) - optional, the diagonalization time. If missing, diagonalize at all times
            mean (bool) - optional. If True, only gevp on the mean is done. If False, gevp on
                          all resamplings is done

        Returns:
            C2ptMatrixData - contains the rotated correlators

        TODO:
            No eigenvector pinning is implemented, so one should only use the defaults
        """
        if mean and td is None:
            ...

        elif mean:
            t0_i = self.tseps.index(t0)
            td_i = self.tseps.index(td)

            Ct0_mean = self._raw_corr_mat[:, :, t0_i, 0]
            Ctd_mean = self._raw_corr_mat[:, :, td_i, 0]

            if not scipy.linalg.ishermitian(Ct0_mean) or not scipy.linalg.ishermitian(Ctd_mean):
                raise ValueError("Matrices must be Hermitian")

            Ct0_eigvals_mean, Ct0_eigvecs_mean = scipy.linalg.eigh(Ct0_mean)
            Ct0_eigvecs_tseps_mean = np.repeat(Ct0_eigvecs_mean[:, :, np.newaxis], len(self.tseps), axis=2)
            Ct0_eigvecs_tseps_samps = np.repeat(Ct0_eigvecs_tseps_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

            Ct0_1h_eigvals_mean = np.sqrt(Ct0_eigvals_mean)
            Ct0_1h_eigvals_samps = np.repeat(Ct0_1h_eigvals_mean[:, np.newaxis], data_handler.get_num_samples()+1, axis=1)

            Ct0_inv1h_eigvals_mean = 1./Ct0_1h_eigvals_mean
            Ctd_rot_mean = np.einsum('ij,jk,kl->il', Ct0_eigvecs_mean.conj().T, Ctd_mean, Ct0_eigvecs_mean)
            Gt_mean = np.einsum('i,ij,j->ij', Ct0_inv1h_eigvals_mean, Ctd_rot_mean, Ct0_inv1h_eigvals_mean)
            Gt_eigvals_mean, Gt_eigvecs_mean = scipy.linalg.eigh(Gt_mean)
            Gt_eigvecs_tseps_mean = np.repeat(Gt_eigvecs_mean[:, :, np.newaxis], len(self.tseps), axis=2)
            Gt_eigvecs_tseps_samps = np.repeat(Gt_eigvecs_tseps_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

            Rot_mean = np.einsum('ij,j,jk->ik', Ct0_eigvecs_mean, Ct0_inv1h_eigvals_mean, Gt_eigvecs_mean)
            Rot_tseps_mean = np.repeat(Rot_mean[:, :, np.newaxis], len(self.tseps), axis=2)
            Rot_tseps_samps = np.repeat(Rot_tseps_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)


        elif td is None:
            ...

        else:
            ...
        
        self._Ct0_eigenvectors = Ct0_eigvecs_tseps_samps
        self._G_eigenvectors = Gt_eigvecs_tseps_samps

        principal_corrs_raw = self.rotate_raw(Rot_tseps_samps)
        principal_corrs = np.empty((self.N, self.N), dtype=object)
        for n_i in range(self.N):
            for n_j in range(self.N):
                c2pt_data_dict = dict()
                for ts_i, ts in enumerate(self.tseps):
                    c2pt_data_dict[ts] = data_handler.Data(principal_corrs_raw[n_i, n_j, ts_i, :])

                principal_corrs[n_i, n_j] = C2ptData(c2pt_data_dict)

        return C2ptMatrixData(principal_corrs)

    @property
    def G_eigenvectors(self):
        if hasattr(self, '_G_eigenvectors'):
            return self._G_eigenvectors
        return None

    def __call__(self, row, col):
        return self._corr_mat[row,col]

