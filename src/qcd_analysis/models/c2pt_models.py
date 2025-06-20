import sys

import numpy as np

from qcd_analysis.data_handling import data_handler



def C2ptModel(model_name):
    model_name_tokens = model_name.split('_')

    if model_name_tokens[0] == "ratio":
        return C2ptRatioModel()
    elif model_name_tokens[0] == "direct":
        num_states, _ = model_name_tokens[1].split('-')
        return C2ptDirectModel(int(num_states))
    else:
        print(f"Unknown C2pt model '{model_name}'")


class C2ptDirectModel(data_handler.FitFunction):

    def __init__(self, num_states):
        if num_states <= 0:
            print("Num states must be greater than zero")
            sys.exit()

        self.num_states = num_states

    @property
    def fit_name(self):
        return f"c2pt_{self.num_states}-exp"

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
        eff_amp = np.exp(eff_energy*tsep) * c2pt_data(tsep).real

        init_guesses_dict['E0'] = eff_energy.mean
        init_guesses_dict['A0'] = eff_amp.mean

        for param in self.params:
            if param == 'E0' or param == 'A0':
                continue

            init_guesses_dict[param] = .1

        return init_guesses_dict
                

class C2ptRatioModel(data_handler.FitFunction):

    @property
    def fit_name(self):
        return "c2pt_ratio"

    @property
    def params(self):
        return ['dE0', 'dA0']


    def function(self, x, p):
        f = p['dA0']*np.exp(-p['dE0']*x)
        return f


    def get_init_guesses(self, c2pt_data, tsep):
        init_guesses_dict = dict()

        eff_energy = c2pt_data.get_effective_energy(1)[tsep + 0.5]
        eff_amp = np.exp(eff_energy*tsep) * c2pt_data(tsep).real

        init_guesses_dict['dE0'] = eff_energy.mean
        init_guesses_dict['dA0'] = eff_amp.mean

        return init_guesses_dict


class MresModel(data_handler.FitFunction):

    def __init__(self):
        pass

    @property
    def fit_name(self):
        return "MresConstant"

    @property
    def params(self):
        return ['A']

    def function(self, x, p):
        return p['A']



class C2ptThermalModel(data_handler.FitFunction):

    def __init__(self, T, num_states=1, constant=True):
        self.T = T
        self.num_states = num_states
        self.constant = constant

    @property
    def fit_name(self):
        if self.constant:
            return f"c2pt_thermal_{self.num_states}-exp_constant"
        else:
            return f"c2pt_thermal_{self.num_states}-exp"

    @property
    def params(self):
        _params = list()
        for n in range(self.num_states):
            _params.append(f"A{n}")
            _params.append(f"E{n}")
            if self.constant:
                _params.append(f"B{n}")

            for m in range(n+1, self.num_states):
                _params.append(f"C{n},{m}")

        return _params

    def function(self, x, p):
        f = 0.
        for n in range(self.num_states):
            #f += p[f'A{n}']*(np.exp(-p[f'E{n}']*x) + np.exp(-p[f'E{n}']*(self.T - x)))
            f += p[f'A{n}']*np.cosh(p[f'E{n}']*(x - self.T/2.))
            if self.constant:
                f += p[f'B{n}']

            for m in range(n+1, self.num_states):
                f += p[f'C{n},{m}']*np.cosh((p[f'E{m}'] - p[f'E{n}'])*(x - self.T/2.))

        return f

    def get_init_guesses(self, c2pt_data, tsep):
        init_guesses_dict = dict()

        eff_energy = c2pt_data.get_effective_energy(1)[tsep + 0.5]
        init_guesses_dict['E0'] = eff_energy.mean

        eff_amp = np.exp(eff_energy*tsep) * c2pt_data(tsep).real
        init_guesses_dict['A0'] = eff_amp.mean * np.exp(-eff_energy.mean*self.T/2.)

        if self.constant:
            init_guesses_dict['B-1'] = .1

        for param in self.params:
            if param in init_guesses_dict:
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

