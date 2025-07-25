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

    def __init__(self, num_single_hadron_states, particle=1):
        self.num_single_hadron_states = num_single_hadron_states
        self.particle = particle

    @property
    def fit_name(self):
        return f"single-hadron-conspiracy_{self.num_single_hadron_states}-states-single-hadron{self.particle}"

    @property
    def params(self):
        _params = [f'E({self.particle})0']
        for i in range(1, self.num_single_hadron_states):
            _params.append(f'DE({self.particle}){i},{i-1}')

        _params.append(f'A({self.particle})0')
        for i in range(1, self.num_single_hadron_states):
            _params.append(f'r({self.particle}){i}')

        return _params


    def function(self, x, p):
        f = 1.
        for i in range(1, self.num_single_hadron_states):
            fi = p[f'r({self.particle}){i}']
            for j in range(i):
                fi *= np.exp(-p[f'DE({self.particle}){j+1},{j}']*x)
            f += fi
        f *= p[f'A({self.particle})0']*np.exp(-p[f'E({self.particle})0']*x)
        return f


    def get_init_guesses(self, c2pt_data, tsep):
        init_guesses_dict = dict()

        eff_energy = c2pt_data.get_effective_energy(1)[tsep + 0.5]
        init_guesses_dict[f'E({self.particle})0'] = eff_energy.mean

        eff_amp = np.exp(eff_energy*tsep) * c2pt_data(tsep).real
        init_guesses_dict[f'A({self.particle})0'] = eff_amp.mean

        for param in self.params:
            if param == f'E({self.particle})0' or param == f'A({self.particle})0':
                continue

            init_guesses_dict[param] = .1

        return init_guesses_dict

    def get_priors(self, c2pt_data, tsep, gap):
        priors_dict = dict()

        eff_energy = c2pt_data.get_effective_energy(1)[tsep + 0.5]
        priors_dict[f'E({self.particle})0'] = data_handler.Prior((eff_energy.mean, eff_energy.mean*.1))

        eff_amp = np.exp(eff_energy*tsep) * c2pt_data(tsep).real
        priors_dict[f'A({self.particle})0'] = data_handler.Prior((eff_amp.mean, eff_amp.mean*.1))

        for i in range(1, self.num_single_hadron_states):
            priors_dict[f'DE({self.particle}){i},{i-1}'] = data_handler.Prior((gap, gap/2.))

        for i in range(1, self.num_single_hadron_states):
            priors_dict[f'r({self.particle}){i}'] = data_handler.Prior((1., .25))

        return priors_dict


class C2ptTwoHadronConspiracyModel(data_handler.FitFunction):

    def __init__(self, num_single_hadron_states, particles):
        if particles[0] == particles[1] and num_single_hadron_states[0] != num_single_hadron_states[1]:
            print("If particles are equal, then they must have same number of states included")
            sys.exit()

        self.num_single_hadron_states = num_single_hadron_states
        self.particles = particles

    @property
    def fit_name(self):
        '''
        if self.degenerate:
            return f"two-hadron-conspiracy_{self.num_single_hadron_states}-exp-deg-single-hadron"
        else:
            return f"two-hadron-conspiracy_{self.num_single_hadron_states}-exp-single-hadron"
        '''
        return f"two-hadron-conspiracy"


    @property
    def params(self):
        part1 = self.particles[0]
        part2 = self.particles[1]

        _params = list()
        if part1 == part2:
            num_states = self.num_single_hadron_states[0]
            for n in range(num_states):
                for m in range(n+1):
                    _params.append(f'dE{n},{m}')

            for n in range(num_states):
                for m in range(n+1):
                    _params.append(f'rp{n},{m}')


            _params.append(f'E({part1})0')
            _params.append(f'A({part1})0')

            for n in range(1, num_states):
                _params.append(f'r({part1}){n}')

            for n in range(1, num_states):
                _params.append(f'DE({part1}){n},{n-1}')

        else:
            num_states1 = self.num_single_hadron_states[0]
            num_states2 = self.num_single_hadron_states[0]
            for n in range(num_states1):
                for m in range(num_states2):
                    _params.append(f'dE{n},{m}')

            for n in range(num_states1):
                for m in range(num_states2):
                    _params.append(f'rp{n},{m}')


            _params.append(f'E({part1})0')
            _params.append(f'E({part2})0')
            _params.append(f'A({part1})0')
            _params.append(f'A({part2})0')

            for n in range(1, num_states1):
                _params.append(f'r({part1}){n}')
            for m in range(1, num_states2):
                _params.append(f'r({part2}){m}')

            for n in range(1, num_states1):
                _params.append(f'DE({part1}){n},{n-1}')
            for m in range(1, num_states2):
                _params.append(f'DE({part2}){m},{m-1}')

        return _params


    def function_prefactor(self, n, m, x, p):
        part1 = self.particles[0]
        part2 = self.particles[1]
        if n == 0:
            r1 = 1.
        else:
            r1 = p[f'r({part1}){n}']

        if m == 0:
            r2 = 1.
        else:
            r2 = p[f'r({part2}){m}']

        if part1 == part2 and n < m:
            fsub = p[f'rp{m},{n}']*np.exp(-p[f'dE{m},{n}']*x)*r1*r2
        else:
            fsub = p[f'rp{n},{m}']*np.exp(-p[f'dE{n},{m}']*x)*r1*r2

        return fsub

    def function(self, x, p):
        part1 = self.particles[0]
        part2 = self.particles[1]
        f = 0.
        for n in range(self.num_single_hadron_states[0]):
            for m in range(self.num_single_hadron_states[1]):
                fsub = self.function_prefactor(n, m, x, p)
                for i in range(n):
                    fsub *= np.exp(-p[f'DE({part1}){i+1},{i}']*x)
                for j in range(m):
                    fsub *= np.exp(-p[f'DE({part2}){j+1},{j}']*x)

                f += fsub

        f *= p[f'A({part1})0']*p[f'A({part2})0']*np.exp(-(p[f'E({part1})0'] + p[f'E({part2})0'])*x)
        return f


    def get_init_guesses(self, c2pt_ratio_data, tsep):
        init_guesses_dict = dict()

        eff_energy = c2pt_ratio_data.get_effective_energy(1)[tsep + 0.5]
        init_guesses_dict[f'dE{0},{0}'] = eff_energy.mean

        eff_amp = np.exp(eff_energy*tsep) * c2pt_ratio_data(tsep).real
        init_guesses_dict[f'rp{0},{0}'] = eff_amp.mean

        part1 = self.particles[0]
        part2 = self.particles[1]

        if part1 == part2:
            num_states = self.num_single_hadron_states[0]
            for n in range(num_states):
                for m in range(n+1):
                    if n == m == 0:
                        continue
                    init_guesses_dict[f'dE{n},{m}'] = 0.

            for n in range(num_states):
                for m in range(n+1):
                    if n == m == 0:
                        continue
                    init_guesses_dict[f'rp{n},{m}'] = .1

        else:
            num_states1 = self.num_single_hadron_states[0]
            num_states2 = self.num_single_hadron_states[0]
            for n in range(num_states1):
                for m in range(num_states2):
                    if n == m == 0:
                        continue
                    init_guesses_dict[f'dE{n},{m}'] = 0.

            for n in range(num_states1):
                for m in range(num_states2):
                    if n == m == 0:
                        continue
                    init_guesses_dict[f'rp{n},{m}'] = .1

        return init_guesses_dict

    def get_priors(self, c2pt_ratio_data, tsep):
        priors_dict = dict()

        eff_energy = c2pt_ratio_data.get_effective_energy(1)[tsep + 0.5]
        priors_dict[f'dE{0},{0}'] = data_handler.Prior((eff_energy.mean, eff_energy.mean))

        eff_amp = np.exp(eff_energy*tsep) * c2pt_ratio_data(tsep).real
        priors_dict[f'rp{0},{0}'] = data_handler.Prior((eff_amp.mean, eff_amp.mean*.1))

        part1 = self.particles[0]
        part2 = self.particles[1]

        if part1 == part2:
            num_states = self.num_single_hadron_states[0]
            for n in range(num_states):
                for m in range(n+1):
                    if n == m == 0:
                        continue
                    priors_dict[f'dE{n},{m}'] = data_handler.Prior((0., eff_energy.mean))

            for n in range(num_states):
                for m in range(n+1):
                    if n == m == 0:
                        continue
                    priors_dict[f'rp{n},{m}'] = data_handler.Prior((1., .25))

        else:
            num_states1 = self.num_single_hadron_states[0]
            num_states2 = self.num_single_hadron_states[0]
            for n in range(num_states1):
                for m in range(num_states2):
                    if n == m == 0:
                        continue
                    priors_dict[f'dE{n},{m}'] = data_handler.Prior((0., eff_energy.mean))

            for n in range(num_states1):
                for m in range(num_states2):
                    if n == m == 0:
                        continue
                    priors_dict[f'rp{n},{m}'] = data_handler.Prior((1., .25))

        return priors_dict
