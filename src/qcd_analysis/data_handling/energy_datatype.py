import sys

from dataclasses import dataclass

import h5py
import itertools
import enum
import copy

import numpy as np

from qcd_analysis.data_handling import data_handler
from qcd_analysis.utils import c2pt_utils

'''
class SingleHadronEnergyMode(enum.Enum):
    ENERGYSQUARED = 1
    ENERGY = 2

class SingleHadronEnergyData(data_handler.DataType):

    def __init__(self, energies, input_mode=SingleHadronEnergyMode.ENERGY):
        """
        Args:
            energies (dict) - {psq: data_handler.Data}: The energies at each psq
        """

        self._input_mode = input_mode
        self._input_energies = energies

        self._current_mode = input_mode
        self._current_energies = copy.deepcopy(energies)

    @property
    def input_mode(self):
        return self._input_mode

    @property
    def current_mode(self):
        return self._current_mode

    def set_current_mode(self, new_current_mode):
        if self.input_mode == new_current_mode:
            self._current_energies = copy.deepcopy(self._input_energies)

        elif new_current_mode is SingleHadronEnergyMode.ENERGYSQUARED:
            for psq, energy in self._input_energies.items():
                self._current_energies[psq] = energy*energy

        elif new_current_mode is SingleHadronEnergyMode.ENERGY:
            for psq, energy_squared in self._input_energies.items():
                self._current_energies[psq] = np.sqrt(energy_squared)

        else:
            print("Error in 'SingleHadronEnergyData.set_current_mode'")
            sys.exit()

        self._current_mode = new_current_mode

    @property
    def data_name(self):
        return f"SingleHadronEnergies"

    @property
    def samples(self):
        return np.stack([d.samples for d in self._current_energies.values()], axis=-1)

    @property
    def num_samples(self):
        return list(self._current_energies.values())[0].num_samples

    @property
    def independent_variables_values(self):
        return np.array(list(self._current_energies.keys()))

    @property
    def num_data(self):
        return len(self._current_energies.keys())

    @property
    def data(self):
        return self._current_energies

    def items(self):
        return self._current_energies.items()

    def __call__(self, psq):
        return self._current_energies[psq]

    def __getitem__(self, psq):
        return self._current_energies[psq]

    def __contains__(self, psq):
        return psq in self._current_energies

    @property
    def psqs(self):
        return list(self._current_energies.keys())

    def remove_psqs(self, psqs_to_remove):
        new_data = dict()
        for psq, data in sorted(self._current_energies.items()):
            if psq in psqs_to_remove:
                continue

            new_data[psq] = data

        return SingleHadronEnergyData(new_data, self.current_mode)
'''

'''
class Particle:
    def __init__(self, name, momentum):
        """
        Args:
            name (str) - the name of the particle, e.g. 'Nucleon' or 'N'
            momentum (int or tuple(3)) - either a momentum squared or an vector of integers for the momentum
        """

        self._name = name
        self._momentum = momentum


class FreeLevel:
    def __init__(self, *particles):
        self._particles = particles
'''


@dataclass(eq=True, frozen=True)
class LittleGroupIrrep:
    momentum: tuple
    irrep: str

    @property
    def Psq(self):
        return self.momentum[0]**2 + self.momentum[1]**2 + self.momentum[2]**2

    def __mul__(self, other):
        return self.Psq*other

    def __rmul__(self, other):
        return self.Psq*other

    @property
    def mom_str(self):
        pref = sorted(self.momentum)
        return f"Pref{pref[0]}{pref[1]}{pref[2]}"

    def __str__(self):
        return f"Pref{self.momentum[0]}{self.momentum[1]}{self.momentum[2]}_{self.irrep}"

"""
LITTLE_GROUP_IRREPS = [
        'Pref000/A1g'
        'Pref000/A1u'
        'Pref000/A2g'
        'Pref000/A2u'
        'Pref000/Eg'
        'Pref000/Eu'
        'Pref000/T1g'
        'Pref000/T1u'
        'Pref000/T2g'
        'Pref000/T2u'
        'Pref000/G1g'
        'Pref000/G1u'
        'Pref000/G2g'
        'Pref000/G2u'
        'Pref000/Hg'
        'Pref000/Hu'
        'Pref00n/A1'
        'Pref00n/A2'
        'Pref00n/B1'
        'Pref00n/B2'
        'Pref00n/E'
        'Pref00n/G1'
        'Pref00n/G2'
        'Pref0nn/A1'
        'Pref0nn/A2'
        'Pref0nn/B1'
        'Pref0nn/B2'
        'Pref0nn/G'
        'Prefnnn/A1'
        'Prefnnn/A2'
        'Prefnnn/E'
        'Prefnnn/F1'
        'Prefnnn/F2'
        'Prefnnn/G'
        'Pref0nm/A1'
        'Pref0nm/A2'
        'Pref0nm/F1'
        'Pref0nm/F2'
        'Prefnnm/A1'
        'Prefnnm/A2'
        'Prefnnm/F1'
        'Prefnnm/F2'
]
"""

class EnergyMode(enum.Enum):
    ELAB = 1
    ECM = 2
    ELAB_SHIFT = 3
    PSQ = 4
    ELAB_SQUARED = 5

energy_label = {
        EnergyMode.ELAB_SHIFT: "dE_lab",
        EnergyMode.ELAB: "E_lab",
}

class EnergyData(data_handler.DataType):

    def __init__(self, energies, input_mode):
        """
        Args:
            energies (dict) - {LittleGroupIrrep: list[data_handler.Data, ..]}
            input_mode (EnergyMode): the type of energies input
        """

        self._input_mode = input_mode
        self._input_energies = energies

        self._current_mode = input_mode
        self._current_energies = copy.deepcopy(energies)

    @property
    def input_mode(self):
        return self._input_mode

    @property
    def current_mode(self):
        return self._current_mode

    def set_current_mode(self, new_mode, **kwargs):
        '''
            Things needed:
                - Ns
                - non_interacting
                - masses for channels
        '''
        if self.current_mode is EnergyMode.ELAB:
            if new_mode is EnergyMode.ECM:
                for lg_irrep in self._input_energies.keys():
                    self._current_energies[lg_irrep] = list()
                    for elab in self._input_energies[lg_irrep]:
                        ecm = c2pt_utils.boost_to_cm(elab, lg_irrep.Psq, kwargs['Ns'])
                        self._current_energies[lg_irrep].append(ecm)

            elif new_mode is EnergyMode.ELAB_SHIFT:
                for lg_irrep in self._input_energies.keys():
                    self._current_energies[lg_irrep] = list()
                    for elab, non_int in zip(self._input_energies[lg_irrep], non_interacting[lg_irrep]):
                        elab_shift = elab - non_int
                        self._current_energies[lg_irrep].append(elab_shift)

            elif new_mode is EnergyMode.PSQ:
                '''
                for lg_irrep in self._input_energies.keys():
                    self._current_energies[lg_irrep] = list()
                    for elab in self._input_energies[lg_irrep]:
                        ecm = c2pt_utils.boost_to_cm(elab, lg_irrep.Psq, Ns)
                        psq_tuple = list()
                '''

                return NotImplementedError

            elif new_mode is EnergyMode.ELAB_SQUARED:
                for lg_irrep in self._input_energies.keys():
                    self._current_energies[lg_irrep] = list()
                    for elab in self._input_energies[lg_irrep]:
                        self._current_energies[lg_irrep].append(elab**2)
            else:
                return NotImplementedError

        elif self.current_mode is EnergyMode.ELAB_SHIFT:
            if new_mode is EnergyMode.ECM:
                free_energies = kwargs['free_energies']
                for lg_irrep in self._input_energies.keys():
                    self._current_energies[lg_irrep] = list()
                    for elab_shift, free_energy in zip(self._input_energies[lg_irrep], free_energies[lg_irrep]):
                        elab = elab_shift + free_energy
                        ecm = c2pt_utils.boost_to_cm(elab, lg_irrep.Psq, kwargs['Ns'])
                        self._current_energies[lg_irrep].append(ecm)
            else:
                return NotImplementedError


        else:
            print("Error in 'MultiHadronEnergyData.set_current_mode'")
            sys.exit()


        self._current_mode = new_mode

    @property
    def data_name(self):
        return f"Energies"

    @property
    def samples(self):
        return np.stack([d.samples for d in list(itertools.chain.from_iterable(self._current_energies.values()))], axis=-1)

    @property
    def num_samples(self):
        return list(self._current_energies.values())[0][0].num_samples

    @property
    def independent_variables_values(self):
        ind_vars = list()
        for k, v in self._current_energies.items():
            ind_vars.extend(len(v)*[k])

        return np.array(ind_vars)

    @property
    def num_data(self):
        return len(self.independent_variables_values)

    @property
    def data(self):
        return self._current_energies

    def items(self):
        return self._current_energies.items()

    def __call__(self, label):
        return self._current_energies[label]

    def __getitem__(self, label):
        return self._current_energies[label]

    def __contains__(self, label):
        return label in self._current_energies

    def write_to_hdf5(self, filename, group_name, free_levels=dict(), overwrite=False):
        fh = h5py.File(filename, 'a')

        if group_name in fh and not overwrite:
            print(f"Group '{group_name}' already exists in file '{filename}'")
            sys.exit()

        fh_group = fh.create_group(group_name)
        for lg, energies in self.items():
            energy_group = fh_group.create_group(f"{lg.mom_str}/{lg.irrep}")
            for level_id, energy in enumerate(energies):
                level_group = energy_group.create_group(f"level_{level_id}")
                level_group.create_dataset(energy_label[self._current_mode], data=energy.samples)

            if lg in free_levels:
                for level_id, free_level in enumerate(free_levels[lg]):
                    level_group = energy_group[f"level_{level_id}"]
                    level_group.attrs['free_level'] = [f'{s}({str(p)})' for s,p in free_level]

        fh.close()


