import enum

import numpy as np

from qcd_analysis.data_handling import data_handler
from qcd_analysis.utils import c2pt_utils

class SingleHadronEnergyMode(enum.Enum):
    ENERGYSQUARED = 1
    ENERGY = 2

class SingleHadronEnergyData(data_handler.DataType):

    def __init__(self, energies):
        """
        Args:
            energies (dict) - {psq: data_handler.Data}: The energies at each psq
        """

        self._energies = energies
        self._mode = SingleHadronEnergyMode.ENERGYSQUARED

    def set_energy_mode(self, mode):
        self._mode = mode

    @property
    def data_name(self):
        return f"SingleHadronEnergies"

    @property
    def samples(self):
        if self._mode is SingleHadronEnergyMode.ENERGYSQUARED:
            return np.stack([(d**2).samples for d in self._energies.values()], axis=-1)
        else:
            return np.stack([d.samples for d in self._energies.values()], axis=-1)

    @property
    def num_samples(self):
        return list(self._energies.values())[0].num_samples

    @property
    def independent_variables_values(self):
        return np.array(list(self._energies.keys()))

    @property
    def num_data(self):
        return len(self._energies.keys())

    @property
    def data(self):
        return self._energies

    def items(self):
        return self._energies.items()

    def __call__(self, psq):
        return self._energies[psq]

    def __getitem__(self, psq):
        return self._energies[psq]

    def __contains__(self, psq):
        return psq in self._energies

    @property
    def psqs(self):
        return list(self._energies.keys())

    def remove_psqs(self, psqs_to_remove):
        new_data = dict()
        for psq, data in sorted(self._energies.items()):
            if psq in psqs_to_remove:
                continue

            new_data[psq] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

class MultiHadronEnergyMode(enum.Enum):
    ELAB = 1
    ECM = 2
    ELABSHIFT = 3
    PSQ = 4

class MultiHadronEnergyData(data_handler.DataType):

    def __init__(self, energies, single_hadron_energies, Ns):
        """
        Args:
            energies (dict) - {(Psq, irrep): list[(data_handler.Data, non_int_level)]}
            single_hadron_energies (dict) - {species: SingleHadronEnergyData}
            Ns (int): 
        """

        self._energies = energies
        self._single_hadron_energies = single_hadron_energies
        self._Ns = Ns
        self._mode = MultiHadronEnergyMode.ECM

    def set_energy_mode(self, mode):
        self._mode = mode

    @property
    def data_name(self):
        return f"MultiHadronEnergies"

    @property
    def samples(self):
        '''
        if self._mode is SingleHadronEnergyMode.ENERGYSQUARED:
            return np.stack([(d**2).samples for d in self._energies.values()], axis=-1)
        else:
            return np.stack([d.samples for d in self._energies.values()], axis=-1)
        '''
        return None

    @property
    def num_samples(self):
        return list(self._energies.values())[0].num_samples

    @property
    def independent_variables_values(self):
        return np.array(list(self._energies.keys()))

    @property
    def num_data(self):
        return len(self._energies.keys())

    @property
    def data(self):
        return self._energies

    def items(self):
        return self._energies.items()
