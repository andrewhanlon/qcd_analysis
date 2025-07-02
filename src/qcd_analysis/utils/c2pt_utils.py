from dataclasses import dataclass

import numpy as np

@dataclass
class NonInteractingLevel:
    particle_names: list
    particle_psqs:list

def boost_energy(m0, psq, Ns):
    boosted_energy = np.sqrt(m0**2 + (2*np.pi/Ns)**2*psq)
    return boosted_energy

def boost_to_cm(Elab, psq, Ns):
    Ecm_sq = Elab**2 - (2.*np.pi/Ns)**2 * psq
    return np.sqrt(Ecm_sq)

def get_non_int_level_cont(masses, psqs, Ns):
    non_int_energy = 0.

    for mass, psq in zip(masses, psqs):
        psqfactor = psq* (2.*np.pi/Ns)**2
        E_sq = mass**2 + (2.*np.pi/Ns)**2 * psq

        non_int_energy += np.sqrt(E_sq)

    return non_int_energy

def get_non_int_level_latt(sh_energies):
    non_int_energy = 0.

    for sh_energy in sh_energies:
        non_int_energy += sh_energy

    return non_int_energy
