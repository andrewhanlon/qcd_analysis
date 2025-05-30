import numpy as np

def boost_energy(m0, psq, Ns):
    boosted_energy = np.sqrt(m0**2 + (2*np.pi/Ns)**2*psq)
    return boosted_energy

def boost_to_cm(Elab, psq, Ns):
    Ecm_sq = Elab**2 - (2.*np.pi/Ns)**2 * psq
    return np.sqrt(Ecm_sq)

def get_absolute_energy_cont(E_shift, masses, non_int_level, Ns):
    non_int_energy = 0.

    for mass, psq in zip(masses, non_int_level):
        psqfactor = psq* (2.*np.pi/Ns)**2
        E_sq = mass**2 + (2.*np.pi/Ns)**2 * psq

        non_int_energy += np.sqrt(E_sq)

    return E_shift + non_int_energy

def get_absolute_energy_latt(E_shift, energies):
    non_int_energy = 0.

    for energy in energies:
        non_int_energy += energy

    return E_shift + non_int_energy

