import os

import numpy as np

import matplotlib.ticker
import matplotlib.pyplot as plt
import pickle

def plot_continuum_dispersion_with_fit(plot_file, energies, Ns, fitter):
    fig, ax = plt.subplots()
    plt.xlabel(r"$(\frac{L}{2\pi})^2 P^2$")
    plt.ylabel(r'$E^2(P^2)$')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    x_vals = list()
    y_vals = list()
    y_errs = list()
    for psq, energy in energies.items():
        x_vals.append(psq)
        y_vals.append(energy.mean)
        y_errs.append(energy.sdev)

    plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color='k', capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none')

    psqs = list(energies.psqs)
    fitted_psqs = psqs # TODO - fix later

    x_vals = list(np.linspace(min(fitted_psqs), max(fitted_psqs), 500))
    mean_vals = list()
    upper_vals = list()
    lower_vals = list()
    for x in x_vals:
        y = fitter.fit_function.fit_functions[0](x, fitter.params)
        mean_vals.append(y.mean)
        upper_vals.append(y.mean+y.sdev)
        lower_vals.append(y.mean-y.sdev)

    ax.plot(x_vals, mean_vals, '-', lw=1., color='blue')
    ax.fill_between(x_vals, lower_vals, upper_vals, alpha=0.2, lw=0, color='blue')

    plt.tight_layout(pad=0.80)

    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()
