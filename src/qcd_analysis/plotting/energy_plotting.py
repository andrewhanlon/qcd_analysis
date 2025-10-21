import os

import numpy as np

import matplotlib.ticker
import matplotlib.pyplot as plt
import pickle

def plot_continuum_dispersion_with_fit(plot_file, energies, Ns, fitter):
    fig, ax = plt.subplots()
    plt.xlabel(r"$(\frac{L}{2\pi})^2 P^2$")
    plt.ylabel(r'$a^2 E^2(P^2)$')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    x_vals = list()
    y_vals = list()
    y_errs = list()
    for lg, energy_list in energies.items():
        if len(energy_list) > 1:
            return NotImplementedError
        energy = energy_list[0]
        x_vals.append(lg.Psq)
        y_vals.append(energy.mean)
        y_errs.append(energy.sdev)

    plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color='k', capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none')

    x_vals = list(np.linspace(min(x_vals), max(x_vals), 500))
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


irrep_latex = {
        'A1g': r'$A_{1g}$',
        'A1u': r'$A_{1u}$',
        'A2g': r'$A_{2g}$',
        'A2u': r'$A_{2u}$',
        'Eg': r'$E_{g}$',
        'Eu': r'$E_{u}$',
        'T1g': r'$T_{1g}$',
        'T1u': r'$T_{1u}$',
        'T2g': r'$T_{2g}$',
        'T2u': r'$T_{2u}$',
        'A1': r'$A_1$',
        'A2': r'$A_2$',
        'B1': r'$B_1$',
        'B2': r'$B_2$',
        'E': r'$E$',
}

def plot_spectrum(plot_file, energies, non_interacting_energies, thresholds, y_label=r"$a E_{\rm cm}$", label_rotation=0, width=1.6):
    """
    Args:
        energies - dict {irrep_latex: [energy_result]}
        non_interacting_energies - dict {irrep_latex: energy_result}
        thresholds - dict {label_latex: energy_result}
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(width*6.4, 4.8)
    fig.set_dpi(width*100.)

    plt.ylabel(y_label)

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    labels = list()
    label_locations = list()
    for x_val, (lg, energy_results) in enumerate(energies.items()):
        labels.append(f"{irrep_latex[lg.irrep]}({lg.Psq})")
        label_locations.append(x_val)

        non_interacting_results = non_interacting_energies[lg] if lg in non_interacting_energies else list()

        for non_int_energy in non_interacting_results:
            plt.hlines(y=non_int_energy.mean, xmin=x_val-.3, xmax=x_val+.3, colors='k', linestyles='--', lw=1.)
            upper = non_int_energy.mean + non_int_energy.sdev
            lower = non_int_energy.mean - non_int_energy.sdev
            x = np.linspace(x_val-.3, x_val+.3, 100)
            ax.fill_between(x, lower, upper, alpha=0.2, color='tab:grey', edgecolor='none')

        
        x_vals = list()
        y_vals = list()
        y_errs = list()
        for energy_result in energy_results:
            x_vals.append(x_val)
            y_vals.append(energy_result.mean)
            y_errs.append(energy_result.sdev)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color='k', capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none')

    plt.xticks(label_locations, labels, rotation=label_rotation)

    # plot thresholds
    for threshold_latex, threshold_energy in thresholds.items():
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        plt.hlines(y=threshold_energy, xmin=xmin, xmax=xmax, colors='tab:grey', linestyles=':', lw=1.)
        plt.text(xmax + .25, threshold_energy, threshold_latex, ha='left', va='center')
        ax.set_xlim(xmin, xmax)


    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()

