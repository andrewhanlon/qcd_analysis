import os

import numpy as np

import matplotlib.ticker
import matplotlib.pyplot as plt
import pickle

hc = 6.62607015e-34 / 1.602176634e-19 / (2*np.pi) * 299792458 * 1e6 # exact: new SI

def plot_correlator(plot_file, corr, logscale=True):

    fig, ax = plt.subplots()
    plt.xlabel(r"$t_{\rm sep}$")
    plt.ylabel(r"$C (t_{\rm sep})$")

    if logscale:
        plt.yscale('log')

    x_vals = list()
    y_vals = list()
    y_errs = list()

    for tsep, data in corr._data.items():
        y_vals.append(data.mean)
        y_errs.append(data.sdev)
        x_vals.append(tsep)

    plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color='k', capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none')

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()


def plot_effective_energy(plot_file, corr, dt=1, cosh=False, y_label=r"$a E_{\rm eff} (t_{\rm sep})$"):
    fig, ax = plt.subplots()
    plt.xlabel(r"$t_{\rm sep}$")
    plt.ylabel(y_label)

    x_vals = list()
    y_vals = list()
    y_errs = list()

    eff_energy = corr.get_effective_energy(dt, cosh)

    for tsep in corr.tseps:
        tsep_dt = tsep + dt
        if tsep_dt not in corr.tseps:
            continue

        x_val = (tsep + tsep_dt)/2
        if x_val not in eff_energy:
            continue

        data_eff_energy = eff_energy[x_val]
        y_vals.append(data_eff_energy.mean)
        y_errs.append(data_eff_energy.sdev)
        x_vals.append(x_val)

    plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color='k', capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none')

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()


def plot_effective_energy_with_fit(plot_file, corr, fitter, print_params={}, dt=1, cosh=False, y_label=r"$a E_{\rm eff} (t_{\rm sep})$"):
    fig, ax = plt.subplots()
    plt.xlabel(r"$t_{\rm sep}$")
    plt.ylabel(y_label)

    x_vals = list()
    y_vals = list()
    y_errs = list()

    eff_energy = corr.get_effective_energy(dt, cosh)

    for tsep in corr.tseps:
        tsep_dt = tsep + dt
        if tsep_dt not in corr.tseps:
            continue

        x_val = (tsep + tsep_dt)/2
        if x_val not in eff_energy:
            continue

        data_eff_energy = eff_energy[x_val]
        y_vals.append(data_eff_energy.mean)
        y_errs.append(data_eff_energy.sdev)
        x_vals.append(x_val)

    plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color='k', capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none')

    fitted_tseps = fitter.input_data.tseps

    x_vals = list(np.linspace(min(fitted_tseps), max(fitted_tseps), 500))
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
        plt.text(0.05, 0.07 + (num_params - height)*.075, rf"${param_text} = {fitter.params[param_name]!s}$", transform=ax.transAxes)

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()



def plot_dispersion(plot_file, energies, Ns):
    fig, ax = plt.subplots()
    plt.xlabel(r"$\bigg(\frac{L}{2\pi})\bigg)^2 P^2$")
    plt.ylabel(r'$E(P^2)$')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

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

    plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color='k', capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none')
    plt.plot(p_range, cont_energies, color='k', lw=.5)

    plt.tight_layout(pad=0.80)

    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()


def plot_tmin(plot_file, energies, chosen_energy_key=None, y_label=r"$a E_{\rm fit}$", include_quality=False):
    """
    Args: energies - dict {label: {tmin: fit_result}}
    """

    if include_quality:
        fig, [energy_ax, q_ax] = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        q_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        q_ax.set_ylabel(r"$Q$")
    else:
        fig, energy_ax = plt.subplots()

    energy_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    plt.xlabel(r"$t_{\rm min}$")
    plt.ylabel(y_label)

    colors = plt.cm.tab10(list(range(len(energies))))

    for color_i, (label, energies_dict) in enumerate(energies.items()):

        x_vals = list()
        y_vals = list()
        y_errs = list()
        q_vals = list()

        for tmin, energy_result in energies_dict.items():

            x_vals.append(tmin)
            y_vals.append(energy_result.mean)
            y_errs.append(energy_result.sdev)
            q_vals.append(energy_result.Q)

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color=colors[color_i], capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none', label=label)

        if include_quality:
            q_ax.plot(x_vals, q_vals, marker='o', ms=2, mew=.5, lw=.05, color=colors[color_i], markerfacecolor='none', ls='none')

    if chosen_energy_key is not None:
        chosen_energy_result = energies[chosen_energy_key[0]][chosen_energy_key[1]]

        tmin, tmax = energy_ax.get_xlim()
        x = np.linspace(tmin, tmax, 100)
        upper = chosen_energy_result.mean + chosen_energy_result.sdev
        lower = chosen_energy_result.mean - chosen_energy_result.sdev
        plt.hlines(y=chosen_energy_result.mean, xmin=tmin, xmax=tmax, colors='k', linestyles='--', lw=1.)
        energy_ax.fill_between(x, lower, upper, alpha=0.2, color='tab:cyan', edgecolor='none')

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()


def plot_spectrum(plot_file, energies, non_interacting_energies, thresholds, y_label=r"$a E_{\rm cm}$", label_rotation=0):
    """
    Args:
        energies - dict {irrep_latex: [energy_result]}
        non_interacting_energies - dict {irrep_latex: energy_result}
        thresholds - dict {label_latex: energy_result}
    """

    fig, ax = plt.subplots()

    plt.xlabel(r"$t_{\rm min}$")
    plt.ylabel(y_label)

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    labels = list()
    label_locations = list()
    for x_val, (irrep_latex, energy_results) in enumerate(energies.items()):
        labels.append(irrep_latex)
        label_locations.append(x_val)

        non_interacting_results = non_interacting_energies[irrep_latex] if irrep_latex in non_interacting_energies else list()

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
        plt.text(xmax, threshold_energy, threshold_latex, ha='right', va='center')


    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()


def overlap_plots(plotfiles, overlaps, labels):
    """
    Args:
        plotfiles - list: a plotfile for each operator
        overlaps - (np.array, Data)[op_id, level_id]
        labels - list: a label for each plot
    """

    # check sizes
    num_ops = overlaps.size[0]
    if (num_ops != len(plotfiles)) or (num_ops != len(labels)):
        print("In function 'overlap_plots', object sizes not consistent")
        sys.exit()


    for op_id, (plotfile, label) in enumerate(zip(plotfiles, labels)):
        fig, ax = plt.subplots()
        plt.xlabel(r"Energy Level")
        plt.ylabel(r'$|Z^{(n)}|^2$')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        
        x_vals = list()
        y_vals = list()
        y_errs = list()
        for n in range(overlaps.size[1]):
            x_vals.append(n)
            y_vals.append(overlaps[op_id, n].mean)
            y_errs.append(overlaps[op_id, n].sdev)

        plt.bar(x_vals, y_vals)
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='none', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        anchored_text = matplotlib.offsetbox.AnchoredText(label, loc=2)
        ax.add_artist(anchored_text)

        plt.tight_layout(pad=0.80)

        plt.savefig(plot_file)

        pickle_file, _ = os.path.splitext(plot_file)
        with open(f"{pickle_file}.pkl", "wb") as fh:
            pickle.dump(fig, fh)

        plt.close()
