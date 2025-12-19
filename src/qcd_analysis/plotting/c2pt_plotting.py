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

def plot_effective_energies(plot_file, corrs, labels, dt=1, cosh=False, y_label=r"$a E_{\rm eff} (t_{\rm sep})$"):
    fig, ax = plt.subplots()
    plt.xlabel(r"$t_{\rm sep}$")
    plt.ylabel(y_label)

    avail_colors = plt.cm.tab10(list(range(10)))

    for i, (corr, label) in enumerate(zip(corrs, labels)):
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

        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color=avail_colors[i], capsize=2, capthick=.5, lw=.5, ls='none', markerfacecolor='none', label=label)

    ax.legend()
    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)

    pickle_file, _ = os.path.splitext(plot_file)
    with open(f"{pickle_file}.pkl", "wb") as fh:
        pickle.dump(fig, fh)

    plt.close()


def plot_effective_energy_with_fit(plot_file, corr, fitter, print_params={}, dt=1, cosh=False, y_label=r"$a E_{\rm eff} (t_{\rm sep})$", corr_index=0):
    """
    Args:
        plot_file (str): the plot file to use
        corr (c2pt_datatype.C2ptData): The data to use for the effective energy data points
        fitter (fitting.Fitter): The fitter object (after a successful fit).
        print_params (dict): A dictionary with the keys being the params to print (as known by the fitter) and the values are the latex you want
                             used when priting the variable name
        dt (int): the dt to use for the effective energy
        cosh (bool): whether to use a cosh function for the effective energy
        y_lable (str): the label to use for the y axis
        corr_index (int): If you are doing simultaneous fits to multiple correlators, this index specifies which one to plot.
    """

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

    fitted_tseps = fitter.input_data.independent_variables_values[corr_index]

    x_vals = list(np.linspace(min(fitted_tseps), max(fitted_tseps), 500))
    mean_vals = list()
    upper_vals = list()
    lower_vals = list()
    for x in x_vals:
        y = fitter.fit_function.fit_functions[corr_index](x - dt/2., fitter.params_list)
        y_dt = fitter.fit_function.fit_functions[corr_index](x + dt/2., fitter.params_list)
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


def plot_tmin(plot_file, results, height_ratios, chosen_results={}):
    """
    Args:
        plot_file - str: where the file is to be saved
        results - dict {yaxis_label: {legend_label: tmin: {Data or float}}}
        height_ratios - list: relative heights of the individual axes
        chosen_results - {yaxis_label: {legend_label: tmin}}
    """

    num_subplots = len(results)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True, height_ratios=height_ratios)

    plt.xlabel(r"$t_{\rm min}$")

    avail_colors = plt.cm.tab10(list(range(10)))

    colors = dict()
    color_i = 0
    legend_elements = list()

    for ax, (ax_label, legends_dict) in zip(axes, results.items()):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel(ax_label)

        disps = np.linspace(-.35, .35, len(legends_dict))

        ax_chosen_results = {} if ax_label not in chosen_results else chosen_results[ax_label]

        for legend_label_i, (legend_label, tmins_dict) in enumerate(legends_dict.items()):
            if legend_label in colors:
                color = colors[legend_label]
            else:
                color = avail_colors[color_i]
                color_i += 1
                colors[legend_label] = color
                legend_elements.append(plt.plot([], label=legend_label, marker='o')[0])

            x_vals = list()
            y_vals = list()
            y_errs = list()

            for tmin, result in tmins_dict.items():
                x_vals.append(tmin+disps[legend_label_i])
                try:
                    y_errs.append(result.sdev)
                    y_vals.append(result.mean)
                except AttributeError:
                    y_vals.append(result)

            if len(y_errs):
                ax.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', color=color, capsize=2, capthick=.5, linewidth=.5, linestyle='none', markerfacecolor='none')
            else:
                ax.plot(x_vals, y_vals, marker='o', color=color, markersize=5, markeredgewidth=.5, linestyle='none', markerfacecolor='none')

            if legend_label in ax_chosen_results:
                tmin = ax_chosen_results[legend_label]
                ax.axvline(x=tmin+disps[legend_label_i], color='k', linestyle='--')

                if len(y_errs):
                    ax_xmin, ax_xmax = ax.get_xlim()
                    x = np.linspace(ax_xmin, ax_xmax, 100)

                    upper = tmins_dict[tmin].mean + tmins_dict[tmin].sdev
                    lower = tmins_dict[tmin].mean - tmins_dict[tmin].sdev
                    ax.fill_between(x, lower, upper, alpha=0.1, color='tab:cyan', edgecolor='none')

                    ax.set_xlim(ax_xmin, ax_xmax)

                else:
                    ax.axhline(y=tmins_cit[tmin].mean, color='k', linestyle='--')

    axes[0].legend(handles=legend_elements)

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
    num_ops = overlaps.shape[0]
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
        for n in range(overlaps.shape[1]):
            x_vals.append(n)
            y_vals.append(overlaps[op_id, n].mean)
            y_errs.append(overlaps[op_id, n].sdev)

        plt.bar(x_vals, y_vals)
        plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='none', color='k', capsize=2, capthick=.5, lw=.5, ls='none')

        anchored_text = matplotlib.offsetbox.AnchoredText(label, loc=2)
        ax.add_artist(anchored_text)

        plt.tight_layout(pad=0.80)

        plt.savefig(plotfile)

        pickle_file, _ = os.path.splitext(plotfile)
        with open(f"{pickle_file}.pkl", "wb") as fh:
            pickle.dump(fig, fh)

        plt.close()
