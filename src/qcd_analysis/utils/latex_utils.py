import os
import pylatex

from qcd_analysis.plotting import c2pt_plotting

def create_spectrum_doc(pdf_dir, title, ):
    ...

def create_raw_data_doc(pdf_dir, title, diagonal_correlators, off_diagonal_correlators={}):
    """
    Args:
        title (str) - title for document
        diagonal_correlators (dict) - {str: list[correlators]}: Sets of diagonal correlators.
            the key is the name for the list of correlators
        off_diagonal_correlators (dict) - {str: list[correlators]}: Sets of off-diagonal correlators.
            the key is the name for the list of correlators
    """ 

    doc = create_doc(title)
    fig_dir = os.path.join(pdf_dir, "plots")

    for channel_name, corr_list in diagonal_correlators.items():
        channel_dir = os.path.join(fig_dir, channel_name.replace(' ', '_'))
        os.makedirs(channel_dir, exist_ok=True)
        with doc.create(pylatex.Section(channel_name)):
            for corr in corr_list:
                with doc.create(pylatex.Subsection(corr.data_name)):
                    add_correlator(doc, corr, pdf_dir, channel_dir) 

            if channel_name in off_diagonal_correlators:
                with doc.create(pylatex.Subsection("Off-diagonal correlators")):
                    for corr in off_diagonal_correlators[channel_name]:
                        with doc.create(pylatex.Subsubsection(corr.data_name)):
                            add_correlator(doc, corr, pdf_dir, channel_dir) 

    pdf_filename = os.path.join(pdf_dir, title)
    compile_pdf(doc, pdf_filename)


def create_gevp_data_doc(pdf_dir, title, operators, rot_diagonal_correlators, rot_off_diagonal_correlators={}, rot_ratios={}, overlaps={}):
    """
    Args:
        title (str) - title for document
        operators (dict) - {str: list[operators]}
        rot_diagonal_correlators (dict) - {str: list[correlators]}: Sets of diagonal correlators.
            the key is the name for the list of correlators
        rot_off_diagonal_correlators (dict) - {str: list[correlators]}: Sets of off-diagonal correlators.
            the key is the name for the list of correlators
        rot_ratios (dict) - {str: str: [single_correlator_list]}
        overlaps (dict) - {str: 
    """ 

    doc = create_doc(title)
    fig_dir = os.path.join(pdf_dir, "plots")

    for channel_name, operator_list in operators.items():
        channel_dir = os.path.join(fit_dir, channel_name.replace(' ', '_'))
        os.makedirs(channel_dir, exist_ok=True)
        with doc.create(pylatex.Section(channel_name)):




    add_raw_data(doc, pdf_dir, fig_dir, diagonal_correlators, off_diagonal_correlators, False)

    pdf_filename = os.path.join(pdf_dir, title)
    compile_pdf(doc, pdf_filename)


###################################################################################################
#     Helper Utilities
###################################################################################################


def compile_pdf(doc, filename, compiler=None):
    doc.generate_tex(filename)
    doc.generate_pdf(filename, clean=False, clean_tex=False, compiler=compiler, compiler_args=['-synctex=1'])
    doc.generate_pdf(filename, clean=True, clean_tex=False, compiler=compiler, compiler_args=['-synctex=1'])
    

def create_doc(title):
    doc = pylatex.Document(geometry_options={'margin': '1.5cm'})
    doc.packages.append(pylatex.Package('hyperref'))
    doc.packages.append(pylatex.Package('amssymb'))
    doc.packages.append(pylatex.Package('amsmath'))
    doc.packages.append(pylatex.Package('float'))
    doc.packages.append(pylatex.Package('caption', {'labelformat': 'empty', 'justification': 'centering'}))
    doc.packages.append(pylatex.NoEscape(r"\usepackage{longtable}[=v4.13]"))

    doc.preamble.append(pylatex.Command('title', title))
    doc.preamble.append(pylatex.Command('date', ''))

    doc.append(pylatex.NoEscape(r"\maketitle"))
    doc.append(pylatex.NoEscape(r"\tableofcontents"))
    doc.append(pylatex.NoEscape(r"\newpage"))
    doc.append(pylatex.NoEscape(r"\captionsetup[subfigure]{labelformat=empty}"))

    return doc

def add_correlator(doc, correlator, latex_dir, plot_dir, dt=1, cosh=False):

    if correlator.snk_operator == correlator.src_operator:
        left_pdf_file = os.path.join(plot_dir, f"corr_{correlator.data_name}.pdf")
        right_pdf_file = os.path.join(plot_dir, f"eff_energy_{correlator.data_name}.pdf")

        c2pt_plotting.plot_correlator(left_pdf_file, correlator.real, True)
        c2pt_plotting.plot_effective_energy(right_pdf_file, correlator.real, dt, cosh)

        left_estimates = correlator.real
        right_estimates = correlator.real.get_effective_energy(dt, cosh)

    else:
        left_pdf_file = os.path.join(plot_dir, f"corr_{correlator.data_name}_real.pdf")
        right_pdf_file = os.path.join(plot_dir, f"corr_{correlator.data_name}_imag.pdf")

        c2pt_plotting.plot_correlator(left_pdf_file, correlator.real, False)
        c2pt_plotting.plot_correlator(right_pdf_file, correlator.imag, False)

        left_estimates = correlator.real
        right_estimates = correlator.imag

    with doc.create(pylatex.Figure(position='H')):
        with doc.create(pylatex.SubFigure(position='b', width=pylatex.NoEscape(r'0.5\linewidth'))) as left_fig:
            add_image(left_fig, latex_dir, left_pdf_file, width="1.0")
        with doc.create(pylatex.SubFigure(position='b', width=pylatex.NoEscape(r'0.5\linewidth'))) as right_fig:
            add_image(right_fig, latex_dir, right_pdf_file, width="1.0")


    if correlator.snk_operator == correlator.src_operator:
        header_row = [
            pylatex.NoEscape(r"$t$"),
            pylatex.NoEscape(r"$C(t)$"),
            pylatex.NoEscape(r"$\delta C(t)$"), 
        ]

        header_row.extend([
            pylatex.NoEscape(rf"$a_t E_{{\rm eff}} (t + {dt}/2)$"),
            pylatex.NoEscape(rf"$\delta a_t E_{{\rm eff}} (t + {dt}/2)$"),
        ])

    else:
        header_row = [
            pylatex.NoEscape(r"$t$"),
            pylatex.NoEscape(r"$Re C(t)$"),
            pylatex.NoEscape(r"$\delta Re C(t)$"), 
            pylatex.NoEscape(r"$Im C(t)$"),
            pylatex.NoEscape(r"$\delta Im C(t)$"),
        ]


    with doc.create(pylatex.Center()) as centered:
        with centered.create(pylatex.LongTabu("X[c] X[2,c] X[2,c] X[2,c] X[2,c]", 
                             to=r"\linewidth")) as data_table:
            data_table.add_row(header_row, mapper=[pylatex.utils.bold])
            data_table.add_hline()
            data_table.end_table_header()
            for t_sep in sorted(left_estimates.tseps):
                left_est_nice = str(left_estimates[t_sep])
                left_rel_error = round(left_estimates[t_sep].sdev/abs(left_estimates[t_sep].mean), 4)
                t_sep_right = t_sep
                if correlator.snk_operator == correlator.src_operator:
                    t_sep_right = t_sep + 0.5*dt

                if t_sep_right in right_estimates:
                    right_est_nice = str(right_estimates[t_sep_right])
                    right_rel_error = round(right_estimates[t_sep_right].sdev/abs(right_estimates[t_sep_right].mean), 4)
                else:
                    right_est_nice = ""
                    right_rel_error = ""

                row = [int(t_sep), left_est_nice, left_rel_error, right_est_nice, right_rel_error]
                data_table.add_row(row)

    doc.append(pylatex.NoEscape(r"\newpage"))


def add_image(figure, latex_dir, pdf_file, width="1.0", caption="", view=True):
    relative_pdf_file = os.path.relpath(pdf_file, latex_dir)
    relative_pickle_file = os.path.splitext(relative_pdf_file)[0] + ".pkl"

    width = rf"{width}\linewidth"
    placement = r"\centering"
    if os.path.isfile(pdf_file):
        if caption and view:
            caption = rf"{caption} \newline \href{{run:{relative_pickle_file}}}{{view}}"
        elif view:
            caption = rf"\href{{run:{relative_pickle_file}}}{{view}}"

    figure.add_image(pdf_file, width=pylatex.NoEscape(width), placement=pylatex.NoEscape(placement))
    if caption:
        figure.add_caption(pylatex.NoEscape(caption))

