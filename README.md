# Overview

This library is meant for analyzing data generated from a lattice QCD calculation.
You will need to write your own driver scripts to suite your needs.
Typically, this involves writing a script to read the data you want to analyze and converting that data to a particular format for use with a data object within this library (e.g. the `C2ptData` type for two-point correlator data).
Then, you can perform fits and/or make plots with these objects.

# Supported Features

- Automatic data resampling
- Simultaneous fits to data
- Bayesian priors

# Installation
1. Clone the repository and change into the highest-level directory within the reposoitory, e.g.

```bash
git clone https://github.com/andrewhanlon/qcd_analysis.git
cd qcd_analysis
```

2. Install using pip

```bash
pip install .
```

# Example Usage

Suppose you have correlator data in a 2d numpy array, called `raw_corr_data`, with the first index corresponding to the config number and the second corresponding to the time separation, where `tseps` maps the tsep index to an actual tsep.
We first create a `C2ptData` object with this data, which takes a dictionary of `Data` objects.
A `Data` object corresponds to the measurements of a single observable.
When you create a `Data` object you must pass in the data as a 1d numpy array and also tell it whether these are bins (or configs) which must be resampled or whether the data is already resampled.
In the former case, you must set `bins=True` and in the latter you must set `bins=False` (the default).
For the `C2ptData` object construction, we also can optionally specify a string for the sink and source operators.
If no operator strings are provided, it is assumed the correlator is diagonal.
In this example, let's also change the resampling method to bootstrap (the default is jackknife) and bin the data by a factor of 4.

```python
from qcd_analysis.data_handling import c2pt_datatype, data_handler

rebin = 4
data_handler.set_rebin(rebin)

num_confs = raw_corr_data.shape[0]
num_samps = 1000
seed = 1234
skip = 10
data_handler.set_to_bootstrap(num_confs, rebin, num_samps, seed, skip)

c2pt_data_dict = dict()
for tsep_i, tsep in enumerate(tseps):
    c2pt_data_dict[tsep] = data_handler.Data(raw_corr_data[:,tsep_i], bins=True)

op_snk_str = "test_operator_1"
op_src_str = "test_operator_1"
c2pt_data = c2pt_datatype.C2ptData(c2pt_data_dict, op_snk_str, op_src_str)
```

Notice we set `bins=True` when creating a `Data` object, as we are assuming the data has not be resampled yet.
We are also assuming, based on the sink and source operator names, that this is a diagonal correlator.

If the data has an imaginary part that we want to ignore, we can do so with
```python
c2pt_data_real = c2pt_data.real
```
We can also make some plots of this data with the following
```python
from qcd_analysis.plotting import c2pt_plotting

plot_file = "c2pt_corr.pdf"
c2pt_plotting.plot_correlator(plot_file, c2pt_data_real, logscale=True)

plot_file = "c2pt_eff_energy.pdf"
c2pt_plotting.plot_effective_energy(plot_file, c2pt_data_real)
```
You may notice that two files get produced for each plot.
The first is the requested plotfile, and the second is a pickle file, which you can open with the `matplotlib_pickle` script in the `tools` directory of this repository.
This allows you to edit the plot without having to rerun the scripts that generated it.

Next, let's fit the real part of this data to a two-exponential fit model:
```python
from qcd_analysis.models import c2pt_models
from qcd_analysis.fitting import fitter

fit_func = c2pt_models.C2ptDirectModel(num_states=2)
guess_time = 12
fit_func.init_guesses = fit_func.get_init_guesses(c2pt_data_real, guess_time)

tmin = 10
tmax = 20
c2pt_data_real_fit = c2pt_data_real.remove_data(tmin, tmax)

the_fitter = fitter.Fitter(c2pt_data_real_fit, fit_func)

fit_success = the_fitter.do_fit()

if fit_success:
    print(the_fitter.output())
    fit_filename = 'fit_results.hdf5'
    fit_name = 'tmin10_tmax20'
    the_fitter.write_to_hdf5(fit_filename, fit_name, ['E0'])
```
This fit used a range of time separations from 10 to 20 and then wrote the results of the ground state `E0` to an hdf5 file called `fit_results.hdf5`.
We can now make a plot of the effective energy with the fit overlaid with
```python

plot_file = "c2pt_eff_energy_with_fit.pdf"
c2pt_plotting.plot_effective_energy_with_fit(plot_file, c2pt_data_real, the_fitter, {'E0': r'$E_0$'})
```
The last argument passed to this plot routine will print the fit parameters given as keys, where the values are the latex representation.

One can also construct correlator matrices and solve for the generalized eigenvalues.
Supppose you have a 4d numpy array `raw_matrix_corr_data`, with the first index being the configuration, the second and third being the correlator matrix indices, and the fourth being the time separation.
You can construct `C2ptMatrixData` object, which gets constructed from a 2d array of `C2ptData` objects corresponding to the indices of the correlator matrix.
Further, let's assume a list of of strings, corresponding to the operators used for the correlator are given in `operators`.
Let's also assume, as above, that we have a list `tseps` containing the actual tseps used.
```python
import numpy as np

num_ops = raw_matrix_corr_data.shape[1]
num_tseps = raw_matrix_corr_data.shape[3]
corr_mat = np.empty((num_ops, num_ops), dtype=object)
for n in range(num_ops):
    for m in range(num_ops):
        c2pt_data_dict = dict()
        for tsep_i, tsep in enumerate(tseps):
            c2pt_data_dict[tsep] = data_handler.Data(raw_matrix_corr_data[:,n,m,tsep_i], bins=True)

        corr_mat[n,m] = c2pt_datatype.C2ptData(c2pt_data_dict, operators[n], operators[m])

corr_mat = c2pt_datatype.C2ptMatrixData(c2pt_data_dict, operators)

t0 = 8
td = 16
rotated_corr_mat = corr_mat.get_principal_correlators(t0, td)
```
After performing the gevp, we can also compute the overlap factors, defined as the overlap between the eigenstates and the states created by the operators in the basis.
This requires a fit to the principal correlators first, in order to obtain their amplitudes, which can be done with
```python
amplitudes = rotated_corr_mat.get_amplitudes(td, tmax)
```
Then we can get the overlaps, write them to disk, and obtain a map from each energy level to an operator(s):
```python
filename = 'overlaps.txt'
corr_mat.compute_overlaps(td, amplitudes, filename)
energy_to_op_map = corr_mat.get_energy_to_operator_map()
```
Note that we are using the original `corr_mat` object for these last two function calls.

# Technical Details

## Data Types
Several Datatypes (e.g. C2ptData) relevant for lattice analysis are defined.

## Fitter
Additionally, the fitter takes a data type and a fit function

## TODO
- Many of the Data Types can be made more abstract.
Make General Data Types (scalar, 1d array, 2d array, Nd array), i.e. MultiNd objects.
The underlying data structure can just be a numpy array.
Then, every data type is just an extension of these MultiNd objects.
If the keys for the data type are not indices starting from zero, then a major feature of each extension DataType is to implement a mapping of the keys to indices in the underlying numpy array.
- Related to the previous item, the numpy arrays are arrays of `Data` objects.
So, maybe it's the `Data` objects that need to be extended?
