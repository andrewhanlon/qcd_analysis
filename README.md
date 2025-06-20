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
>>> git clone https://github.com/andrewhanlon/qcd_analysis.git
>>> cd qcd_analysis
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
For the `C2ptData` object construction, we also need to specify a string for the sink and source operators.

```python
from qcd_analysis.data_handling import c2pt_datatype, data_handler

c2pt_data_dict = dict()
for tsep_i, tsep in enumerate(tseps):
    c2pt_data_dict[tsep] = data_handler.Data(raw_corr_data[:,tsep_i], bins=True)

op_snk_str = "test_operator_1"
op_src_str = "test_operator_1"
c2pt_data = c2pt_datatype.C2ptData(c2pt_data_dict, op_snk_str, op_src_str)
```

Notice we set `bins=True` when creating a `Data` object, as we are assuming the data has not be resampled yet.

Next, we are also assuming, based on the sink and source operator names, that this is a diagonal correlator.
Let's next fit the real part of this data to a two-exponential fit model:

```python
from qcd_analysis.models import c2pt_models
from qcd_analysis.fitting import fitter

c2pt_data_real = c2pt_data.real

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

# Technical Details

## Data Types
Several Datatypes (e.g. C2ptData) relevant for lattice analysis are defined.

## Fitter
Additionally, the fitter takes a data type and a fit function

