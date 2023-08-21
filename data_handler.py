import enum
import abc

import random
import numpy as np
import gvar as gv

import matplotlib.pyplot as plt
#plt.rc('font', family='serif', serif=['Times'], size=55)
#plt.rc('text', usetex=True)
plt.style.use('/home/ahanlon/code/qcd_scripts/analysis/plots.mplstyle')


class SamplingMode(enum.Enum):
  JACKKNIFE = 1
  BOOTSTRAP = 2

SAMPLING_MODE = SamplingMode.JACKKNIFE
NUM_JACK = None
BOOTSTRAPS = None

def set_to_jackknife():
  global SAMPLING_MODE, BOOTSTRAPS
  SAMPLING_MODE = SamplingMode.JACKKNIFE
  BOOTSTRAPS = None

def set_to_bootstrap(num_confs, rebin, num_samples, seed, skip=0):
  global SAMPLING_MODE, NUM_SAMPLES, BOOTSTRAPS
  SAMPLING_MODE = SamplingMode.BOOTSTRAP
  num_bins = num_confs // rebin
  BOOTSTRAPS = np.zeros((num_samples, num_bins), dtype=np.int32)
  random.seed(seed)
  for samp_i in range(num_samples):
    for _ in range(num_bins):
      for skip_i in range(skip):
        random.randrange(num_bins)
      bin_i = random.randrange(num_bins)
      BOOTSTRAPS[samp_i,bin_i] += 1

  NUM_SAMPLES = BOOTSTRAPS.shape[0]

def get_sampling_mode():
  return SAMPLING_MODE

def get_num_samples():
  if get_sampling_mode() == SamplingMode.BOOTSTRAP:
    return BOOTSTRAPS.shape[0]
  else:
    return NUM_JACK

REBIN = 1
def set_rebin(rebin):
  global REBIN
  REBIN = rebin

def get_rebin():
  return REBIN

class ComplexArg(enum.Enum):
  REAL = 0
  IMAG = 1


class Data:

  def __init__(self, input_data, bins=False):
    if isinstance(input_data, gv.GVar):
      self._bins = None
      self._create_samples_from_gvar(input_data, bins)
    elif bins:
      self._bins = input_data
      self._create_samples()
    else:
      self._bins = None
      # check input samples
      if len(input_data.shape) != 1:
        raise TypeError("Samplings must have one index")

      if get_sampling_mode() == SamplingMode.BOOTSTRAP and input_data.shape[0] != get_num_samples()+1:
        raise TypeError("Number of samples does not match current bootstrap settings")

      self._samples = input_data

  @property
  def bins(self):
    return self._bins

  @property
  def samples(self):
    return self._samples

  def _create_samples_from_gvar(self, gvar_data, num_jack):
    if SAMPLING_MODE == SamplingMode.JACKKNIFE:
      self._samples = np.zeros(num_jack+1)
      self._samples[0] = gvar_data.mean
      self._samples[1:] = np.random.normal(gvar_data.mean, gvar_data.sdev / (num_jack - 1)**0.5, num_jack)
    else:
      num_samples = get_num_samples()
      self._samples = np.zeros(num_samples+1)
      self._samples[0] = gvar_data.mean
      self._samples[1:] = np.random.normal(gvar_data.mean, gvar_data.sdev, num_samples)

  def _create_samples(self):
    num_bins = len(self._bins)
    rebin = get_rebin()
    num_rebin_bins = num_bins // rebin
    rebin_bins = np.zeros(num_rebin_bins, dtype=self._bins.dtype)
    for rebin_ci in range(num_rebin_bins):
      for ci in range(rebin_ci*rebin, rebin_ci*rebin + rebin):
        rebin_bins[rebin_ci] += self._bins[ci]

      rebin_bins[rebin_ci] /= rebin

    if SAMPLING_MODE == SamplingMode.JACKKNIFE:
      global NUM_JACK
      NUM_JACK = num_rebin_bins
      ensemble_ave = 0.
      for bin_value in rebin_bins:
        ensemble_ave += bin_value

      self._samples = np.zeros(num_rebin_bins+1, dtype=rebin_bins.dtype)
      self._samples[0] = ensemble_ave / num_rebin_bins

      for sample_i, bin_value in enumerate(rebin_bins, 1):
        self._samples[sample_i] = (ensemble_ave - bin_value) / (num_rebin_bins - 1)

    else:
      if num_bins != BOOTSTRAPS.shape[1]:
        print(num_bins)
        print(BOOTSTRAPS.shape)
        raise TypeError("Number of bins does not match number of configs")

      num_samples = get_num_samples()

      ensemble_ave = 0.
      for bin_value in rebin_bins:
        ensemble_ave += bin_value

      self._samples = np.zeros(num_samples+1, dtype=rebin_bins.dtype)
      self._samples[0] = ensemble_ave / num_rebin_bins
      for sample_i in range(num_samples):
        sample_ave = 0.
        for rebin_i in range(num_rebin_bins):
          sample_ave += BOOTSTRAPS[sample_i,rebin_i] * rebin_bins[rebin_i]

        self._samples[sample_i+1] = sample_ave / num_rebin_bins


  @property
  def num_samples(self):
    return len(self._samples) - 1

  @property
  def mean(self):
    return self.ensemble_average()

  @property
  def sdev(self):
    return self.error()

  def ensemble_average(self):
    return self.samples[0]

  def sample_average(self):
    ave = 0.
    for sample in self.samples[1:]:
      ave += sample

    return ave / self.num_samples

  def error(self):
    if SAMPLING_MODE == SamplingMode.JACKKNIFE:
      return (self.num_samples - 1)**0.5 * np.std(self.samples[1:])

    else:
      sorted_samples = np.sort(self.samples[1:])
      percentile_index = int(round(self.num_samples * 0.16))
      error = (self.samples[0] - sorted_samples[percentile_index], sorted_samples[-percentile_index] - self.samples[0])
      return 0.5*(error[0] + error[1])

  def invert_samples(self):
    temp_samples = np.zeros(self._samples.shape, dtype=self._samples.dtype)
    temp_samples[0] = self._samples[0]
    for i in range(self.num_samples):
      temp_samples[i+1] = self._samples[0] - (self._samples[i+1] - self._samples[0])

    self._samples = temp_samples


  def plot_samples(self, plot_file, num_bins=20):
    fig, ax = plt.subplots()
    ax.hist(self.samples[1:], bins=num_bins)

    plt.tight_layout(pad=0.80)
    plt.savefig(plot_file)
    plt.close()


  def gvar(self):
    return gv.gvar(self.mean, self.sdev)
      
  def exp(self):
    return Data(np.exp(self.samples))

  def log(self):
    return Data(np.log(self.samples))

  def sqrt(self):
    return Data(np.sqrt(self.samples))

  def conj(self):
    return Data(np.conj(self.samples))
  
  def __add__(self, other):
    if isinstance(other, self.__class__):
      return Data(self.samples + other.samples)
    return Data(self.samples + other)

  def __radd__(self, other):
    if isinstance(other, self.__class__):
      return Data(other.samples + self.samples)
    return Data(other + self.samples)

  def __sub__(self, other):
    if isinstance(other, self.__class__):
      return Data(self.samples - other.samples)
    return Data(self.samples - other)

  def __rsub__(self, other):
    if isinstance(other, self.__class__):
      return Data(other.samples - self.samples)
    return Data(other - self.samples)

  def __mul__(self, other):
    if isinstance(other, self.__class__):
      return Data(self.samples * other.samples)
    return Data(self.samples * other)

  def __rmul__(self, other):
    if isinstance(other, self.__class__):
      return Data(other.samples * self.samples)
    return Data(other * self.samples)

  def __truediv__(self, other):
    if isinstance(other, self.__class__):
      return Data(self.samples / other.samples)
    return Data(self.samples / other)

  def __rtruediv__(self, other):
    if isinstance(other, self.__class__):
      return Data(other.samples / self.samples)
    return Data(other / self.samples)

  def __pow__(self, other):
    return Data(pow(self.samples, other))

  def __rpow__(self, other):
    return Data(pow(other, self.samples))

  def __neg__(self):
    return Data(-self.samples)

  @property
  def real(self):
    return Data(self.samples.real)

  @property
  def imag(self):
    return Data(self.samples.imag)

  def __str__(self):
    return str(self.gvar())

  def __call__(self, samp_i):
    return self.samples[samp_i]


class DataType(metaclass=abc.ABCMeta):

  _cov = None
  _cov_re = None
  _cov_im = None

  @abc.abstractmethod
  def get_samples(self):
    """ gets samples of the data

    Returns: Must be of shape (num_samples+1, num_observables)
    """
    pass

  @property
  @abc.abstractmethod
  def num_samples(self):
    pass
  
  @abc.abstractmethod
  def get_independent_data(self):
    pass

  @abc.abstractmethod
  def get_organized_independent_data(self):
    pass

  def get_covariance(self, correlated=True):
    samples = self.get_samples()

    if get_sampling_mode() == SamplingMode.JACKKNIFE:
      cov_factor = 1. - 1./self.num_samples
    else:
      cov_factor = 1./(self.num_samples - 1)

    diffs = samples[1:,:] - samples[0,:]

    if correlated:
      return cov_factor * np.tensordot(diffs.conj(), diffs, axes=(0,0))
    else:
      return cov_factor * np.diag(np.einsum('ij,ji->i', diffs.conj().T, diffs))


