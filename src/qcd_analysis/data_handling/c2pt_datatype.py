import os
import sys

import numpy as np
import scipy.linalg
import warnings
import copy

from qcd_analysis.data_handling import data_handler
from qcd_analysis.fitting import fitter
from qcd_analysis.models import c2pt_models

###################################################################################################
#     C2ptData
###################################################################################################

class C2ptData(data_handler.DataType):

    def __init__(self, data, snk_operator="op", src_operator="op"):
        """
        Args:
            data (dict) - {tsep: data_handler.Data}: The correlator data.
            snk_operator (str): optional, default='op'
            src_operator (str): optional, default='op'

        Notes:
            While providing source and sink operators is optional, if you do not provide them,
            then the defaults are used which implies this is a diagonal correlator.
            If a correlator is diagonal, this can affect how latex routines place the correlator
            in PDFs.
        """

        self.ratio = False
        self._data = data
        self._snk_op = snk_operator
        self._src_op = src_operator
    
    @property
    def snk_operator(self):
        return self._snk_op

    @property
    def src_operator(self):
        return self._src_op

    @property
    def corr_name(self):
        if self.diagonal:
            return self.snk_operator
        else:
            return f"{self.snk_operator} - {self.src_operator}"

    @property
    def diagonal(self):
        return self.snk_operator == self.src_operator

    @property
    def data_name(self):
        return f"c2pt"

    @property
    def samples(self):
        return np.stack([d.samples for d in self._data.values()], axis=-1)

    @property
    def num_samples(self):
        return list(self._data.values())[0].num_samples

    @property
    def independent_variables_values(self):
        return np.array(list(self._data.keys()))

    @property
    def num_data(self):
        return len(self._data.keys())


    @property
    def data(self):
        return self._data

    def items(self):
        return self.data.items()

    def __call__(self, tsep):
        return self[tsep]

    def __getitem__(self, tsep):
        return self.data[tsep]

    def __contains__(self, tsep):
        return tsep in self.data

    def log(self):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            new_data[tsep] = data.log()

        return C2ptData(new_data)

    def sqrt(self):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            new_data[tsep] = data.sqrt()

        return C2ptData(new_data)

    def conj(self):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            new_data[tsep] = data.conj()

        return C2ptData(new_data, snk_operator=self.src_operator, src_operator=self.snk_operator)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.corr_name != other.corr_name:
                print("mismatch in operators when adding correlators")
                sys.exit()

            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = data + other(tsep)

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = data + other

        return C2ptData(new_data, snk_operator=self.snk_operator, src_operator=self.src_operator)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = data - other(tsep)

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = data - other

        return C2ptData(new_data)

    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = other(tsep) - data

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = other - data

        return C2ptData(new_data)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = data * other(tsep)

            return C2ptData(new_data)

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = data * other

            return C2ptData(new_data, snk_operator=self.snk_operator, src_operator=self.src_operator)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = data / other(tsep)

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = data / other

        return C2ptData(new_data)

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = other(tsep) / data

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = other / data

        return C2ptData(new_data)

    def __pow__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = pow(data, other(tsep))

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = pow(data, other)

        return C2ptData(new_data)

    def __rpow__(self, other):
        if isinstance(other, self.__class__):
            new_data = dict()
            for tsep, data in sorted(self.items()):
                if tsep in other:
                    new_data[tsep] = pow(other(tsep), data)

        else:
            new_data = dict()
            for tsep, data in sorted(self.items()):
                new_data[tsep] = pow(other, data)

        return C2ptData(new_data)

    def __neg__(self):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            if tsep in other:
                new_data[tsep] = -data
        
        return C2ptData(new_data)

    @property
    def real(self):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            new_data[tsep] = data.real

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    @property
    def imag(self):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            new_data[tsep] = data.imag

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def conj(self):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            new_data[tsep] = data.conj()

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.tseps != other.tseps:
                return False

            for tsep in self.tseps:
                if not self(tsep) == other(tsep):
                    return False

            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def tseps(self):
        return list(self.data.keys())

    def remove_data(self, tmin=0, tmax=-1, tmax_rel_error=0.):
        new_data = dict()
        for tsep, data in sorted(self.items()):
            if tsep < tmin:
                continue

            if tmax > 0 and tsep > tmax:
                break

            corr_val = data.mean
            corr_err = data.sdev
            if (abs(corr_val) / corr_err) < tmax_rel_error:
                break

            new_data[tsep] = data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data

        return new_obj

    def get_correlator_estimates(self):
        estimates = dict()
        for tsep, corr_tsep in self.items():
            estimates[tsep] = str(corr_tsep)

        return estimates

    def get_eff_energy_estimates(self, dt=1, cosh=False):
        estimates = dict()
        for tsep, eff_energy_tsep in self.get_effective_energy(dt, cosh).items():
            estimates[tsep] = str(eff_energy_tsep)

        return estimates

    def print_correlator(self):
        for tsep, tsep_data in self.items():
            print(f"C({tsep}) = {tsep_data}")

    def print_effective_energy(self, dt=1, cosh=False):
        for tsep, tsep_data in self.get_effective_energy(dt, cosh).items():
            print(f"E_eff({tsep}) = {tsep_data}")


    def get_effective_energy(self, dt=1, cosh=False):
        eff_energy = dict()
        for tsep in self.tseps:
            tsep_dt = tsep + dt
            if tsep_dt not in self.tseps:
                continue

            x_val = (tsep + tsep_dt)/2

            data = self[tsep]
            data_dt = self[tsep_dt]

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    if cosh:
                        data_eff_energy = (-1./dt)*np.log(data_dt/data)
                    else:
                        data_eff_energy = (-1./dt)*np.log(data_dt/data)
                except Warning as e:
                    #print(f"Warning for tsep={tsep}: {e}")
                    continue

            eff_energy[x_val] = data_eff_energy

        return eff_energy

    def get_shifted_c2pt(self, shift):
        new_data = dict()
        for tsep, tsep_data in self.items():
            tsep_prime = tsep - shift
            if tsep_prime < 0:
                continue
            new_data[tsep_prime] = tsep_data

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data
        new_obj._shift = shift

        return new_obj

    def get_prony_matrix(self, N, delta=1):
        mat = np.empty((N,N,), dtype=object)
        for i in range(N):
            for j in range(N):
                mat[i,j] = self.get_shifted_c2pt(delta*(i+j))

        return C2ptMatrixData(mat)


    def get_ratio_correlator(self, corr_list):
        new_data = dict()
        for tsep, tsep_data in self.items():
            denom_tsep_data = list()
            all_tseps = True
            for corr in corr_list:
                if tsep not in corr._data:
                    all_tseps = False
                    break

                denom_tsep_data.append(corr._data[tsep])

            if not all_tseps:
                continue

            new_data[tsep] = tsep_data / np.prod(denom_tsep_data)

        new_obj = copy.deepcopy(self)
        new_obj._data = new_data
        new_obj.ratio = True

        return new_obj

    def get_amplitude(self, tmin, tmax=-1, tmax_rel_error=0.0, num_exps=1, non_int_corrs=[]):
        fit_data = self.remove_data(tmin, tmax, tmax_rel_error)
        if non_int_corrs:
            fit_data = fit_data.get_ratio_correlator(non_int_corrs)
            non_int_amps = list()
            for non_int_corr in non_int_corrs:
                non_int_amps.append(non_int_corr.get_amplitude([], tmin, tmax, tmax_rel_error, num_exps))

        fit_func = c2pt_models.C2ptDirectModel(num_exps)
        fit_func.init_guesses = fit_func.get_init_guesses(fit_data, tmin)

        the_fitter = fitter.Fitter(fit_data, fit_func)
        if not the_fitter.do_fit():
            print(f"Fit failed for amplitude")
            sys.exit()
        
        if non_int_corrs:
            return the_fitter.params['A0'] * np.prod(non_int_amps)
        else:
            return the_fitter.params['A0']


###################################################################################################
#     C2ptMatrixData
###################################################################################################

class C2ptMatrixData:

    def __init__(self, corr_mat, operators=list(), norm_time=None, make_hermitian=True):
        """
        Args:
            corr_mat (ndarray[snk_op_i,src_op_i], dtype is C2ptData)

        TODO: It's possible that the operator list is not consistent with the operators
              used for the C2ptData objects that construct this class. Is this an issue?
              Currently, I think this is a user issue.
        """
        self._input_corr_mat = corr_mat
        self._corr_mat = copy.deepcopy(corr_mat)
        self._norm_time = norm_time

        self._set_operators(operators)
        self._set_tseps()

        if make_hermitian:
            self._make_hermitian()

        if norm_time is not None:
            self._normalize(norm_time)

        self._set_samplings_corr_mat()

    def _set_operators(self, operators):
        if len(operators):
            if self.N != len(operators):
                print(f"Mismatch in input correlator matrix size '{self.N}' and input operators of size '{len(operators)}'")
                sys.exit()
            self._operators = operators
        else:
            self._operators = list()
            for n in range(self.N):
                self._operators.append(f"op_{n}")

    def _set_tseps(self):
        tsep_min = None
        tsep_max = None
        for i in range(self.N):
            for j in range(self.N):
                tseps = self._corr_mat[i,j].independent_variables_values
                if tsep_min is None or tsep_min < min(tseps):
                    tsep_min = min(tseps)
                if tsep_max is None or tsep_max > max(tseps):
                    tsep_max = max(tseps)

        self._tseps = list(range(tsep_min, tsep_max+1))

        if len(self._tseps) == 0:
            print("No tseps left. All elements of correlator matrices must share some tseps")
            sys.exit()

    def _make_hermitian(self):
        for i in range(self.N):
            for j in range(i+1, self.N):
                self._corr_mat[i,j] = 0.5*(self._corr_mat[i,j] + self._corr_mat[j,i].conj())
                self._corr_mat[j,i] = self._corr_mat[i,j].conj()

            self._corr_mat[i,i] = self._corr_mat[i,i].real

    def _normalize(self, norm_time):
        normalizations = dict()
        for i in range(self.N):
            normalizations[i] = self._corr_mat[i, i](self._norm_time).samples[0]

        for i in range(self.N):
            for j in range(self.N):
                new_data_ij_dict = dict()
                for tsep in self.tseps:
                    new_data_ij_dict[tsep] = self._corr_mat[i, j](tsep) / np.sqrt(normalizations[i]*normalizations[j])

                self._corr_mat[i, j] = C2ptData(new_data_ij_dict, self.input_corr_mat[i,j].snk_operator, self.input_corr_mat[i,j].src_operator)

    def _set_samplings_corr_mat(self):
        samplings_corr_mat = np.empty((self.N, self.N, len(self.tseps), data_handler.get_num_samples()+1), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):
                for ti, tsep in enumerate(self.tseps):
                    samplings_corr_mat[i, j, ti, :] = self.corr_mat[i, j](tsep).samples[:]

        self._samplings_corr_mat = samplings_corr_mat

    def remove_operator_indices(self, op_indices):
        # check
        if min(op_indices) < 0:
            print("Cannot remove operator index less than 0")
            sys.exit()

        if max(op_indices) >= self.N:
            print("Cannot remove operator index greater than or equal to max")
            sys.exit()

        if len(op_indices) != len(set(op_indices)):
            print("Duplicates in operator indices to remove")
            sys.exit()

        new_corr_mat = np.empty((self.N-len(op_indices), self.N-len(op_indices)), dtype=object)

        kept_indices = set(range(self.N)) - set(op_indices)

        new_operators = list()
        for i_i, i in enumerate(kept_indices):
            new_operators.append(self.operators[i])
            for j_i, j in enumerate(kept_indices):
                new_corr_mat[i_i,j_i] = self.input_corr_mat[i,j]

        return C2ptMatrixData(new_corr_mat, new_operators, self.norm_time, self.hermitian)

    def remove_operators(self, op_names):
        op_indices = list()
        for op_i, op_name in enumerate(self.operators):
            if op_name in op_names:
                op_indices.append(op_i)

        return self.remove_operator_indices(op_indices)

    @property
    def input_corr_mat(self):
        return self._input_corr_mat

    @property
    def corr_mat(self):
        return self._corr_mat

    @property
    def samplings_corr_mat(self):
        return self._samplings_corr_mat

    @property
    def tseps(self):
        return self._tseps

    @property
    def operators(self):
        return self._operators

    @property
    def N(self):
        return self.corr_mat.shape[0]

    @property
    def norm_time(self):
        return self._norm_time

    @property
    def hermitian(self):
        for i in range(self.N):
            for j in range(i, self.N):
                if self._corr_mat[i,j] != self._corr_mat[j,i].conj():
                    return False
        return True

    def get_correlator_set(self):
        corrs = list()
        for i in range(self.N):
            corrs.append(self._corr_mat[i, i])

        for i in range(self.N):
            for j in range(i+1, self.N):
                corrs.append(self._corr_mat[i, j])

        return corrs
    
    def get_diagonal_correlator_set(self):
        corrs = list()
        for i in range(self.N):
            corrs.append(self.corr_mat[i, i])

        return corrs

    def get_off_diagonal_correlator_set(self):
        corrs = list()
        for i in range(self.N):
            for j in range(i+1, self.N):
                corrs.append(self.corr_mat[i, j])

        return corrs

    def get_principal_correlators(self, t0=None, td=None, mean=True, bypass_hermitian=False):
        """
        Args:
            t0 (int) - optional, if not given, then t0 is the smallest value that satisifies td/2 < t0 < td.
                       TODO: Is this the best choice?
            td (int) - optional, the diagonalization time. If missing, diagonalize at all times
            mean (bool) - optional. If True, only gevp on the mean is done. If False, gevp on
                          all resamplings is done

        Returns:
            C2ptMatrixData - contains the rotated correlators

        TODO:
            No eigenvector pinning is implemented, so one should only use the defaults
            - Remove all this repeat to save on memory
        """

        if not self._make_hermitian and not bypass_hermitian:
            if not self.is_hermitian():
                print("Correlator matrix must be Hermitian!")
                sys.exit()

        self._t0 = t0
        self._td = td
        self._mean = mean

        if mean and td is None and t0 is None:
            raise NotImplementedError("not implemented options: mean and td is None and t0 is None")
        elif mean and td is None:
            raise NotImplementedError("not implemented options: mean and td is None")
            eigvals = np.empty((self.N, len(self.tseps)), dtype=np.complex128)
            eigvecs = np.empty((self.N, self.N, len(self.tseps)), dtype=np.complex128)
            t0_i = self.tseps.index(t0)
            for td_i, td in enumerate(self.tseps):
                eigvals[:, td_i], eigvecs[:, :, td_i] = scipy.linalg.eigh(self.samplings_corr_mat[:, :, td_i, 0], self.samplings_corr_mat[:, :, t0_i, 0])

            ref_td = max(1, min(self.tseps))
            for td_i, td in enumerate(self.tseps):
                if td < ref_td:
                    reorder = find_reorder(eigvecs[:, :, td_i+1], eigvecs[:, :, td_i])
                elif td > ref_td:
                    reorder = find_reorder(eigvecs[:, :, td_i-1], eigvecs[:, :, td_i])

                if td != ref_td:
                    # reorder the eigvecs and eigvals at td_i based on reorder
                    ...

            eigvecs = np.repeat(eigvecs[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)
            eigvals = np.repeat(eigvals[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

        elif mean:
            t0_i = self.tseps.index(t0)
            td_i = self.tseps.index(td)
            Ct0_mean = self.samplings_corr_mat[:, :, t0_i, 0]
            Ctd_mean = self.samplings_corr_mat[:, :, td_i, 0]

            if not scipy.linalg.ishermitian(Ct0_mean) or not scipy.linalg.ishermitian(Ctd_mean):
                raise ValueError("Matrices must be Hermitian")

            _, eigvecs = scipy.linalg.eigh(Ctd_mean, Ct0_mean)
            eigvecs_mean = np.repeat(eigvecs[:, :, np.newaxis], len(self.tseps), axis=2)
            eigvecs = np.repeat(eigvecs_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

        elif td is None and t0 is None:
            raise NotImplementedError("not implemented options: td is None and t0 is None")
        elif td is None:
            raise NotImplementedError("not implemented options: td is None")
            eigvecs = np.empty((self.N, self.N, len(self.tseps), data_handler.get_num_samples()+1), dtype=np.complex128)
            t0_i = self.tseps.index(t0)
            for td_i, td in enumerate(self.tseps):
                for s_i in range(data_handler.get_num_samples()+1):
                    _, eigvecs[:, :, td_i, s_i] = scipy.linalg.eigh(self.samplings_corr_mat[:, :, td_i, s_i], self.samplings_corr_mat[:, :, t0_i, s_i])

        else:
            raise NotImplementedError("not implemented options:")
            eigvecs = np.empty((self.N, self.N, data_handler.get_num_samples()+1), dtype=np.complex128)
            t0_i = self.tseps.index(t0)
            td_i = self.tseps.index(td)
            for s_i in range(data_handler.get_num_samples()+1):
                _, eigvecs[:, :, s_i] = scipy.linalg.eigh(self.samplings_corr_mat[:, :, td_i, s_i], self.samplings_corr_mat[:, :, t0_i, s_i])

            eigvecs = np.repeat(eigvecs[:, :, np.newaxis, :], len(self.tseps), axis=2)

        eigvecs = np.flip(eigvecs, axis=1)  # orders eigenvectors by energy
        self._eigen_vecs = eigvecs

        principal_corrs_raw = self.rotate_raw(eigvecs)
        principal_corrs = np.empty((self.N, self.N), dtype=object)
        rot_operators = [f"Level {n_i}" for n_i in range(self.N)]
        for n_i in range(self.N):
            for n_j in range(self.N):
                c2pt_data_dict = dict()
                for ts_i, ts in enumerate(self.tseps):
                    c2pt_data_dict[ts] = data_handler.Data(principal_corrs_raw[n_i, n_j, ts_i, :])

                principal_corrs[n_i, n_j] = C2ptData(c2pt_data_dict, snk_operator=rot_operators[n_i], src_operator=rot_operators[n_j])

        return C2ptMatrixData(principal_corrs, rot_operators, make_hermitian=False)


    def rotate_raw(self, eigvecs):
        return np.einsum('ijlm,jklm,knlm->inlm', np.transpose(eigvecs.conj(), (1,0,2,3)) , self.samplings_corr_mat, eigvecs)

    @property
    def eigenvectors(self):
        if hasattr(self, '_eigen_vecs'):
            return self._eigen_vecs
        else:
            raise AttributeError("Eigenvectors not yet computed!")

    def compute_overlaps(self, t, amplitudes, filename=None):
        """
        Args:
            t - (int), specifies time for eigenvectors
            amplitudes - (np.array, float64)[energy_id, sample_i]

        Returns:
            overlaps - (np.array, Data)[op_id, level_id]
        """
        t0_i = self.tseps.index(self._t0)
        t_i = self.tseps.index(t)

        overlaps = np.einsum('jks,kns,ns->jns', self.samplings_corr_mat[:, :, t0_i, :] , self.eigenvectors[:, :, t_i, :], np.sqrt(amplitudes))
        overlap_list = list()
        for op_i in range(self.N):
            op_overlap_list = list()
            op_overlap_sum = 0.
            for level_i in range(overlaps.shape[1]):
                overlap = overlaps[op_i, level_i, :]*np.conj(overlaps[op_i, level_i, :])
                overlap = data_handler.Data(overlap.real)
                op_overlap_list.append(overlap)
                op_overlap_sum += overlap.samples[0]

            for level_i in range(overlaps.shape[1]):
                op_overlap_list[level_i] /= op_overlap_sum

            overlap_list.append(op_overlap_list)

        self._overlaps = np.array(overlap_list)

        if filename is not None:
            with open(filename, 'w') as f:
                for op_id in range(self.overlaps.shape[0]):
                    f.write(f"{self.operators[op_id]}\n")
                    for level_id in range(self.overlaps.shape[1]):
                        f.write(f"  {level_id}: {str(self.overlaps[op_id, level_id])}\n")
                    f.write('\n')

    @property
    def overlaps(self):
        if hasattr(self, '_overlaps'):
            return self._overlaps
        else:
            raise AttributeError("Overlaps not yet computed!")

    def get_energy_to_operator_map(self):
        """
        INFO: This uses the overlap means to try and associate an operator (or set of operators) to
              each energy level.

              For each energy level, each operator's overlaps are checked to see if that level
              has the Nth largest overlap. If it does, then that operator is added to the list
              of operators associated with that energy.

              Why N and not the largest? Sometimes, an energy level will never have the largest
              overlap for any of the operators. So, N starts as 0, but if no operator is found,
              then N becomes 1 and increases in this fashion until the level is matched to at
              least one operator.

        Returns:
            dict - Map from energy index to list of operator indices

        """
        num_ops = self.overlaps.shape[0]
        num_levels = self.overlaps.shape[1]
        overlap_means = np.ones((num_ops, num_levels), dtype=np.float64)
        for op_id in range(num_ops):
            for level_id in range(num_levels):
                overlap_means[op_id, level_id] = self.overlaps[op_id, level_id].mean

        energy_to_ops_map = dict()
        for level_id in range(num_levels):
            energy_to_ops_map[level_id] = list()
            rank = 0
            while len(energy_to_ops_map[level_id]) == 0:
                for op_id in range(num_ops):
                    overlap = sorted(list(overlap_means[op_id, :]), reverse=True)[rank]
                    rank_level_id = list(overlap_means[op_id, :]).index(overlap)
                    if rank_level_id == level_id:
                        energy_to_ops_map[level_id].append(op_id)

                rank += 1

        return energy_to_ops_map
    

    def get_amplitudes(self, tmin, tmax=-1, tmax_rel_error=0.0, num_exps=1, non_int_corrs=None):
        amplitudes = list()
        for n in range(self.N):
            corr_En = self[n,n].real
            tmax = max(corr_En.tseps)
            if non_int_corrs is not None:
                amplitude = corr_En.get_amplitude(tmin, tmax, tmax_rel_error, num_exps, non_int_corrs[n])
            else:
                amplitude = corr_En.get_amplitude(tmin, tmax, tmax_rel_error, num_exps)

            amplitudes.append(amplitude.samples)

        amplitudes = np.array(amplitudes)

        return amplitudes

    '''
    def get_principal_correlators_from_ev(self, t0, td, mean):
        """
        INFO: This does things more properly (see notes from Colin).
        Args:
            t0 (int) - required, the metric time
            td (int) - optional, the diagonalization time. If missing, diagonalize at all times
            mean (bool) - optional. If True, only gevp on the mean is done. If False, gevp on
                          all resamplings is done

        Returns:
            C2ptMatrixData - contains the rotated correlators

        TODO:
            No eigenvector pinning is implemented, so one should only use the defaults
        """
        if mean and td is None:
            ...

        elif mean:
            t0_i = self.tseps.index(t0)
            td_i = self.tseps.index(td)

            Ct0_mean = self.samplings_corr_mat[:, :, t0_i, 0]
            Ctd_mean = self.samplings_corr_mat[:, :, td_i, 0]

            if not scipy.linalg.ishermitian(Ct0_mean) or not scipy.linalg.ishermitian(Ctd_mean):
                raise ValueError("Matrices must be Hermitian")

            Ct0_eigvals_mean, Ct0_eigvecs_mean = scipy.linalg.eigh(Ct0_mean)
            Ct0_eigvecs_tseps_mean = np.repeat(Ct0_eigvecs_mean[:, :, np.newaxis], len(self.tseps), axis=2)
            Ct0_eigvecs_tseps_samps = np.repeat(Ct0_eigvecs_tseps_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

            Ct0_1h_eigvals_mean = np.sqrt(Ct0_eigvals_mean)
            Ct0_1h_eigvals_samps = np.repeat(Ct0_1h_eigvals_mean[:, np.newaxis], data_handler.get_num_samples()+1, axis=1)

            Ct0_inv1h_eigvals_mean = 1./Ct0_1h_eigvals_mean
            Ctd_rot_mean = np.einsum('ij,jk,kl->il', Ct0_eigvecs_mean.conj().T, Ctd_mean, Ct0_eigvecs_mean)
            Gt_mean = np.einsum('i,ij,j->ij', Ct0_inv1h_eigvals_mean, Ctd_rot_mean, Ct0_inv1h_eigvals_mean)
            Gt_eigvals_mean, Gt_eigvecs_mean = scipy.linalg.eigh(Gt_mean)
            Gt_eigvecs_tseps_mean = np.repeat(Gt_eigvecs_mean[:, :, np.newaxis], len(self.tseps), axis=2)
            Gt_eigvecs_tseps_samps = np.repeat(Gt_eigvecs_tseps_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)

            Rot_mean = np.einsum('ij,j,jk->ik', Ct0_eigvecs_mean, Ct0_inv1h_eigvals_mean, Gt_eigvecs_mean)
            Rot_tseps_mean = np.repeat(Rot_mean[:, :, np.newaxis], len(self.tseps), axis=2)
            Rot_tseps_samps = np.repeat(Rot_tseps_mean[:, :, :, np.newaxis], data_handler.get_num_samples()+1, axis=3)


        elif td is None:
            ...

        else:
            ...
        
        self._Ct0_eigenvectors = Ct0_eigvecs_tseps_samps
        self._G_eigenvectors = Gt_eigvecs_tseps_samps

        principal_corrs_raw = self.rotate_raw(Rot_tseps_samps)
        principal_corrs = np.empty((self.N, self.N), dtype=object)
        for n_i in range(self.N):
            for n_j in range(self.N):
                c2pt_data_dict = dict()
                for ts_i, ts in enumerate(self.tseps):
                    c2pt_data_dict[ts] = data_handler.Data(principal_corrs_raw[n_i, n_j, ts_i, :])

                principal_corrs[n_i, n_j] = C2ptData(c2pt_data_dict)

        return C2ptMatrixData(principal_corrs)

    @property
    def G_eigenvectors(self):
        if hasattr(self, '_G_eigenvectors'):
            return self._G_eigenvectors
        return None
    '''

    def __call__(self, row, col):
        return self._corr_mat[row,col]

    def __getitem__(self, indices):
        return self._corr_mat[*indices]
