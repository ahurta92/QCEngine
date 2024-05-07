import re
import json
import logging
import pandas as pd

# from decimal import Decimal
from typing import Tuple

import numpy as np

# import qcelemental as qcel
from qcelemental.models import Molecule
from qcelemental.models.results import AtomicResultProperties

from ..util import PreservingDict

logger = logging.getLogger(__name__)


# collect the scf_info json
# first iterate through single numbers
# Then iteration throught tensor values
# this format should work for response values in the future
def harvest_scf_info(scf_info):
    """Harvest the SCF information from the SCF JSON"""
    psivar = PreservingDict()

    # We only need to last set in the list
    scf_tensor_vars = ["scf_dipole_moment"]
    for var in scf_tensor_vars:
        if scf_info.get(var) is not None:
            psivar[var.upper()] = tensor_to_numpy(scf_info.get(var))

    return psivar


def harvest_calc_info(calc_info):
    """Harvest the Calc information from the Calc JSON"""
    psivar = PreservingDict()
    qcvars = ["calcinfo_nbasis", "calcinfo_nmo", "calcinfo_nalpha", "calcinfo_nbeta", "calcinfo_natom", "return_energy"]
    # TODO (ahurta92) I can add more qcvars here from ['scf_e_data']  coulomb kinetic, xc, pcm, nuclear, etc ...

    for var in qcvars:
        if calc_info.get(var) is not None:
            psivar[var.upper()] = calc_info.get(var)

    return psivar


def tensor_to_numpy(j):
    array = np.empty(j["size"])
    array[:] = j["vals"]
    print(tuple(j["dims"]))
    return np.reshape(array, tuple(j["dims"]))


def read_frequency_proto_iter_data(my_iter_data, num_states):
    num_iters = len(my_iter_data)
    dres = np.empty((num_iters, num_states))
    res_X = np.empty((num_iters, num_states))
    res_Y = np.empty((num_iters, num_states))
    polar = np.empty((num_iters, 3, 3))
    for i in range(num_iters):
        dres[i, :] = tensor_to_numpy(my_iter_data[i]["density_residuals"])
        res_X[i, :] = tensor_to_numpy(my_iter_data[i]["res_X"])
        res_Y[i, :] = tensor_to_numpy(my_iter_data[i]["res_Y"])
        polar[i, :, :] = tensor_to_numpy(my_iter_data[i]["polar"])
    data = {}
    names = ["density_residuals", "res_X", "res_Y", "polar"]
    vals = [dres, res_X, res_Y, polar]
    for name, val in zip(names, vals):
        data[name] = val
    return data


def read_excited_proto_iter_data(my_iter_data, num_states):
    num_iters = len(my_iter_data)
    dres = np.empty((num_iters, num_states))
    res_X = np.empty((num_iters, num_states))
    res_Y = np.empty((num_iters, num_states))
    omega = np.empty((num_iters, num_states))
    for i in range(num_iters):
        dres[i, :] = tensor_to_numpy(my_iter_data[i]["density_residuals"])
        res_X[i, :] = tensor_to_numpy(my_iter_data[i]["res_X"])
        res_Y[i, :] = tensor_to_numpy(my_iter_data[i]["res_Y"])
        omega[i, :] = tensor_to_numpy(my_iter_data[i]["omega"])
    data = {}
    names = ["density_residuals", "res_X", "res_Y", "omega"]
    vals = [dres, res_X, res_Y, omega]
    for name, val in zip(names, vals):
        data[name] = val
    return data


# input response_info json and returns a dict of response paramters


def harvest_response_properties(response_base_files):
    """Harvest the response information from the response JSON"""
    psivar = PreservingDict()

    print(response_base_files)

    alphas = np.zeros((len(response_base_files), 10), dtype=np.float64)
    i = 0
    for file_name, rp in response_base_files.items():
        print("file_name", file_name)
        print("response_base", rp)
        parameters = rp["parameters"]
        data = rp["response_data"]["data"]
        # make a 10,1 array of zeros
        values = np.zeros((10,), dtype=np.float64)
        values[0] = parameters["omega"]

        alpha = data["alpha"]
        alpha = tensor_to_numpy(alpha)
        values[1:] = alpha[-1, :].reshape(9,)
        alphas[i, :] = values[:]
    # pandas dataframe

    psivar["polarizability"] = np.array(alphas, dtype=np.float64)

    return psivar
