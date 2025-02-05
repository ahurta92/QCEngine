""" Calls the Madness moldft executable.
"""

# import re
import copy
import json
import logging
import pprint
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import qcelemental as qcel
from qcelemental.models import AtomicResult, Provenance, AtomicInput
from qcelemental.util import safe_version, which

from qcengine.config import TaskConfig, get_config
from qcengine.exceptions import UnknownError

from ...exceptions import InputError
from ..model import ProgramHarness

from .harvester import harvest_scf_info, harvest_calc_info, harvest_response_properties
from ...util import create_mpi_invocation, execute

pp = pprint.PrettyPrinter(width=120, compact=True, indent=1)
logger = logging.getLogger(__name__)


class MadnessHarness(ProgramHarness):
    """
    Notes
    -----
    * To use the TCE, specify ``AtomicInput.model.method`` as usual, then also include ``qc_module = True`` in ``AtomicInput.keywords``.
    """

    _defaults = {
        "name": "madness",
        "scratch": True,
        "thread_safe": False,
        "thread_parallel": True,
        "node_parallel": True,
        "managed_memory": True,
    }
    # ATL: OpenMP only >=6.6 and only for Phi; potential for Mac using MKL and Intel compilers
    version_cache: Dict[str, str] = {}

    class Config(ProgramHarness.Config):
        pass

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        """Whether Madness harness is ready for operation, with both the QC program and any particular dependencies found.

        Parameters
        ----------
        raise_error: bool
            Passed on to control negative return between False and ModuleNotFoundError raised.

         Returns
         -------
         bool
             If both m-a-d-n-e-s-s and its harness dependency networkx are found, returns True.
             If raise_error is False and nwchem or networkx are missing, returns False.
             If raise_error is True and nwchem or networkx are missing, the error message for the first missing one is raised.

        """
        qc = which(
            "mad-dft",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install via https://github.com/m-a-d-n-e-s-s/madness",
        )
        return bool(qc)  # and dep

    ## gotta figure out which input file and from where
    def get_version(self) -> str:
        if self.found(raise_error=True):
            config = get_config()
            which_prog = str(which("madqc"))
            if config.use_mpiexec:
                command = create_mpi_invocation(str(which_prog), config)
            else:
                command = [which_prog]
            if which_prog not in self.version_cache:
                success, output = execute(
                    command,
                    infiles=None,
                    scratch_directory=config.scratch_directory,
                )
                for line in output["stdout"].splitlines():
                    if "multiresolution suite" in line:
                        version = line.strip().split()[1]
                        print("version", version)
                        self.version_cache[which_prog] = safe_version(version)
                return str(self.version_cache[which_prog])
        else:
            raise ModuleNotFoundError("MADNESS executable not found.")

    def compute(self, input_model: "AtomicInput", config: "TaskConfig") -> "AtomicResult":
        """
        Runs madness in executable mode
        """
        self.found(raise_error=True)

        job_inputs = self.build_input(input_model, config)

        tmpdir = config.scratch_directory

        success, output = execute(
            job_inputs["commands"],
            job_inputs["infiles"],
            outfiles=["calc_path.json"],
            scratch_directory=tmpdir,
            scratch_messy=True,
            scratch_exist_ok=True,
        )

        def collect_output_jsons(calc_path_json, output_dir):
            # first print calc_path_json
            print("calc_path_json: ", calc_path_json)
            calc_path_json = json.loads(calc_path_json)
            # first get calc_info and scf_info
            calc_info_path = Path(output_dir) / calc_path_json["moldft"]["outfiles"]["calc_info"]
            scf_info_path = Path(output_dir) / calc_path_json["moldft"]["outfiles"]["scf_info"]

            with open(calc_info_path) as file:
                calc_info_json = json.load(file)
            with open(scf_info_path) as file:
                scf_info_json = json.load(file)

            output_json = {}
            output_json["moldft"] = {}
            output_json["moldft"]["calc_info"] = calc_info_json
            output_json["moldft"]["scf_info"] = scf_info_json

            # now get the response
            if calc_path_json["response"]["calc_dirs"] is not None:
                for response_base, response_calc in zip(
                    calc_path_json["response"]["outfiles"], calc_path_json["response"]["calc_dirs"]
                ):
                    print(response_base, response_calc)
                    dir_path = Path(response_calc).stem
                    print(dir_path)
                    response_base_path = output_dir / response_base
                    output_json["response"] = {}
                    with open(response_base_path) as file:
                        output_json["response"][dir_path] = json.load(file)
            return output_json

        if success:
            output_dir = output["scratch_directory"]
            output_json = collect_output_jsons(output["outfiles"]["calc_path.json"], output_dir)
            output_json["stdout"] = output["stdout"]
            output_json["stderr"] = output["stderr"]

            return self.parse_output(output_json, input_model)
        else:
            print(output["stdout"])
            raise UnknownError(output["stderr"])

    def build_input(
        self, input_model: AtomicInput, config: TaskConfig, template: Optional[str] = None
    ) -> Dict[str, Any]:

        #
        madnessrec = {
            "infiles": {},
            "scratch_directory": config.scratch_directory,
            "scratch_messy": config.scratch_messy,
        }

        ## These are the madness keywords
        opts = copy.deepcopy(input_model.keywords)
        print("opts1: ", opts)

        print(input_model.keywords)

        json_input = json.dumps(input_model.keywords, indent=4)
        print("json input", json_input)
        # Handle Molecule
        molcmd, moldata = input_model.molecule.to_string(dtype="madness", units="bohr", return_data=True)
        print("moldata", moldata)
        print("molcmd", molcmd)
        # Log the job settings (LORI)  Not sure if i need this
        logger.debug("JOB_OPTS")
        logger.debug(pp.pformat(opts))

        madnessrec["infiles"]["input.json"] = json_input
        madnessrec["infiles"]["mol_input"] = molcmd
        madnessrec["commands"] = [which("mad-dft"), "input.json", "mol_input"]
        # optcmd="dft\n xc hf \nend\n"
        # print(madnessrec["infiles"]["input"])
        return madnessrec

    def parse_output(self, output_json: Dict, input_model: "AtomicInput") -> "AtomicResult":

        madmol = input_model.molecule  # qcprops = extract_formatted_properties(output_json)

        print("output_json keys", output_json.keys())

        outfiles = {}
        outfiles["moldft"] = output_json["moldft"]
        outfiles["response"] = output_json.get("response", None)

        scf_info = harvest_scf_info(output_json["moldft"]["scf_info"])
        print("scf_info", scf_info)
        calcinfo = harvest_calc_info(output_json["moldft"]["calc_info"])
        print("calcinfo", calcinfo)

        qcvars = {**scf_info, **calcinfo}
        print("qcvars", qcvars)

        properties = {k.lower(): v for k, v in qcvars.items()}
        native_files = {}
        native_files["calc_info.json"] = json.dumps(outfiles["moldft"]["calc_info"])
        native_files["scf_info.json"] = json.dumps(outfiles["moldft"]["scf_info"])
        if outfiles["response"] is not None:
            native_files["response"] = {}
            for response_key in outfiles["response"].keys():
                native_files["response"][response_key] = json.dumps(outfiles["response"][response_key])
        print("native_files", native_files.keys())
        print("native files response keys", native_files["response"].keys())

        provenance = Provenance(creator="madness", version=self.get_version(), routine="madness")

        def get_return_results(input_model, qcvars):
            if input_model.driver == "energy":
                retres = qcvars["RETURN_ENERGY"]
            elif input_model.driver == "gradient":
                retres = qcvars[
                    "CURRENT GRADIENT"
                ]  # in madness I need to create a gradient and hession output json file
            elif input_model.driver == "hessian":
                retres = qcvars["CURRENT HESSIAN"]
            else:
                raise InputError("Driver not understood")
            return retres

        retres = get_return_results(input_model, qcvars)
        stdout = output_json.pop("stdout")
        stderr = output_json.pop("stderr")

        # Format them inout an output
        output_data = {
            "schema_version": 1,
            "molecule": madmol,  # overwrites with outfile Cartesians in case fix_*=F
            "extras": {**input_model.extras},
            "native_files": native_files,
            "properties": properties,
            "provenance": provenance,
            "return_result": retres,
            "stderr": stderr,
            "stdout": stdout,
            "success": True,
        }
        # got to even out who needs plump/flat/Decimal/float/ndarray/list
        # Decimal --> str preserves precision
        output_data["extras"]["qcvars"] = {
            k.upper(): str(v) if isinstance(v, Decimal) else v for k, v in qcel.util.unnp(qcvars, flat=True).items()
        }

        output_data["extras"]["response_properties"] = harvest_response_properties(outfiles["response"])

        output_data["extras"]["outfiles"] = native_files

        return AtomicResult(**{**input_model.dict(), **output_data})
