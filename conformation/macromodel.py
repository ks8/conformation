""" Run Schrodinger's MacroModel conformational search tools (LMOD and MCMM). """
import datetime
import io
from logging import Logger
import os
import pickle
import subprocess
import time
from typing_extensions import Literal

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    num_steps: int = 10  # Number of search steps
    search_type: Literal['LMCS', 'MCMM'] = 'MCMM'  # Search type
    schrodinger_root: str = "/data/swansonk1/schrodinger2020-3"  # Path to Schrodinger install
    init_minimize: bool = False  # Whether or not to FF-minimize the initial ETKDG-generated conformation
    timeout: int = 3600*6  # Timeout for subprocess.check_call
    save_dir: str  # Directory path for output files


LMCS = \
    """confsearch.mae
confsearch-out.maegz
 MMOD       0      1      0      0     0.0000     0.0000     0.0000     0.0000
 FFLD      10      1      0      0     1.0000     0.0000     0.0000     0.0000
 BDCO       0      0      0      0    41.5692 99999.0000     0.0000     0.0000
 READ       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 CRMS       0      0      0      0     0.0000     0.2500     0.0000     0.0000
 LMCS    5000      0      0      0     0.0000     0.0000     3.0000     6.0000
 NANT       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 MCNV       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 MCSS       2      0      0      0    21.0000     0.0000     0.0000     0.0000
 MCOP       1      0      0      0     0.5000     0.0000     0.0000     0.0000
 DEMX       0 333333      0      0    21.0000    42.0000     0.0000     0.0000
 MSYM       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 AUOP       0      0      0      0  2500.0000     0.0000     0.0000     0.0000
 AUTO       0      2      1      1     0.0000    -1.0000     0.0000     2.0000
 CONV       2      0      0      0     0.0010     0.0000     0.0000     0.0000
 MINI       1      0 999999      0     0.0000     0.0000     0.0000     0.0000
 """

MCMM = \
    """confsearch.mae
confsearch-out.maegz
 MMOD       0      1      0      0     0.0000     0.0000     0.0000     0.0000
 FFLD      10      1      0      0     1.0000     0.0000     0.0000     0.0000
 BDCO       0      0      0      0    41.5692 99999.0000     0.0000     0.0000
 READ       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 CRMS       0      0      0      0     0.0000     0.2500     0.0000     0.0000
 MCMM    5000      0      0      0     0.0000     0.0000     0.0000     0.0000
 NANT       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 MCNV       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 MCSS       2      0      0      0    21.0000     0.0000     0.0000     0.0000
 MCOP       1      0      0      0     0.5000     0.0000     0.0000     0.0000
 DEMX       0 333333      0      0    21.0000    42.0000     0.0000     0.0000
 MSYM       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 AUOP       0      0      0      0  2500.0000     0.0000     0.0000     0.0000
 AUTO       0      2      1      1     0.0000    -1.0000     0.0000     2.0000
 CONV       2      0      0      0     0.0010     0.0000     0.0000     0.0000
 MINI       1      0 999999      0     0.0000     0.0000     0.0000     0.0000
 """


def create_macromodel_conf(num_steps, search_type: Literal['LMCS', 'MCMM']):
    """
    Create macromodel search .com file.
    :param num_steps: Number of steps to run conformational search.
    :param search_type: Conformational search type (either LMCS or MCMM).
    """
    # Change input and output file names in the .com file
    if search_type == 'LMCS':
        comf = io.StringIO(LMCS)
    else:
        comf = io.StringIO(MCMM)
    com = comf.readlines()
    com[0] = 'INPUT.mae\n'
    com[1] = 'OUTPUT.mae\n'

    # Change the step count in the .com file
    cycles = (str(num_steps)).rjust(6)
    temp = list(com[7])
    temp[7:13] = list(cycles)
    com[7] = "".join(temp)
    comf.truncate(0)
    comf.seek(0)
    comf.writelines(com)

    return comf.getvalue()


def check_log_success(log_str):
    """
    Check log success.
    """
    if 'BatchMin: normal termination' in log_str:
        return True
    return False


def create_job_basename(programname):
    """
    Create job basename.
    """
    now = datetime.datetime.now()
    timestamp_string = now.strftime("%Y%m%d_%H%M%S.%f")
    return "{}.{}".format(programname, timestamp_string)


# noinspection PyUnresolvedReferences
def mol_to_sdfstr(mol: rdchem.Mol) -> str:
    """
    Convert an RDKit molecule to an SDF string.
    """
    f = io.StringIO()
    writer = Chem.SDWriter(f)
    writer.write(mol)
    # noinspection PyArgumentList
    writer.close()
    return f.getvalue()


def run_macromodel_cache(args: Args, mol, num_steps, search_type, walltime=3600 * 6):
    """
    Run MacroModel.
    """
    start_time = time.time()

    job_string = create_macromodel_conf(num_steps, search_type)

    sdconvert_path = f"{args.schrodinger_root}/utilities/sdconvert"
    bmin_path = f"{args.schrodinger_root}/bmin"

    basename = create_job_basename("macromodel")
    basepath = os.path.join(os.path.abspath(args.save_dir), basename)

    mol.SetProp("_Name", "")  # get rid of any name string for caching reasons
    mol_sdf = mol_to_sdfstr(mol)

    sdf_filename = basepath + ".sdf"
    mae_filename = basepath + ".mae"

    input_filename = basepath + ".com"
    log_filename = basepath + ".log"
    maeout_filename = basepath + "_out.mae"

    cache_key = {'sdf': mol_sdf,
                 'job_string': job_string}

    # noinspection PyUnusedLocal
    was_successful = False

    with open(sdf_filename, 'w') as sdf_fid:
        sdf_fid.write(mol_sdf)
    try:
        subprocess.check_call(f"{sdconvert_path} -isd {sdf_filename} -omae {mae_filename}", shell=True)

        job_string = job_string.replace("INPUT.mae", mae_filename)
        job_string = job_string.replace("OUTPUT.mae", maeout_filename)

        with open(input_filename, 'w') as job_fid:
            job_fid.write(job_string)
            job_fid.flush()

        timeout_error = False
        # noinspection PyUnusedLocal
        try:
            subprocess.check_call(f"{bmin_path} -WAIT {basepath} ", cwd=os.path.abspath(args.save_dir),
                                  shell=True, timeout=walltime)
        except subprocess.TimeoutExpired as e:
            timeout_error = True

        if os.path.exists(log_filename):
            job_log = open(log_filename, 'r').read()
        else:
            job_log = None

        if os.path.exists(maeout_filename):
            mae_out = open(maeout_filename, 'rb').read()
        else:
            mae_out = None

        results = {'job_log': job_log,
                   'mae_out': mae_out,
                   'mol': mol,
                   'params': {num_steps, search_type}}

        if timeout_error:
            was_successful = False
            results['fail_reason'] = 'timeout'
        elif job_log is not None and check_log_success(job_log):
            was_successful = True

        else:
            results['fail_reason'] = 'job_log'
            was_successful = False

    except Exception as e:
        was_successful = False
        results = {'exception': e, 'fail_reason': 'exception'}

    end_time = time.time()
    return {'runtime': end_time - start_time,
            'success': was_successful,
            'cache_key': cache_key,
            'result': results,
            }


def macromodel(args: Args, logger: Logger):
    """
    Run Schrodinger's MacroModel conformational search tools (LMOD and MCMM).
    :param args: System arguments.
    :param logger: System logger.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    # Embed molecule
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    if args.init_minimize:
        AllChem.MMFFOptimizeMoleculeConfs(mol)

    debug(f'Starting conformational search...')
    start_time = time.time()

    job_string = create_macromodel_conf(args.num_steps, args.search_type)

    sdconvert_path = f"{args.schrodinger_root}/utilities/sdconvert"
    bmin_path = f"{args.schrodinger_root}/bmin"

    basename = create_job_basename("macromodel")
    basepath = os.path.join(os.path.abspath(args.save_dir), basename)

    mol.SetProp("_Name", "")  # get rid of any name string for caching reasons
    mol_sdf = mol_to_sdfstr(mol)

    sdf_filename = basepath + ".sdf"
    mae_filename = basepath + ".mae"

    input_filename = basepath + ".com"
    log_filename = basepath + ".log"
    maeout_filename = basepath + "_out.mae"

    cache_key = {'sdf': mol_sdf,
                 'job_string': job_string}

    # noinspection PyUnusedLocal
    was_successful = False

    with open(sdf_filename, 'w') as sdf_fid:
        sdf_fid.write(mol_sdf)

    try:
        # Convert SDF file to MAE file for MacroModel input
        subprocess.check_call(f"{sdconvert_path} -isd {sdf_filename} -omae {mae_filename}", shell=True)

        job_string = job_string.replace("INPUT.mae", mae_filename)
        job_string = job_string.replace("OUTPUT.mae", maeout_filename)

        with open(input_filename, 'w') as job_fid:
            job_fid.write(job_string)
            job_fid.flush()

        timeout_error = False
        # noinspection PyUnusedLocal
        try:
            subprocess.check_call(f"{bmin_path} -WAIT {basepath} ", cwd=os.path.abspath(args.save_dir), shell=True,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired as e:
            timeout_error = True

        if os.path.exists(log_filename):
            job_log = open(log_filename, 'r').read()
        else:
            job_log = None

        if os.path.exists(maeout_filename):
            mae_out = open(maeout_filename, 'rb').read()
        else:
            mae_out = None

        results = {'job_log': job_log,
                   'mae_out': mae_out,
                   'mol': mol,
                   'params': {args.num_steps, args.search_type}}

        if timeout_error:
            was_successful = False
            results['fail_reason'] = 'timeout'
        elif job_log is not None and check_log_success(job_log):
            was_successful = True

        else:
            results['fail_reason'] = 'job_log'
            was_successful = False

    except Exception as e:
        was_successful = False
        results = {'exception': e, 'fail_reason': 'exception'}

    # Convert output MAE file to SDF file
    sdf_out = os.path.join(args.save_dir, "conformations.sdf")
    subprocess.check_call(f"{sdconvert_path} -imae {maeout_filename} -osd {sdf_out}", shell=True)

    # Load conformations into RDKit and save
    debug(f'Loading conformations into RDKit...')
    suppl = Chem.SDMolSupplier(sdf_out, removeHs=False)
    mol_result = None
    for i, tmp in enumerate(tqdm(suppl)):
        if i == 0:
            mol_result = tmp
        else:
            c = tmp.GetConformer()
            c.SetId(i)
            mol_result.AddConformer(c)

    debug(f'Saving conformations from RDKit...')
    bin_str = mol_result.ToBinary()
    with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as b:
        b.write(bin_str)

    end_time = time.time()
    res = {'runtime': end_time - start_time, 'success': was_successful, 'cache_key': cache_key, 'result': results}
    debug(f'Success: {was_successful}')
    debug(f'Total Time (s): {end_time - start_time}')
    pickle.dump(res, open(os.path.join(args.save_dir, "results.pickle"), 'wb'))
