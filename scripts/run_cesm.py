"""Run CESM simulations"""
import os
from datetime import date
import time
import sys

from ecrlgcm import cesm_argparse, EXPERIMENT_DICTIONARY
from ecrlgcm.environment import EnvironmentConfig
from ecrlgcm.utilities import (get_logger)
from ecrlgcm.preprocessing import (solar_constant, interpolate_co2)
from ecrlgcm.preprocessing import PreProcessing
from ecrlgcm.experiment import Experiment, Configuration

logger = get_logger()


if __name__ == '__main__':
    parser = cesm_argparse()
    args = parser.parse_args()

    cwd = os.getcwd()

    if args.exp not in EXPERIMENT_DICTIONARY:
        logger.error("Select valid case")
        sys.exit()
    else:
        args.res = EXPERIMENT_DICTIONARY[args.exp]['res']
        args.compset = EXPERIMENT_DICTIONARY[args.exp]['compset']
        args.custom = EXPERIMENT_DICTIONARY[args.exp]['custom']

    if args.co2_value is None:
        args.co2_value = args.multiplier * interpolate_co2(args.year)

    config = EnvironmentConfig(args.config)
    cesmexp = Experiment(config, gcm_type='cesm',
                         multiplier=args.multiplier,
                         land_year=args.year,
                         res=args.res,
                         exp_type=args.exp,
                         sea_level=args.sea_level,
                         max_depth=args.max_depth)
    pre_proc = PreProcessing(config)

    if args.timing:
        args.case = f'{config.CESM_REPO_DIR}/cases/timing_test'
    else:
        args.case = f'{config.CESM_REPO_DIR}/cases/{cesmexp.name}'

    args.orbit_year = args.year * 10**6 - date.today().year
    logger.info('**Starting experiment**')
    logger.info(f'Using CO2 Value {args.co2_value} ppm')
    logger.info(f'Using orbital year {args.orbit_year} B.P.')
    logger.info(f'Using solar constant {solar_constant(args.year)} W/m^2')

    if not args.restart:
        logger.info('**Modifying input files**')
        pre_proc.modify_cesm_input_files(
            cesmexp, remap=args.remap, remap_hires=args.remap_hires)
        logger.info('**Done modifying input files**')

    create_case_cmd = f'create_newcase --case {args.case} --res '
    create_case_cmd += f'{args.res} --mach aws_c5 --compset {args.compset} '
    create_case_cmd += '--handle-preexisting-dirs r --output-root '
    create_case_cmd += f'{config.CIME_OUT_DIR}'

    sim_config = Configuration(cesmexp, args)

    if args.custom:
        create_case_cmd += ' --run-unsupported'

    if args.setup or args.run_all:
        logger.info(f'Removing case directory: {args.case}')
        os.system(f'rm -rf {args.case}')
        if args.timing:
            logger.info(
                "Removing output directory: "
                f"{config.CIME_OUT_DIR}/{args.case.split('/')[-1]}")
            os.system(
                f"rm -rf {config.CIME_OUT_DIR}/{args.case.split('/')[-1]}")
        else:
            logger.info(f"Removing output directory: {cesmexp.file_path}")
            os.system(f"rm -rf {cesmexp.file_path}")

        os.system(create_case_cmd)
        logger.info(f"Creating case: {create_case_cmd}")
        os.chdir(args.case)
        logger.info("Changing namelists")
        pre_proc.edit_namelists(config, cesmexp, sim_config)
        os.chdir(args.case)
        logger.info(f"Changing xml files: {sim_config['change_xml_cmd']}")
        os.system(sim_config['change_xml_cmd'])
        os.system('./case.setup --reset')
        os.system('./preview_run')

    if args.build or args.run_all:

        os.chdir(args.case)
        os.system('./case.build --skip-provenance-check')

    if args.restart:
        os.chdir(args.case)
        cmd = './xmlchange CONTINUE_RUN=TRUE, '
        cmd += f'STOP_OPTION={args.step_type}, '
        cmd += f'REST_N={args.nsteps//5+1}, '
        cmd += f'STOP_N={args.nsteps}'
        os.system(cmd)
        logger.info(f"Changing xml files: {cmd}")

    if args.restart or args.run or args.run_all:
        start_time = time.time()
        try:
            os.chdir(args.case)
            os.system('./case.submit')
        except Exception:
            logger.error("Error submitting case")
            sys.exit()
        end_time = time.time()
        logger.info(f'Simulation time: {end_time-start_time}')
