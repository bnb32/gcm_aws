import ecrlgcm.environment
from ecrlgcm.misc import edit_namelists,get_logger,land_year_range,min_land_year,max_land_year,get_base_topofile
from ecrlgcm.preprocessing import eccentricity, obliquity, solar_constant, interpolate_co2
from ecrlgcm.preprocessing import modify_cesm_input_files
from ecrlgcm.experiment import Experiment, Configuration

import os
import argparse
import numpy as np
from datetime import date
import time

logger = get_logger()

experiment_dictionary = {
        'cam_clmCN_docnSOM':{'compset':'F1850_CAM60_CLM50%CN_SICE_DOCN%SOM_SROF_SGLC_SWAV','res':'f19_f19_mg17','custom':True},
        'cam_clmCN_docnDOM':{'compset':'F1850_CAM60_CLM50%CN_SICE_DOCN%DOM_SROF_SGLC_SWAV','res':'f19_f19_mg17','custom':True},
        'aqua':{'compset':'QSC6','res':'f19_f19_mg17','custom':True},
        }

parser=argparse.ArgumentParser(description="Run CESM")
parser.add_argument('-exp',default='cam_clmCN_docnDOM',
                    choices=[e for e in experiment_dictionary])
parser.add_argument('-multiplier',default=1.0,type=float, help="CO2 Multiplier")
parser.add_argument('-co2_value',default=None, help="CO2 Value")
parser.add_argument('-sea_level',default=0, type=float, help="Sea level")
parser.add_argument('-max_depth',default=1000, type=float, help="Max ocean depth")
parser.add_argument('-year',default=0,type=land_year_range,
                    metavar=f'[{min_land_year}-{max_land_year}]',
                    help="Years prior to current era in units of Ma")
parser.add_argument('-case',default='test',type=str)
parser.add_argument('-ntasks',default=96,type=int)
parser.add_argument('-nthrds',default=8,type=int)
parser.add_argument('-start_date',default="0001-01-01")
parser.add_argument('-step_type',default='ndays')
parser.add_argument('-nsteps',default=300,type=int)
parser.add_argument('-restart',default=False,action='store_true')
parser.add_argument('-setup',default=False,action='store_true')
parser.add_argument('-build',default=False,action='store_true')
parser.add_argument('-run',default=False,action='store_true')
parser.add_argument('-run_all',default=False,action='store_true')
parser.add_argument('-remap',default=False,action='store_true')
parser.add_argument('-remap_hires',default=False,action='store_true')
parser.add_argument('-timing',default=False,action='store_true')
args=parser.parse_args()

cwd=os.getcwd()

if args.exp not in experiment_dictionary:
    logger.error("Select valid case")
    exit()
else:
    args.res = experiment_dictionary[args.exp]['res']
    args.compset = experiment_dictionary[args.exp]['compset']
    args.custom = experiment_dictionary[args.exp]['custom']

if args.co2_value is None:
    args.co2_value = args.multiplier*interpolate_co2(args.year)

cesmexp = Experiment(gcm_type='cesm',
                     multiplier=args.multiplier,
                     land_year=args.year,
                     res=args.res,
                     exp_type=args.exp,
                     sea_level=args.sea_level,
                     max_depth=args.max_depth)

if args.timing:
    args.case = f'{os.environ["CESM_REPO_DIR"]}/cases/timing_test'
else:
    args.case = f'{os.environ["CESM_REPO_DIR"]}/cases/{cesmexp.name}'

args.orbit_year = args.year*10**6-date.today().year
logger.info('**Starting experiment**')
logger.info(f'Using CO2 Value {args.co2_value} ppm')
logger.info(f'Using orbital year {args.orbit_year} B.P.')
logger.info(f'Using solar constant {solar_constant(args.year)} W/m^2')

if not args.restart:
    logger.info('**Modifying input files**')
    modify_cesm_input_files(cesmexp,remap=args.remap,remap_hires=args.remap_hires)
    logger.info('**Done modifying input files**')

create_case_cmd=f'create_newcase --case {args.case} --res {args.res} --mach aws_c5 --compset {args.compset} --handle-preexisting-dirs r --output-root {os.environ["CIME_OUT_DIR"]}'

sim_config = Configuration(cesmexp,args)

if args.custom: create_case_cmd+=' --run-unsupported'

if args.setup or args.run_all:# or not os.path.exists(args.case):
    logger.info(f'Removing case directory: {args.case}')
    os.system(f'rm -rf {args.case}')
    if args.timing:
        logger.info(f"Removing output directory: {os.environ['CIME_OUT_DIR']}/{args.case.split('/')[-1]}")
        os.system(f"rm -rf {os.environ['CIME_OUT_DIR']}/{args.case.split('/')[-1]}")
    else:
        logger.info(f"Removing output directory: {cesmexp.file_path}")
        os.system(f"rm -rf {cesmexp.file_path}")

    os.system(create_case_cmd)
    logger.info(f"Creating case: {create_case_cmd}")
    os.chdir(args.case)
    logger.info(f"Changing namelists")
    edit_namelists(cesmexp,sim_config)
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
    cmd='./xmlchange CONTINUE_RUN=TRUE,'
    cmd+=f'STOP_OPTION={args.step_type},'
    cmd+=f'REST_N={args.nsteps//5+1},'
    cmd+=f'STOP_N={args.nsteps}'
    os.system(cmd)
    logger.info(f"Changing xml files: {cmd}")

if args.restart or args.run or args.run_all:
    start_time = time.time()
    try:
        os.chdir(args.case)
        os.system('./case.submit')
    except:
        logger.error("Error submitting case")
        exit()
    end_time = time.time()
    logger.info(f'Simulation time: {end_time-start_time}')
