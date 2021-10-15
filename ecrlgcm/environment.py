import os

USERNAME="ec2-user"
NETID="bnb32"
PROJECT_CODE="UCOR0044"

ROOT_DIR="/data"
USER_DIR=f"{ROOT_DIR}/{NETID}"
BASE_DIR=f"/home/{USERNAME}/environment"
os.environ['SCRATCH_DIR'] = SCRATCH_DIR = f"{USER_DIR}/scratch"

RUN_DIR=f"{ROOT_DIR}/cesm"
CESM_DIR=f"{USER_DIR}/my_cesm"
CESM_SCRIPTS=f"{CESM_DIR}/cime/scripts"

ISCA_DIR=f"{USER_DIR}/isca"

os.environ['CIME_OUT_DIR'] = f"{SCRATCH_DIR}/archive"
os.environ['CESM_INPUT_DATA_DIR'] = CESM_INPUT_DATA_DIR = f"{RUN_DIR}/inputdata"

os.environ['GFDL_BASE'] = GFDL_BASE = ISCA_DIR

os.environ['GCM_REPO_DIR'] = GCM_REPO_DIR = BASE_DIR+"/gcm_aws"
os.environ['ISCA_REPO_DIR'] = ISCA_REPO_DIR = BASE_DIR+"/isca_aws"
os.environ['CESM_REPO_DIR'] = CESM_REPO_DIR = BASE_DIR+"/cesm_aws"

os.environ['RAW_TOPO_DIR'] = RAW_TOPO_DIR = f'/data/{NETID}/paleodem_raw'

os.environ['ISCA_TOPO_DIR'] = ISCA_TOPO_DIR = os.path.join(RAW_TOPO_DIR,'land_masks/')

os.environ["PATH"]+=f":{CESM_SCRIPTS}"

os.environ['ORIG_CESM_CO2_FILE'] = ORIG_CESM_CO2_FILE = f'{CESM_INPUT_DATA_DIR}/atm/waccm/lb/LBC_2000climo_CMIP6_0p5degLat_c180227.nc'

T42_TOPO_FILE = 'T42_nc3000_Co060_Fi001_PF_nullRR_Nsw042_20180111.nc'
f19_TOPO_FILE = 'fv_1.9x2.5_nc3000_Nsw084_Nrs016_Co120_Fi001_ZR_GRNL_031819.nc'
f09_TOPO_FILE = 'fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
os.environ['T42_TOPO_FILE'] = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/{T42_TOPO_FILE}'
os.environ['f19_TOPO_FILE'] = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/{f19_TOPO_FILE}'
os.environ['f09_TOPO_FILE'] = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/{f09_TOPO_FILE}'

os.environ['ORIG_ISCA_TOPO_FILE'] = ORIG_ISCA_TOPO_FILE = os.path.join(GFDL_BASE,'input/land_masks/era_land_t42.nc')
os.environ['ORIG_TOPO_FILE'] = ORIG_TOPO_FILE = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/{f19_TOPO_FILE}'

os.environ['CESM_TOPO_DIR'] = CESM_TOPO_DIR = f'{USER_DIR}/topo_files/'

os.environ['CESM_CO2_DIR'] = CESM_CO2_DIR = f'{USER_DIR}/co2_files/'

os.environ['GFDL_ENV'] = GFDL_ENV = 'aws'

os.environ['GFDL_WORK'] = GFDL_WORK = f"{USER_DIR}/gfdl_work"

os.environ['GFDL_DATA'] = GFDL_DATA = f"{USER_DIR}/gfdl_data"

os.environ['RAW_CO2_DIR'] = RAW_CO2_DIR = os.path.join(GFDL_BASE,'exp/test_cases/variable_co2_concentration/input/')

os.environ['CO2_DIR'] = CO2_DIR = os.path.join(ISCA_REPO_DIR,'experiments/input')
