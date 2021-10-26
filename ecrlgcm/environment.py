import os

USERNAME="ec2-user"
NETID="bnb32"
PROJECT_CODE="UCOR0044"

ROOT_DIR="/data"
os.environ['USER_DIR'] = USER_DIR = f"{ROOT_DIR}/{NETID}"
os.environ['USER_INPUT_DIR'] = USER_INPUT_DIR = f"{ROOT_DIR}/{NETID}/inputdata"
os.environ['BASE_DIR'] = BASE_DIR = f"/home/{USERNAME}/environment"
os.environ['SCRATCH_DIR'] = SCRATCH_DIR = f"{USER_DIR}/scratch"
os.environ['GCM_REPO_DIR'] = GCM_REPO_DIR = BASE_DIR+"/gcm_aws"
os.environ['HIGH_RES_TOPO_DIR'] = HIGH_RES_TOPO_DIR = f'{USER_INPUT_DIR}/topo_files/high_res'
os.environ['NCL_SCRIPTS'] = f'{os.environ["GCM_REPO_DIR"]}/scripts/ncl'
os.environ['RAW_TOPO_DIR'] = RAW_TOPO_DIR = f'{USER_INPUT_DIR}/paleodem_raw'

#CESM params
RUN_DIR=f"{ROOT_DIR}/cesm"
CESM_DIR=f"{USER_DIR}/my_cesm"
CESM_SCRIPTS=f"{CESM_DIR}/cime/scripts"
os.environ["PATH"]+=f":{CESM_SCRIPTS}"

os.environ['CIME_OUT_DIR'] = f"{SCRATCH_DIR}/archive"
os.environ['CESM_INPUT_DATA_DIR'] = CESM_INPUT_DATA_DIR = f"{RUN_DIR}/inputdata"
os.environ['CESM_REPO_DIR'] = CESM_REPO_DIR = BASE_DIR+"/cesm_aws"

os.environ['CESM_TOPO_DIR'] = CESM_TOPO_DIR = f'{USER_INPUT_DIR}/topo_files/cesm'
os.environ['CESM_CO2_DIR'] = CESM_CO2_DIR = f'{USER_INPUT_DIR}/co2_files'
os.environ['CESM_SOLAR_DIR'] = CESM_SOLAR_DIR = f'{USER_INPUT_DIR}/solar_files'
os.environ['CESM_LANDFRAC_DIR'] = CESM_SOLAR_DIR = f'{USER_INPUT_DIR}/landfrac_files'
os.environ['CESM_OCEANFRAC_DIR'] = CESM_OCEANFRAC_DIR = f'{USER_INPUT_DIR}/oceanfrac_files'
os.environ['CESM_LANDPLANT_DIR'] = CESM_LANDPLANT_DIR = f'{USER_INPUT_DIR}/landplant_files'
os.environ['CESM_MAPPING_DIR'] = CESM_MAPPING_DIR = f'{USER_INPUT_DIR}/mapping_files'
os.environ['REMAPPED_LAND_DIR'] = REMAPPED_LAND_DIR = f'{USER_INPUT_DIR}/remapped_land_files'

os.environ['CESM_MAP_FILE'] = f'{CESM_MAPPING_DIR}/map_fv1.9x2.5_to_gx1v6_tmp.nc'

os.environ['T42_TOPO_FILE'] = T42_TOPO_FILE = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/T42_nc3000_Co060_Fi001_PF_nullRR_Nsw042_20180111.nc'
os.environ['f19_TOPO_FILE'] = f19_TOPO_FILE = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/fv_1.9x2.5_nc3000_Nsw084_Nrs016_Co120_Fi001_ZR_GRNL_031819.nc'
os.environ['f09_TOPO_FILE'] = f09_TOPO_FILE = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
os.environ['ORIG_TOPO_FILE'] = ORIG_TOPO_FILE = f'{CESM_INPUT_DATA_DIR}/atm/cam/topo/{f19_TOPO_FILE}'

os.environ['ORIG_CESM_LANDFRAC_FILE'] = ORIG_LANDFRAC_FILE = f'{CESM_INPUT_DATA_DIR}/share/domains/domain.lnd.fv1.9x2.5_gx1v6.090206.nc'
os.environ['ORIG_CESM_OCEANFRAC_FILE'] = ORIG_OCEANFRAC_FILE = f'{CESM_INPUT_DATA_DIR}/share/domains/domain.ocn.gx1v6.090206.nc'
os.environ['ORIG_CESM_CO2_FILE'] = ORIG_CESM_CO2_FILE = f'{os.environ["CESM_INPUT_DATA_DIR"]}/atm/cam/inic/gaus/cami_0000-01-01_64x128_L30_c090102.nc'
os.environ['ORIG_CESM_SOLAR_FILE'] = ORIG_CESM_SOLAR_FILE = f'{CESM_INPUT_DATA_DIR}/atm/cam/solar/SolarForcingCMIP6piControl_c160921.nc'
os.environ['ORIG_CESM_OCEANMASK_FILE'] = ORIG_CESM_OCEANMASK_FILE = f'{CESM_INPUT_DATA_DIR}/ocn/pop/gx1v6/grid/region_mask_20090205.ieeei4'
os.environ['ORIG_CESM_OCEANTOPO_FILE'] = ORIG_CESM_OCEANTOPO_FILE = f'{CESM_INPUT_DATA_DIR}/ocn/pop/gx1v6/grid/topography_20090204.ieeei4'
os.environ['ORIG_CESM_LANDPLANT_FILE'] = ORIG_CESM_LANDPLANT_FILE = f'{CESM_INPUT_DATA_DIR}/lnd/clm2/surfdata_map/release-clm5.0.18/surfdata_1.9x2.5_hist_16pfts_Irrig_CMIP6_simyr1850_c190304.nc'
#os.environ['ORIG_INIT_OCEAN_FILE'] = ORIG_INIT_OCEAN_FILE = f'{CESM_INPUT_DATA_DIR}/ocn/pop/gx1v6/ic/ecosys_jan_IC_omip_POP_gx1v7_c200323.nc'
os.environ['ORIG_INIT_OCEAN_FILE'] = ORIG_INIT_OCEAN_FILE = f'{CESM_INPUT_DATA_DIR}/ocn/pop/gx1v6/ic/ts_PHC2_jan_ic_gx1v6_20090205.ieeer8'
os.environ['OCEAN_REGION_MASK_FILE'] = OCEAN_REGION_MASK_FILE = f'{CESM_OCEANFRAC_DIR}/gx1v6_region_ids'
os.environ['ORIG_DOCN_DOMAIN_FILE'] = ORIG_DOCN_DOMAIN_FILE = f'{CESM_INPUT_DATA_DIR}/ocn/docn7/domain.ocn.1x1.111007_bkup.nc'
os.environ['ORIG_DOCN_SOM_FILE'] = ORIG_DOCN_SOM_FILE = f'{CESM_INPUT_DATA_DIR}/ocn/docn7/SOM/pop_frc.gx1v6.100513.nc'

#ISCA params
os.environ['ISCA_TOPO_DIR'] = ISCA_TOPO_DIR = f'{USER_INPUT_DIR}/topo_files/isca'
os.environ['ISCA_REPO_DIR'] = ISCA_REPO_DIR = BASE_DIR+"/isca_aws"
ISCA_DIR=f"{USER_DIR}/isca"
os.environ['GFDL_BASE'] = GFDL_BASE = ISCA_DIR
os.environ['GFDL_ENV'] = GFDL_ENV = 'aws'
os.environ['GFDL_WORK'] = GFDL_WORK = f"{USER_DIR}/gfdl_work"
os.environ['GFDL_DATA'] = GFDL_DATA = f"{USER_DIR}/gfdl_data"
os.environ['RAW_ISCA_CO2_DIR'] = RAW_CO2_DIR = os.path.join(GFDL_BASE,'exp/test_cases/variable_co2_concentration/input')
os.environ['ISCA_CO2_DIR'] = CO2_DIR = f'{USER_DIR}/inputdata/co2_files/isca'

#make sure directories exist
os.system(f'mkdir -p {os.environ["HIGH_RES_TOPO_DIR"]}')
os.system(f'mkdir -p {os.environ["CESM_TOPO_DIR"]}')
os.system(f'mkdir -p {os.environ["ISCA_TOPO_DIR"]}')
os.system(f'mkdir -p {os.environ["CESM_CO2_DIR"]}')
os.system(f'mkdir -p {os.environ["CESM_SOLAR_DIR"]}')
os.system(f'mkdir -p {os.environ["CESM_LANDFRAC_DIR"]}')
os.system(f'mkdir -p {os.environ["CESM_OCEANFRAC_DIR"]}')
os.system(f'mkdir -p {os.environ["CESM_LANDPLANT_DIR"]}')
os.system(f'mkdir -p {os.environ["HIGH_RES_TOPO_DIR"]}')
os.system(f'mkdir -p {os.environ["CESM_MAPPING_DIR"]}')
os.system(f'mkdir -p {os.environ["REMAPPED_LAND_DIR"]}')
