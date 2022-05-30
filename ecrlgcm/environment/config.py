"""Personal configuration file"""
import os

USERNAME = "ec2-user"
NETID = "bnb32"
PROJECT_CODE = "UCOR0044"
ROOT_DIR = "/tmp"

USER_DIR = f"{ROOT_DIR}/{NETID}"
USER_INPUT_DIR = f"{ROOT_DIR}/{NETID}/inputdata"
USER_OUTPUT_DIR = f"{ROOT_DIR}/{NETID}/outputdata"
USER_FIGS_DIR = f"{ROOT_DIR}/{NETID}/figs"
USER_ANIMS_DIR = f"{ROOT_DIR}/{NETID}/figs/anims"
BASE_DIR = f"/home/{USERNAME}/environment"
SCRATCH_DIR = f"{USER_DIR}/scratch"
GCM_REPO_DIR = BASE_DIR + "/gcm_aws"
HIGH_RES_TOPO_DIR = f'{USER_INPUT_DIR}/topo_files/high_res'
RAW_TOPO_DIR = f'{USER_INPUT_DIR}/paleodem_raw'
NCL_SCRIPTS = f'{GCM_REPO_DIR}/scripts/ncl'

# CESM params
RUN_DIR = f"{ROOT_DIR}/cesm"
CESM_DIR = f"{USER_DIR}/my_cesm"
CESM_SCRIPTS = f"{CESM_DIR}/cime/scripts"

CIME_OUT_DIR = f"{SCRATCH_DIR}/archive"
CESM_INPUT_DATA_DIR = f"{RUN_DIR}/inputdata"
INIT_CONDITIONS_DIR = f"{RUN_DIR}/inputdata/init"
CESM_REPO_DIR = BASE_DIR + "/cesm_aws"

ORIG_CESM_TOPO_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/atm/cam/topo/fv_1.9x2.5_nc3000_Nsw084_'
    'Nrs016_Co120_Fi001_ZR_GRNL_031819.nc')
ORIG_CESM_LANDFRAC_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/share/domains/domain.lnd.fv1.9x2.5_gx1v7.181205.nc')
ORIG_CESM_OCEANFRAC_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/share/domains/domain.ocn.fv1.9x2.5_gx1v7.181205.nc')
ORIG_CESM_CO2_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/atm/cam/inic/gaus/cami_0000-01-01_64x128_L30_c090102.nc')
ORIG_CESM_SOLAR_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/atm/cam/solar/SolarForcingCMIP6piControl_c160921.nc')
ORIG_CESM_OCEANMASK_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/ocn/pop/gx1v6/grid/region_mask_20090205.ieeei4')
ORIG_CESM_OCEANTOPO_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/ocn/pop/gx1v6/grid/topography_20090204.ieeei4')
ORIG_CESM_LANDPLANT_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/lnd/clm2/surfdata_map/release-clm5.0.18/'
    'surfdata_1.9x2.5_hist_16pfts_Irrig_CMIP6_simyr1850_c190304.nc')
ORIG_INIT_OCEAN_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/ocn/pop/gx1v6/ic/ts_PHC2_jan_ic_gx1v6_20090205.ieeer8')
ORIG_INIT_ATM_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/cesm2_init/b.e20.B1850.f19_g17.release_cesm2_1_0.020/'
    '0301-01-01/b.e20.B1850.f19_g17.release_cesm2_1_0.020'
    '.cam.i.0301-01-01-00000.nc')
ORIG_INIT_LND_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/lnd/clm2/initdata_map/clmi.BHIST.2000-01-01.0.9x1.25_gx1v7_'
    'simyr2000_c200728.nc')
ORIG_DOCN_DOMAIN_FILE = os.path.join(
    CESM_INPUT_DATA_DIR, '/ocn/docn7/domain.ocn.1x1.111007.nc')
ORIG_DOCN_SOM_FILE = os.path.join(
    CESM_INPUT_DATA_DIR, '/ocn/docn7/SOM/pop_frc.f19.nc')
ORIG_DOCN_SST_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/atm/cam/sst/sst_HadOIBl_bc_1.9x2.5_clim_pi_c101028.nc')
ORIG_DOCN_OCNFRAC_FILE = os.path.join(
    CESM_INPUT_DATA_DIR,
    '/atm/cam/ocnfrac/domain.camocn.1.9x2.5_gx1v6_090403.nc')

ISCA_DIR = f"{USER_DIR}/isca"
ISCA_TOPO_DIR = f'{USER_INPUT_DIR}/topo_files/isca'
ISCA_REPO_DIR = BASE_DIR + "/isca_aws"
GFDL_BASE = ISCA_DIR
GFDL_ENV = 'aws'
GFDL_WORK = f"{USER_DIR}/gfdl_work"
GFDL_DATA = f"{USER_DIR}/gfdl_data"
RAW_ISCA_CO2_DIR = os.path.join(
    GFDL_BASE, 'exp/test_cases/variable_co2_concentration/input')
ORIG_ISCA_CO2_FILE = f'{RAW_ISCA_CO2_DIR}/co2.nc'
ISCA_CO2_DIR = f'{USER_DIR}/inputdata/co2_files/isca'
ORIG_ISCA_TOPO_FILE = os.path.join(GFDL_BASE,
                                   'input/land_masks/era_land_t42.nc')

CESM_TOPO_DIR = f'{USER_INPUT_DIR}/topo_files/cesm'
CESM_CO2_DIR = f'{USER_INPUT_DIR}/co2_files'
CESM_SOLAR_DIR = f'{USER_INPUT_DIR}/solar_files'
CESM_LANDFRAC_DIR = f'{USER_INPUT_DIR}/landfrac_files'
CESM_OCEANFRAC_DIR = f'{USER_INPUT_DIR}/oceanfrac_files'
CESM_LANDPLANT_DIR = f'{USER_INPUT_DIR}/landplant_files'
CESM_MAPPING_DIR = f'{USER_INPUT_DIR}/mapping_files'
REMAPPED_LAND_DIR = f'{USER_INPUT_DIR}/remapped_land_files'
