import os

USERNAME="ec2-user"
NETID="bnb32"
PROJECT_CODE="UCOR0044"
ROOT_DIR="/data/%s" %(NETID)
SCRATCH_DIR=ROOT_DIR+"/scratch"
BASE_DIR="/home/%s/environment" %(USERNAME)
ISCA_DIR=ROOT_DIR+"/isca"
os.environ['GFDL_BASE']=GFDL_BASE=ISCA_DIR

os.environ['GCM_REPO_DIR']=BASE_DIR+"/gcm_aws"

RAW_TOPO_DIR =f'/data/{NETID}/paleodem_raw'
os.environ['RAW_TOPO_DIR'] = RAW_TOPO_DIR

TOPO_DIR = os.path.join(RAW_TOPO_DIR,'land_masks/')
os.environ['TOPO_DIR'] = TOPO_DIR

os.environ['BASE_TOPO_FILE'] = os.path.join(GFDL_BASE,'input/land_masks/era_land_t42.nc')
