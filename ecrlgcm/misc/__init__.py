import ecrlgcm.environment

import logging
import glob
import os
from sys import stdout
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(stdout)
sh.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

def get_base_topofile(res):
    if res.split('_')[0][0:3]=='T42':
        os.environ['ORIG_TOPO_FILE']=os.environ['T42_TOPO_FILE']
    if res.split('_')[0][0:3]=='f19':
        os.environ['ORIG_TOPO_FILE']=os.environ['f19_TOPO_FILE']
    if res.split('_')[0][0:3]=='f09':
        os.environ['ORIG_TOPO_FILE']=os.environ['f09_TOPO_FILE']
    return os.environ['ORIG_TOPO_FILE']

def get_logger():
    return logger

land_years = glob.glob(os.environ["RAW_TOPO_DIR"]+'/Map*.nc')
land_years = sorted([float(l.strip('Ma.nc').split('_')[-1]) for l in land_years])
land_years = [int(x) if int(x)==float(x) else float(x) for x in land_years]
min_land_year = int(min(land_years))
max_land_year = int(max(land_years))

def land_year_range(arg):
    try:
        if int(arg)==float(arg):
            f = int(arg)
    except:    
        try:
            f = float(arg)
        except:
            raise argparse.ArgumentTypeError('land_year must be float or integer')
    
    if f < min_land_year or f > max_land_year:
        raise argparse.ArgumentTypeError(f'land_year must be < {max_land_year} and > {min_land_year}')
    return f

def none_or_str(arg):
    if arg == 'None':
        return None
    return arg

def sig_round(number,figs):
    return float('%s' % float(f'%.{figs}g' % number))

def sliding_std(data,dx=3,dy=3):
    tmp = np.array(data)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            i_min = max(0,i-dx//2)
            i_max = min(tmp.shape[0],i+dx//2)
            j_min = max(0,j-dx//2)
            j_max = min(tmp.shape[1],j+dx//2)
            tmp[i,j] = np.std(tmp[i_min:i_max+1,j_min:j_max+1])
    return tmp

def overlap_fraction(inlat,inlon,outlat,outlon,landmask):
    tmp = np.zeros((length(outlat),length(outlon)))
