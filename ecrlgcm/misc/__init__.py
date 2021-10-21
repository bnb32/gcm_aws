import ecrlgcm.environment

import logging
import glob
import os
from sys import stdout
import numpy as np
from tqdm import tqdm

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

def count_between(array,min_val,max_val):
    return ((min_val <= array) & (array <= max_val)).sum()

def cell_overlap(inlat,inlon,lat,lon,lat_dx,lon_dx,landmask):

    total_count=0
    mask_count=0
    
    if lon >= 180.0: lon-=360.0

    max_i = np.argmin(np.abs(inlat-(lat-lat_dx/2)))
    min_i = np.argmin(np.abs(inlat-(lat+lat_dx/2)))
    
    min_j = np.argmin(np.abs(inlon-(lon-lon_dx/2)))
    max_j = np.argmin(np.abs(inlon-(lon+lon_dx/2)))

    for i in range(min_i,max_i+1):
        for j in range(min_j,max_j+1):
            if ((np.abs(inlat[i]-lat) < lat_dx/2) and
                (np.abs(inlon[j]-lon) < lon_dx/2)):
                total_count+=1
                if landmask[i,j] > 0:
                    mask_count+=1
    return mask_count/total_count

def overlap_fraction(inlat,inlon,outlat,outlon,landmask):

    tmp = np.zeros((len(outlat),len(outlon)))

    lat_dx = outlat[1]-outlat[0]
    lon_dx = outlon[1]-outlon[0]

    for i in tqdm(range(len(outlat))):
        for j in range(len(outlon)):
            tmp[i,j] = cell_overlap(inlat,inlon,
                                    outlat[i],outlon[j],
                                    lat_dx,lon_dx,
                                    landmask)
    return tmp        
