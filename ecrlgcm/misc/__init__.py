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

def polar_to_cartesian(radius=1,lat=0,lon=0):
     x = radius * np.cos(np.pi/180*lat) * np.cos(np.pi/180*lon)
     y = radius * np.cos(np.pi/180*lat) * np.sin(np.pi/180*lon)
     z = radius * np.sin(np.pi/180*lat)
     return {'x':x,'y':y,'z':z}

def mapping_map_to_sphere(lon, lat, radius=1):
    lon=np.array(lon, dtype=np.float64)
    lat=np.array(lat, dtype=np.float64)
    lon=np.pi/180.0*lon
    lat=np.pi/180.0*lat
    xs=radius*np.cos(lon)*np.cos(lat)
    ys=radius*np.sin(lon)*np.cos(lat)
    zs=radius*np.sin(lat)
    return xs, ys, zs

def interp(a,b,dt):
    a_tmp = a[:min((a.shape[0],b.shape[0]))]
    b_tmp = b[:a_tmp.shape[0]]
    return a_tmp*(1-dt)+dt*b_tmp
    #return a*(1-dt)+dt*b

def edit_namelists(experiment,configuration):
        
    case = f'{os.environ["CESM_REPO_DIR"]}/cases/{experiment.name}'
    nl_cam_file=f"{case}/user_nl_cam"
    nl_cpl_file=f"{case}/user_nl_cpl"
    nl_clm_file=f"{case}/user_nl_clm"
    nl_pop_file=f"{case}/user_nl_pop"
    nl_docn_file=f"{case}/user_nl_docn"
    nl_ice_file=f"{case}/user_nl_ice"
                    
    logger.info(f"**Changing namelist file: {nl_cam_file}**")
    with open(nl_cam_file,'w') as f:
        for l in configuration['atm']:
            f.write(f'{l}\n')
    f.close()
    
    logger.info(f"**Changing namelist file: {nl_cpl_file}**")
    with open(nl_cpl_file,'w') as f:
        for l in configuration['cpl']:
            f.write(f'{l}\n')
    f.close()    

    logger.info(f"**Changing namelist file: {nl_clm_file}**")
    with open(nl_clm_file,'w') as f:
        for l in configuration['lnd']:
            f.write(f'{l}\n')
    f.close()    
    
    logger.info(f"**Changing namelist file: {nl_docn_file}**")
    with open(nl_docn_file,'w') as f:
        for l in configuration['docn']:
            f.write(f'{l}\n')
    f.close()

    logger.info(f"**Changing namelist file: {nl_pop_file}**")
    with open(nl_pop_file,'w') as f:
        for l in configuration['ocn']:
            f.write(f'{l}\n')
    f.close()    

    logger.info(f"**Changing namelist file: {nl_ice_file}**")
    with open(nl_ice_file,'w') as f:
        for l in configuration['ocn']:
            f.write(f'{l}\n')
    f.close()    
    
    if configuration['change_som_stream']:
        logger.info(f"**Changing docn dom stream file**")
        with open(f'{os.environ["GCM_REPO_DIR"]}/templates/user_docn.streams.txt.som') as f:
            docn_file=f.read()
        
        os.system(f'cp {os.environ["GCM_REPO_DIR"]}/templates/user_docn.streams.txt.som {case}')
        with open(f'{case}/user_docn.streams.txt.som','w') as f:
            tmp_file = experiment.docn_som_file.split('/')[-1]
            tmp_path = experiment.docn_som_file.split('/')[:-1]
            tmp_path = '/'.join(tmp_path)
            f.write(docn_file.replace('%DOCN_SOM_FILE%',tmp_file).replace('%DOCN_SOM_DIR%',tmp_path))
    
    if configuration['change_dom_stream']:
        logger.info(f"**Changing docn dom stream file**")
        with open(f'{os.environ["GCM_REPO_DIR"]}/templates/user_docn.streams.txt.prescribed') as f:
            docn_file=f.read()
        
        os.system(f'cp {os.environ["GCM_REPO_DIR"]}/templates/user_docn.streams.txt.prescribed {case}')
        with open(f'{case}/user_docn.streams.txt.prescribed','w') as f:
            sst_file = experiment.docn_sst_file.split('/')[-1]
            sst_path = experiment.docn_sst_file.split('/')[:-1]
            sst_path = '/'.join(sst_path)
            ocnfrac_file = experiment.docn_ocnfrac_file.split('/')[-1]
            ocnfrac_path = experiment.docn_ocnfrac_file.split('/')[:-1]
            ocnfrac_path = '/'.join(ocnfrac_path)
            f.write(docn_file.replace('%DOCN_SST_FILE%',sst_file).replace('%DOCN_SST_DIR%',sst_path).replace('%DOCN_OCNFRAC_FILE%',ocnfrac_file).replace('%DOCN_OCNFRAC_DIR%',ocnfrac_path))

def get_base_topofile(res):
    tmp_res = res.replace('.','').split('_')[0]
    if '42' in tmp_res:
        os.environ['ORIG_CESM_TOPO_FILE']=os.environ['T42_TOPO_FILE']
    if '19' in tmp_res:
        os.environ['ORIG_CESM_TOPO_FILE']=os.environ['f19_TOPO_FILE']
    if '09' in tmp_res:
        os.environ['ORIG_CESM_TOPO_FILE']=os.environ['f09_TOPO_FILE']
    return os.environ['ORIG_CESM_TOPO_FILE']

def get_logger():
    return logger

land_years = glob.glob(os.environ["RAW_TOPO_DIR"]+'/Map*.nc')
land_years = sorted([float(l.strip('Ma.nc').split('_')[-1]) for l in land_years])
land_years = [int(x) if int(x)==float(x) else float(x) for x in land_years]
min_land_year = int(min(land_years))
max_land_year = int(max(land_years))

stored_years = glob.glob(os.environ["CIME_OUT_DIR"]+'/*/atm/hist/*cam.*.nc')
stored_years = sorted([float(l.split('_')[-2].strip('Ma')) for l in stored_years])
stored_years = [int(x) if int(x)==float(x) else float(x) for x in stored_years]

isca_plevels = [4.32865344, 15.54935838, 25.36458666, 39.7283521,
                59.85607174, 86.9016501, 121.79748549, 165.08883856,
                216.79319817, 276.31361584, 342.42730307, 413.35702347,
                486.91737837, 560.71355652, 632.36253604, 699.70472315,
                760.97778738, 814.93562293, 860.90496012, 898.78288051,
                928.98741429, 952.37693492, 970.15513959, 983.77727078,
                994.87004887]

cesm_plevels = [3.64346569, 7.59481965, 14.35663225, 24.61222,
                35.92325002, 43.19375008, 51.67749897, 61.52049825,
                73.75095785, 87.82123029, 103.31712663, 121.54724076,
                142.99403876, 168.22507977, 197.9080867, 232.82861896,
                273.91081676, 322.24190235, 379.10090387, 445.9925741,
                524.68717471, 609.77869481, 691.38943031, 763.40448111,
                820.85836865, 859.53476653, 887.02024892, 912.64454694,
                936.19839847, 957.48547954, 976.32540739, 992.55609512]

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

def none_or_int(arg):
    if arg == 'None':
        return None
    return int(arg)

def none_or_float(arg):
    if arg == 'None':
        return None
    return float(arg)

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
