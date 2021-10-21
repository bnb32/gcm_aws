import ecrlgcm.environment
from ecrlgcm.data import co2_series, ecc_series, obl_series
from ecrlgcm.experiment import Experiment
from ecrlgcm.misc import land_years, get_logger, sliding_std, get_base_topofile

import os
import netCDF4 as nc
import numpy as np
import glob
import xarray as xr
import xesmf as xe
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

logger = get_logger()

def mila_cycle(Amin=0,Amax=0,Acurr=0,T=0,land_year=0):

    w = 2*np.pi/T
    A = (Amax-Amin)/2.0
    A0 = (Amax+Amin)/2.0
    phase = np.arcsin((Acurr-A0)/A)
    return A0 + A*np.sin(w*land_year + phase)

def eccentricity(land_year): 

    #return mila_cycle(Amin=0.005,Amax=0.057,Acurr=0.0167,T=0.092,land_year=-land_year)
    return interpolate_ecc(land_year)

def obliquity(land_year):

    #return mila_cycle(Amin=22.1,Amax=24.5,Acurr=23.4,T=0.041,land_year=-land_year)
    return interpolate_obl(land_year)

def solar_fraction(land_year):
    #assuming years prior to current era is expressed as a positive value
    time = -float(land_year)/4700.0
    return 1.0/(1-0.4*time)

def solar_constant(land_year):
    #assuming years prior to current era is expressed as a positive value
    return 1370.0*solar_fraction(land_year)

def interp(a,b,dt):
    return a*(1-dt)+dt*b

def interpolate_series(land_year,series):

    year = float(land_year)

    keys = sorted(series)

    if land_year in keys:
        return series[keys[keys.index(land_year)]]

    if year <= keys[0]:
        return series[keys[0]]

    if year >= keys[-1]:
        return series[keys[-1]]

    for i in range(len(keys)):
        if keys[i] <= year <= keys[i+1]:
            return interp(series[keys[i]],
                          series[keys[i+1]],
                          (year-keys[i])/(keys[i+1]-keys[i]))

def interpolate_co2(land_year):
    return interpolate_series(land_year,co2_series)

def interpolate_ecc(land_year):
    return interpolate_series(land_year,ecc_series)

def interpolate_obl(land_year):
    return interpolate_series(land_year,obl_series)

def adjust_co2(multiplier=1,land_year=0,co2_value=None,outfile=None):
    
    ecrlexp = Experiment(multiplier=multiplier,land_year=land_year,co2_value=co2_value)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    co2_path = os.environ['RAW_CO2_DIR']
    input_dir = os.environ['CO2_DIR']
    new_file = os.path.join(input_dir,f'{ecrlexp.co2_file}')

    os.system(f'mkdir -p {input_dir}')
    os.system(f'rm -f {new_file}')
    os.system(f'cp {co2_path}/co2.nc {new_file}')
    filename = new_file
    ncfile = nc.Dataset(filename,'r+')
    co2 = ncfile.variables['co2']
        
    if co2_value is None:    
        co2[:,:,:,:] = float(multiplier)*interpolate_co2(land_year)
    else:    
        co2[:,:,:,:] = float(co2_value)

    ncfile.variables['co2'][:,:,:,:] = co2[:,:,:,:]
    ncfile.close()

def adjust_continents(basefile='',outfile='',land_year=0,sea_level=0,max_depth=0):
    land = interpolate_land(land_year)
    ds_out = regrid_continent_data(land,basefile=basefile,
                                   sea_level=sea_level,
                                   max_depth=max_depth)

    os.system(f'rm -f {out_file}')
    ds_out.to_netcdf(out_file)
    print(f'{out_file}')

def regrid_continent_maps(remap_file,basefile='',outfile='',sea_level=0,max_depth=0):
    land_year = f'{remap_file.strip(".nc").split("_")[-1]}'
    land = xr.open_mfdataset(remap_file)
    ds_out = regrid_continent_data(land,basefile=basefile,
                                   sea_level=sea_level,
                                   max_depth=max_depth)

    os.system(f'rm -f {out_file}')
    ds_out.to_netcdf(out_file)
    print(f'{out_file}')

def regrid_continent_data(land,basefile='',sea_level=0,max_depth=0):

    base = xr.open_mfdataset(basefile)

    if all(-np.pi < l < np.pi for l in base['lat'].values):
        lats = 180.0/np.pi * base['lat'].values
    else:
        lats = base['lat'].values
    if all(-2*np.pi < l < 2*np.pi for l in base['lon'].values):
        lons = 180.0/np.pi * base['lon'].values
    else:
        lons = base['lon'].values

    if len(land['lat'].shape)==2:
        raw_lats = land['lat'][:,0].values
        raw_lons = land['lon'][0,:].values
    else:
        raw_lats = land['lat'].values
        raw_lons = land['lon'].values

    raw_landmask = np.array(land['z'].values > sea_level, dtype=float)
    
    ds_out = xr.Dataset({'lat': (['lat'], lats),
                         'lon': (['lon'], lons)})

    logger.info('Regridding continent data')
    regridder = xe.Regridder(land, ds_out, 'bilinear')
    ds_out = regridder(land)
    
    logger.info('Calculating landfrac_dict')
    landfrac_dict = overlap_fraction(raw_lats,raw_lons,lats,lons,raw_landmask)
    landfrac = get_landfrac(shape=ds_out['z'].shape,landfrac_dict=landfrac_dict)
    landmask = np.array(landfrac>0,dtype=np.int32)

    landmask[landmask>1]=1
    landfrac[landfrac>1]=1
    landmask[landmask<0]=0
    landfrac[landfrac<0]=0
    
    tmp_lfrac = landfrac
    tmp_lmask = landmask

    '''
    for i in range(landfrac.shape[0]):
        for j in range(landfrac.shape[1]):
            if ((ds_out['lat'].values[i] > 20.0) or
                (ds_out['lat'].values[i] < -20.0)):
                landfrac[i,j] = tmp_lfrac[i,j]
                landmask[i,j] = tmp_lmask[i,j]
    '''

    ds_out['landfrac'] = (ds_out['z'].dims,landfrac)
    ds_out['landmask'] = (ds_out['z'].dims,landmask.astype(np.int32))
    
    ds_out['oceanfrac'] = (ds_out['z'].dims,1-ds_out['landfrac'].values)
    ds_out['oceanmask'] = (ds_out['z'].dims,np.array(1-ds_out['landmask'].values,dtype=np.int32))

    height = ds_out['z'].values
    depth = ds_out['z'].values
    height[height<-max_depth] = -max_depth
    depth[depth>sea_level] = 0
    ds_out['z'] = (ds_out['z'].dims,height)
    ds_out['depth'] = (ds_out['z'].dims,depth)
    ds_out['PHIS'] = (ds_out['z'].dims,9.8*ds_out['z'].values)
    ds_out = ds_out.rename({'z':'zsurf'})
    ds_out = ds_out.fillna(0)
    return ds_out
    
def interpolate_land(land_year):
    year = float(land_year)

    keys = sorted(land_years)

    if land_year in keys:
        ds_out = get_original_map_data(keys[keys.index(land_year)])

    if year <= keys[0]:
        ds_out = get_original_map_data(keys[0])

    if year >= keys[-1]:
        ds_out = get_original_map_data(keys[-1])

    for i in range(len(keys)):
        if keys[i] <= year <= keys[i+1]:
            ds_out = get_original_map_data(keys[i])
            tmp = interp(get_original_map_data(keys[i])['z'].values,
                         get_original_map_data(keys[i+1])['z'].values,
                         (year-keys[i])/(keys[i+1]-keys[i]))
            ds_out['z'] = (ds_out['z'].dims,tmp)
    
    ds_out = ds_out.rename({'latitude':'lat','longitude':'lon'})
    regridder = xe.Regridder(ds_out,xe.util.grid_global(0.5, 0.5),'bilinear')
    return regridder(ds_out)
    return ds_out

def get_original_map_data(land_year):
    land_year = str(land_year)
    file = glob.glob(os.environ["RAW_TOPO_DIR"]+f'/Map*_{land_year}Ma.nc')
    land = xr.open_mfdataset(file)
    return land

def zonal_band_anomaly_squared_distance(y,y0):
    return (y-y0)**2

def meridional_band_anomaly_squared_distance(r,x,x0,y):
    ymin=np.sqrt(r)+1
    ymax=90-ymin-1
    if ymin<y<ymax: return (x-x0)**2
    elif y<=ymin: return (x-x0)**2+(y-ymin)**2
    elif y>=ymax: return (x-x0)**2+(y-ymax)**2

def disk_anomaly_squared_distance(x,x0,y,y0):
    return (x-x0)**2+(y-y0)**2

def anomaly_smoothing(max_val,d,r):
    if d<=r:
        return max_val*(r-d)/r#*np.exp(-dd/r)
    else:
        return 0

def fill_poles(data):
    tmp = data['land_mask'].values
    for i in range(len(data['lat'].values)):
        for j in range(len(data['lon'].values)):
            if np.abs(data['lat'].values[i]-90.0)<1:
                tmp[i,j]=1.0
    return tmp            

def modify_input_files(cesmexp,sea_level=0,max_depth=0,remap=True):
    
    land_year = cesmexp.land_year
    
    #landfrac file
    if not os.path.exists(cesmexp.landfrac_file) or remap:
        logger.info('Modifying landfrac_file')
        landfrac_data = xr.open_mfdataset(os.environ['ORIG_CESM_LANDFRAC_FILE'])
        
        raw_topo_data,landfrac_dict = get_landfrac_dict(orig_data=landfrac_data,land_year=land_year,sea_level=sea_level)
        landfrac = get_landfrac(shape=landfrac_data['mask'].shape,landfrac_dict=landfrac_dict)
        #landmask = get_landmask(shape=fin['mask'].shape,landfrac_dict=landfrac_dict)
        landmask = np.array(landfrac>0.5,dtype=float)
        #landfrac = landmask
        landfrac[landfrac<0]=0
        landfrac[landfrac>1]=1
        landmask[landmask<0]=0
        landmask[landmask>1]=1
        
        #landfrac_data['mask'] = landfrac_data['LANDMASK'] = (landfrac_data['mask'].dims,landmask.astype(landfrac_data['mask'].dtype))
        #landfrac_data['frac'] = (landfrac_data['frac'].dims,landfrac)
        #landfrac_data['frac'] = (landfrac_data['frac'].dims,landmask)

        landfrac_data.to_netcdf(cesmexp.landfrac_file)
        logger.info(f'Saved landfrac_file: {cesmexp.landfrac_file}')

    base_topofile = get_base_topofile(cesmexp.res)
    if not os.path.exists(cesmexp.topo_file) or remap:
        logger.info('Modifying topo_file')
        data = regrid_continent_data(raw_topo_data,
                                     basefile=base_topofile,
                                     sea_level=sea_level,
                                     max_depth=max_depth)
    #topo file
        topo_data=xr.open_mfdataset(base_topofile)
        topo_data['PHIS'] = (data['PHIS'].dims,data['PHIS'].values)
        topo_data['landmask'] = (data['landmask'].dims,landfrac_data['mask'].values)
        topo_data['LANDFRAC'] = (data['landfrac'].dims,landfrac_data['frac'].values)
        topo_data['LANDM_COSLAT'] = (data['landfrac'].dims,landfrac_data['frac'].values)
        topo_data['SGH'] = (data['zsurf'].dims,sliding_std(data['zsurf'].values))
        topo_data.to_netcdf(cesmexp.topo_file)
        logger.info(f'Saved topo_file: {cesmexp.topo_file}')
    
    #oceanfrac file
    if not os.path.exists(cesmexp.oceanfrac_file) or remap:
        logger.info('Modifying oceanfrac_file')
        oceanfrac_data=xr.open_mfdataset(os.environ['ORIG_CESM_OCEANFRAC_FILE'])
        
        #raw_topo_data,landfrac_dict = get_landfrac_dict(orig_data=oceanfrac_data,land_year=land_year,sea_level=sea_level)
        #oceanfrac = get_oceanfrac(shape=oceanfrac_data['mask'].shape,landfrac_dict=landfrac_dict)

        landfrac_data = landfrac_data.rename({'yc':'lat','xc':'lon'})

        ds_out = xr.Dataset({'lat': (['nj','ni'], oceanfrac_data['yc'].values),
                             'lon': (['nj','ni'], oceanfrac_data['xc'].values)})

        logger.info('Regridding oceanfrac_file')
        regridder = xe.Regridder(topo_data, ds_out, 'patch')
        ds_out = regridder(topo_data)
        
        oceanfrac = 1-ds_out['LANDFRAC'].values
        oceanmask = 1-ds_out['landmask'].values
        
        #oceanfrac[oceanfrac<0]=0
        #oceanfrac[oceanfrac>1]=1
        #oceanmask[oceanmask<0]=0
        #oceanmask[oceanmask>1]=1
 
        #fin['frac'] = (fin['frac'].dims,oceanfrac)
        oceanfrac_data['frac'] = (oceanfrac_data['frac'].dims,oceanmask)
        oceanfrac_data['mask'] = (oceanfrac_data['mask'].dims,oceanmask.astype(oceanfrac_data['mask'].dtype))
        oceanfrac_data.fillna(0)
        
        oceanfrac_data.to_netcdf(cesmexp.oceanfrac_file)
        logger.info(f'Saved oceanfrac_file: {cesmexp.oceanfrac_file}')
    
    #co2 file
    if not os.path.exists(cesmexp.co2_file) or remap:
        logger.info('Modifying co2_file')
        co2value = interpolate_co2(land_year)
        f=xr.open_mfdataset(os.environ['ORIG_CESM_CO2_FILE'],decode_times=False)
        tmp = np.full(f['co2vmr'].shape,cesmexp.multiplier*co2value*1.0e-6)
        f['co2vmr'] = (f['co2vmr'].dims,tmp)
        f.to_netcdf(cesmexp.co2_file)
        logger.info(f'Saved co2_file: {cesmexp.co2_file}')
    
    #solar file
    if not os.path.exists(cesmexp.solar_file) or remap:
        logger.info('Modifying solar_file')
        solar_frac = solar_fraction(land_year)
        f=xr.open_mfdataset(os.environ['ORIG_CESM_SOLAR_FILE'],decode_times=False)
        f['ssi'] = (f['ssi'].dims,solar_frac*f['ssi'].values)
        f.to_netcdf(cesmexp.solar_file)
        logger.info(f'Saved solar_file: {cesmexp.solar_file}')

def modify_topo_file(land_year=0,infile='',outfile='',sea_level=0,max_depth=0):
    
    data = regrid_continent_data(interpolate_land(land_year),
                                 basefile=infile,
                                 sea_level=sea_level,
                                 max_depth=max_depth)
    f=xr.open_mfdataset(infile)
    f['PHIS'] = (data['PHIS'].dims,data['PHIS'].values)
    f['landmask'] = (data['land_mask'].dims,data['land_mask'].values)
    f['oceanmask'] = (data['ocean_mask'].dims,data['ocean_mask'].values)
    f['LANDFRAC'] = (data['landfrac'].dims,data['landfrac'].values)
    f['OCEANFRAC'] = (data['oceanfrac'].dims,data['oceanfrac'].values)
    f['LANDM_COSLAT'] = (data['landfrac'].dims,data['landfrac'].values)
    f['SGH'] = (data['zsurf'].dims,sliding_std(data['zsurf'].values))
    f.to_netcdf(outfile)
    logger.info(f'Saved topo_file: {outfile}')
    return f

def modify_variable(source='',infile='',outfile='',srcvar='',outvar=''):
    fsrc=xr.open_mfdataset(source)
    fin=xr.open_mfdataset(infile)
    fin[outvar] = (fin[outvar].dims,fsrc[srcvar].values)
    fin.to_netcdf(outfile)
    logger.info(f'Saved file: {outfile}')
    return fin

def modify_landfrac_file(topo_file='',infile='',outfile='',land_year=None,sea_level=0):
    fsrc=xr.open_mfdataset(topo_file)
    fin=xr.open_mfdataset(infile)
    
    landfrac_dict = get_landfrac_dict(orig_data=fin,source_data=fsrc,land_year=land_year,sea_level=sea_level)
    landfrac = get_landfrac(shape=fin['mask'].shape,landfrac_dict=landfrac_dict)
    landmask = np.array(landfrac>0,dtype=np.int32)
    
    landmask[landmask>1]=1
    landfrac[landfrac>1]=1
    landmask[landmask<0]=0
    landfrac[landfrac<0]=0

    fin['mask'] = (fin['mask'].dims,landmask)
    fin['frac'] = (fin['frac'].dims,landfrac)
    fin.fillna(0)

    fin.to_netcdf(outfile)
    logger.info(f'Saved landfrac_file: {outfile}')
    return fin

def modify_oceanfrac_file(ocn_file='',infile='',outfile='',land_year=None,sea_level=0):
    fsrc=xr.open_mfdataset(ocn_file)
    fin=xr.open_mfdataset(infile)
    
    fsrc = fsrc.rename({'yc':'lat','xc':'lon'})

    ds_out = xr.Dataset({'lat': (['nj','ni'], fin['yc'].values),
                         'lon': (['nj','ni'], fin['xc'].values)})

    logger.info('Regridding oceanfrac_file')
    regridder = xe.Regridder(fsrc, ds_out, 'bilinear')
    ds_out = regridder(fsrc)

    oceanfrac = 1-ds_out['frac'].values
    oceanmask = 1-ds_out['mask'].values
    oceanfrac[oceanfrac<0]=0
    oceanfrac[oceanfrac>1]=1
    oceanmask[oceanmask<0]=0
    oceanmask[oceanmask>1]=1
 
    fin['frac'] = (fin['frac'].dims,oceanfrac)
    fin['mask'] = (fin['mask'].dims,oceanmask)
    fin.fillna(0)

    ''' 
    landfrac_dict = get_landfrac_dict(orig_data=fin,source_data=fsrc,land_year=land_year,sea_level=sea_level)
    oceanmask = get_oceanmask(shape=fin['mask'].shape,landfrac_dict=landfrac_dict)
    oceanfrac = get_oceanfrac(shape=fin['mask'].shape,landfrac_dict=landfrac_dict)
    oceanfrac[oceanfrac<0]=0
    oceanfrac[oceanfrac>1]=1
    oceanmask[oceanmask<0]=0
    oceanmask[oceanmask>1]=1
    
    fin['frac'] = (fin['frac'].dims,oceanfrac)
    fin['mask'] = (fin['mask'].dims,oceanmask)
    '''

    fin.to_netcdf(outfile)
    logger.info(f'Saved oceanfrac_file: {outfile}')
    return fin

def modify_co2_file(land_year=0,multiplier=1,infile='',outfile=''):
    
    co2value = interpolate_co2(land_year)
    f=xr.open_mfdataset(infile,decode_times=False)
    tmp = np.full(f['co2vmr'].shape,multiplier*co2value*1.0e-6)
    f['co2vmr'] = (f['co2vmr'].dims,tmp)
    f.to_netcdf(outfile)
    logger.info(f'Saved co2_file: {outfile}')

def modify_solar_file(land_year=0,infile='',outfile=''):
    
    solar_frac = solar_fraction(land_year)
    f=xr.open_mfdataset(infile,decode_times=False)
    f['ssi'] = (f['ssi'].dims,solar_frac*f['ssi'].values)
    f.to_netcdf(outfile)
    logger.info(f'Saved solar_file: {outfile}')

def cell_overlap(lats,lons,lat,lon,dx,dy,landmask):
    
    lons_in = ((lon-dx/2 < lons) & (lons < lon+dx/2))
    lats_in = ((lat-dy/2 < lats) & (lats < lat+dy/2))

    lat_count = lats_in.sum()
    lon_count = lons_in.sum()
    total_count = lat_count*lon_count
    mask_count = landmask[np.outer(lats_in,lons_in)].sum()

    return {'mask_count':mask_count,'total_count':total_count}

def overlap_fraction(inlat,inlon,outlat,outlon,landmask):

    tmp = {}

    inlon = [x+360.0 if x < 0 else x for x in inlon]
    outlon = [x+360.0 if x < 0 else x for x in outlon]
    dx = outlon[1]-outlon[0]
    dy = outlat[1]-outlat[0]

    for i in tqdm(range(len(outlat))):
        for j in range(len(outlon)):
            lat = outlat[i]
            lon = outlon[j]
            key = (i,j)
            tmp[key] = cell_overlap(inlat,inlon,lat,lon,dx,dy,landmask)
    
    '''
    for i in tqdm(range(len(inlat))):
        for j in range(len(inlon)):
            lat = inlat[i]
            lon = inlon[j]
            lat_idx = np.argmin(np.abs(outlat-lat))
            lon_idx = np.argmin(np.abs(outlon-lon))
            
            key = (lat_idx,lon_idx)#outlat[i],outlon[j])
            if key not in tmp:
                tmp[key] = {'mask_count':landmask[i,j],'total_count':1.0}
            else:
                tmp[key]['mask_count']+=landmask[i,j]
                tmp[key]['total_count']+=1.0
    '''
    return tmp        

def get_landfrac_dict(orig_data=None,source_data=None,land_year=None,sea_level=0):

    landfrac_dict = {}
    if land_year is not None:
        source_data = interpolate_land(land_year)
    raw_landmask = np.array(source_data['z'].values > sea_level, dtype=float)
    logger.info('Calculating landfrac_dict')
    outlat = orig_data['yc'].values[:,0]
    outlon = orig_data['xc'].values[0,:]

    if len(source_data['lat'].shape)==2:
        inlat = source_data['lat'][:,0].values
        inlon = source_data['lon'][0,:].values
    else:
        inlat = source_data['lat'].values
        inlon = source_data['lon'].values
    
    return source_data,overlap_fraction(inlat,inlon,outlat,outlon,raw_landmask)

def get_landfrac(shape=None,landfrac_dict=None):

    landfrac = np.zeros(shape)
    logger.info('Calculating landfrac')
    for i in tqdm(range(landfrac.shape[0])):
        for j in range(landfrac.shape[1]):
            key = (i,j)
            landfrac[i,j] = landfrac_dict[key]['mask_count']/landfrac_dict[key]['total_count']
    return landfrac        

def get_oceanfrac(shape=None,landfrac_dict=None):

    oceanfrac = np.zeros(shape)
    logger.info('Calculating oceanfrac')
    for i in tqdm(range(oceanfrac.shape[0])):
        for j in range(oceanfrac.shape[1]):
            key = (i,j)
            oceanfrac[i,j] = 1-landfrac_dict[key]['mask_count']/landfrac_dict[key]['total_count']
    return oceanfrac        

def isocean(lat,lon,data,sea_level=0):
    
    lat_idx = np.argmin(np.abs(lat-data['lat'].values))
    lon_idx = np.argmin(np.abs(lon-data['lon'].values))

    if 'z' in data:
        alt_key = 'z'
    else:
        alt_key = 'PHIS'

    if data[alt_key][lat_idx,lon_idx]<=sea_level: return True
    return False

def anomaly_value(max_val,r,x,x0,y,y0,anomaly_type='disk'):
    if x>=180.0: x-=360.0
    if x0>=180.0: x0-=360.0
    if anomaly_type=='disk':
        d=disk_anomaly_squared_distance(x,x0,y,y0)
    if anomaly_type=='zonal_band':
        d=zonal_band_anomaly_squared_distance(y,y0)
    if anomaly_type=='meridional_band':
        d=meridional_band_anomaly_squared_distance(r,x,x0,y)
    if anomaly_type=='none':
        return 0
    return anomaly_smoothing(max_val,np.sqrt(d),np.sqrt(r))

def inject_anomaly(basefile='',anomaly_type='disk',
                   variable='PHIS',exp_type='dry_hs',
                   max_anomaly=0,squared_radius=1,
                   anomaly_lon=0,anomaly_lat=0,
                   use_lapse_rate=True,
                   just_surface=False):

    base = xr.open_mfdataset(basefile)
    data = base[variable].values
    lats = base['lat'].values
    lons = base['lon'].values

    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
        
            value=anomaly_value(max_anomaly,squared_radius,
                                lon,anomaly_lon,lat,anomaly_lat,
                                anomaly_type=anomaly_type)

            if exp_type=='aqua':
                data[:,i,j]+=value
            elif exp_type=='dry_hs':
                if use_lapse_rate:
                    for k in range(len(data[0,:,0,0])):
                        lapse_rate=value/(len(data[0,:,0,0])-1)
                        data[:,k,i,j]+=k*lapse_rate
                elif just_surface:
                    data[:,-1,i,j]+=value
                else:
                    data[:,:,i,j]+=value
    
    return data
