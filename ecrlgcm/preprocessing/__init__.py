import ecrlgcm.environment
from ecrlgcm.data import co2_series, ecc_series, obl_series
from ecrlgcm.experiment import Experiment
from ecrlgcm.misc import land_years, get_logger, sliding_std, overlap_fraction

import os
import netCDF4 as nc
import numpy as np
import glob
import xarray as xr
import xesmf as xe
import warnings
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

    raw_lats = land['latitude'].values
    raw_lons = land['longitude'].values
    
    raw_landmask = np.array(land['z'].values > sea_level, dtype=float)
    raw_oceanmask = np.array(land['z'].values <= sea_level, dtype=float)

    ds_out = xr.Dataset({'lat': (['lat'], lats),
                         'lon': (['lon'], lons)})

    regridder = xe.Regridder(land, ds_out, 'bilinear')
    ds_out = regridder(land)
    
    ds_out['land_mask'] = (ds_out['z'].dims,np.array(ds_out['z'].values > sea_level, dtype=float))
    logger.info('Calculating landfrac')
    ds_out['landfrac'] = (ds_out['land_mask'].dims,overlap_fraction(raw_lats,raw_lons,lats,lons,raw_landmask))
    
    ds_out['ocean_mask'] = (ds_out['z'].dims,np.array(ds_out['z'].values <= sea_level, dtype=float))
    logger.info('Calculating oceanfrac')
    ds_out['oceanfrac'] = (ds_out['ocean_mask'].dims,overlap_fraction(raw_lats,raw_lons,lats,lons,raw_oceanmask))

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
        return get_original_map_data(keys[keys.index(land_year)])

    if year <= keys[0]:
        return get_original_map_data(keys[0])

    if year >= keys[-1]:
        return get_original_map_data(keys[-1])

    for i in range(len(keys)):
        if keys[i] <= year <= keys[i+1]:
            ds_out = get_original_map_data(keys[i])
            tmp = interp(get_original_map_data(keys[i])['z'].values,
                         get_original_map_data(keys[i+1])['z'].values,
                         (year-keys[i])/(keys[i+1]-keys[i]))
            ds_out['z'] = (ds_out['z'].dims,tmp)
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
            if np.abs(data['lat'].values[i]+90.0)<1:
                tmp[i,j]=1.0
    return tmp            

def modify_topo_file(land_year=0,infile='',outfile='',sea_level=0,max_depth=0):
    
    data = regrid_continent_data(interpolate_land(land_year),
                                 basefile=infile,
                                 sea_level=sea_level,
                                 max_depth=max_depth)
    f=xr.open_mfdataset(infile)
    f['PHIS'] = (data['PHIS'].dims,data['PHIS'].values)
    f['landmask'] = (data['land_mask'].dims,fill_poles(data))
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

def modify_landfrac_file(topo_file='',infile='',outfile=''):
    fsrc=xr.open_mfdataset(topo_file)
    fin=xr.open_mfdataset(infile)
    
    ds_out = xr.Dataset({'lat': (['nj','ni'], fin['yc'].values),
                         'lon': (['nj','ni'], fin['xc'].values)})

    regridder = xe.Regridder(fsrc, ds_out, 'bilinear')
    ds_out = regridder(fsrc)
    
    fin['mask'] = (fin['mask'].dims,np.array(ds_out['LANDFRAC'].values>0,dtype=float))
    fin['frac'] = (fin['frac'].dims,ds_out['LANDFRAC'].values)
    fin.to_netcdf(outfile)
    logger.info(f'Saved landfrac_file: {outfile}')
    return fin

def modify_oceanfrac_file(ocn_file='',infile='',outfile=''):
    fsrc=xr.open_mfdataset(ocn_file)
    fin=xr.open_mfdataset(infile)
    
    ds_out = xr.Dataset({'lat': (['nj','ni'], fin['yc'].values),
                         'lon': (['nj','ni'], fin['xc'].values)})

    regridder = xe.Regridder(fsrc, ds_out, 'bilinear')
    ds_out = regridder(fsrc)

    fin['mask'] = (fin['mask'].dims,np.array(ds_out['OCEANFRAC'].values>0,dtype=float))
    fin['frac'] = (fin['frac'].dims,ds_out['OCEANFRAC'].values)
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
