import ecrlgcm.environment
from ecrlgcm.misc import polar_to_cartesian, mapping_map_to_sphere, sig_round, interp
from ecrlgcm.misc import land_years, get_logger, stored_years, cesm_plevels, isca_plevels
from ecrlgcm.preprocessing import solar_constant
from ecrlgcm.experiment import Experiment

import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.animation as anim
from IPython.display import HTML
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import unregister_cmap
from metpy.calc import smooth_n_point
import plotly.offline as po
import plotly.graph_objs as go
import plotly.io as pio
import os
import time

logger = get_logger()

variable_dictionary = {}
reference_data = Experiment(land_year=stored_years[0]).sim_data()
for e in list(reference_data):
    if 'lat' in reference_data[e].dims and 'lon' in reference_data[e].dims:
        try:
            variable_dictionary[e] = reference_data[e].long_name
        except:
            pass

def define_noaa_colormap():
    color_array = [
    (0.00000, 0.06275, 0.02353, 0.70588),
    (0.15380, 0.06275, 0.02353, 0.70588),
    (0.15380, 0.00000, 0.05490, 0.79608),
    (0.25640, 0.00000, 0.05490, 0.79608),
    (0.25640, 0.04706, 0.26667, 0.90588),
    (0.35900, 0.04706, 0.26667, 0.90588),
    (0.35900, 0.08235, 0.61961, 0.98824),
    (0.46150, 0.08235, 0.61961, 0.98824),
    (0.46150, 0.25490, 0.78431, 1.00000),
    (0.51280, 0.25490, 0.78431, 1.00000),
    (0.51280, 0.36863, 0.87451, 1.00000),
    (0.53850, 0.36863, 0.87451, 1.00000),
    (0.53850, 0.54118, 0.89020, 1.00000),
    (0.56410, 0.54118, 0.89020, 1.00000),
    (0.56410, 0.20000, 0.40000, 0.00000),
    (0.56920, 0.00000, 0.20000, 0.40000),
    (0.56920, 0.20000, 0.80000, 0.40000),
    (0.58970, 0.20000, 0.80000, 0.40000),
    (0.58970, 1.00000, 0.86275, 0.72549),
    (0.61540, 1.00000, 0.86275, 0.72549),
    (0.61540, 0.95294, 0.79216, 0.53725),
    (0.66670, 0.95294, 0.79216, 0.53725),
    (0.66670, 0.85098, 0.65098, 0.15294),
    (0.76920, 0.85098, 0.65098, 0.15294),
    (0.76920, 0.62353, 0.48235, 0.05098),
    (0.87180, 0.62353, 0.48235, 0.05098),
    (0.87180, 0.69804, 0.46275, 0.46275),
    (1.00000, 0.69804, 0.46275, 0.46275)]

    tmp=[]
    for i in range(len(color_array)):
        entry = color_array[i]
        tmp.append((entry[1],entry[2],entry[3],entry[0]))
    color_array = list(plt.get_cmap('GnBu_r')(range(64)))[::64//(len(tmp)//4)+1]+tmp[len(tmp)//4:]    
    map_object = LinearSegmentedColormap.from_list(name='custom_noaa',colors=color_array)
    
    # register this new colormap with matplotlib
    unregister_cmap('custom_noaa')
    plt.register_cmap(cmap=map_object)

def hires_interp(land_year,stored_years=stored_years):
    
    interp_file = f'{os.environ["USER_OUTPUT_DIR"]}/hires_{land_year}.nc'
    if os.path.exists(interp_file):
        return xr.open_mfdataset(interp_file)
    
    year = float(land_year)
    keys = sorted(stored_years)
    basefile = xr.open_mfdataset(Experiment(land_year=0).high_res_file)
    ds_out = xr.Dataset({'lat': (['lat'], basefile['lat'].values),
                         'lon': (['lon'], basefile['lon'].values)})
    
    if land_year in keys:
        exp = Experiment(land_year=keys[keys.index(land_year)])
        ds_out['z'] = (('lat','lon'),exp.hires()['z'].values)

    elif year <= keys[0]:
        exp = Experiment(land_year=keys[0])
        ds_out['z'] = (('lat','lon'),exp.hires()['z'].values)

    elif year >= keys[-1]:
        exp = Experiment(land_year=keys[-1])
        ds_out['z'] = (('lat','lon'),exp.hires()['z'].values)
    
    else:
        for i in range(len(keys)):
            if keys[i] <= year <= keys[i+1]:
                exp0 = Experiment(land_year=keys[i])
                exp1 = Experiment(land_year=keys[i+1])
                tmp = interp(exp0.hires()['z'].values,
                             exp1.hires()['z'].values,
                             (year-keys[i])/(keys[i+1]-keys[i]))
                ds_out['z'] = (('lat','lon'),tmp)
    
    logger.info(f'Regridding hires_interp, year: {land_year}')
    ds_new = xr.Dataset({'lat': (['lat'], basefile['lat'].values[:-1:3]),
                         'lon': (['lon'], basefile['lon'].values[::3])})
    regridder = xe.Regridder(ds_out, ds_new, 'bilinear', ignore_degenerate=True)
    ds_new = regridder(ds_out)
    ds_tmp = xr.Dataset({'lat': (['lat'], basefile['lat'].values[:-1:3]),
                         'lon': (['lon'], [0.0]+list(basefile['lon'].values[::3])+[360.0])})

    ds_tmp['z'] = (ds_new['z'].dims,smooth_n_point(close_lon_gap(ds_new,'z'),9))
    ds_tmp['PHIS'] = (ds_new['z'].dims,9.8*ds_tmp['z'].values)
    ds_tmp['mask'] = (ds_new['z'].dims,np.where(ds_tmp['z'].values>0,1,0))
    ds_tmp.to_netcdf(interp_file)
    return ds_tmp

def close_lon_gap(data,field):
    if data[field].ndim==3:
        tmp = np.zeros((len(data['time'].values),len(data['lat'].values),len(data['lon'].values)+2))
        tmp[:,:,1:-1] = data[field].values
        tmp[:,:,-1] = 0.5*(data[field].values[:,:,0]+data[field].values[:,:,-1])
        tmp[:,:,0] = 0.5*(data[field].values[:,:,0]+data[field].values[:,:,-1])
    else:    
        tmp = np.zeros((len(data['lat'].values),len(data['lon'].values)+2))
        tmp[:,1:-1] = data[field].values
        tmp[:,-1] = 0.5*(data[field].values[:,0]+data[field].values[:,-1])
        tmp[:,0] = 0.5*(data[field].values[:,0]+data[field].values[:,-1])
    return tmp

def field_interp(land_year,stored_years=stored_years,field=None,level=None,gcm_type='cesm',plevel=None,time_avg=True):
    
    if gcm_type=='cesm':
        if field is not None:
            if 'lev' not in reference_data[field].dims:
                plevel = None
    if gcm_type=='isca':
        if field is not None:
            if 'pfull' not in reference_data[field].dims:
                plevel = None

    interp_file = f'{os.environ["USER_OUTPUT_DIR"]}/{gcm_type}_{field}_{land_year}_{plevel}.nc'
    if os.path.exists(interp_file):
        return xr.open_mfdataset(interp_file)

    year = float(land_year)
    keys = sorted(stored_years)
    if gcm_type=='cesm':
        height_key = 'z'
        basefile = xr.open_mfdataset(os.environ['ORIG_CESM_TOPO_FILE'])
        if plevel is not None:
            level = np.argmin(np.abs(plevel-np.array(cesm_plevels))) 
    else:
        height_key = 'zsurf'
        basefile = xr.open_mfdataset(os.environ['ORIG_ISCA_TOPO_FILE'])
        if plevel is not None:
            level = np.argmin(np.abs(plevel-np.array(isca_plevels))) 

    ds_out = xr.Dataset({'lat': (['lat'], basefile['lat'].values),
                         'lon': (['lon'], basefile['lon'].values)})
    
    if land_year in keys:
        exp = Experiment(gcm_type=gcm_type,land_year=keys[keys.index(land_year)])
        ds_out['z'] = (('lat','lon'),exp.base()[height_key].values)
        if field is not None:
            if exp.sim_data()[field].ndim==2:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].values)
            if exp.sim_data()[field].ndim==3:
                if not time_avg:
                    ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].values)
                else:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=0).values)
            if exp.sim_data()[field].ndim==4:
                if level is None:
                    if not time_avg:
                        ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].mean(axis=(1)).values) 
                    else:
                        ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0,1)).values)
                else:
                    if not time_avg:
                        ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].values[level])
                    else:
                        ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0))[level].values)
            ds_out[field].attrs['long_name']= exp.sim_data()[field].long_name    
            ds_out[field].attrs['units'] = exp.sim_data()[field].units

    elif year <= keys[0]:
        exp = Experiment(gcm_type=gcm_type,land_year=keys[0])
        ds_out['z'] = (('lat','lon'),exp.base()[height_key].values)
        if field is not None:
            if exp.sim_data()[field].ndim==2:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].values)
            if exp.sim_data()[field].ndim==3:
                if not time_avg:
                    ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].values)
                else:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=0).values)
            if exp.sim_data()[field].ndim==4:
                if level is None:
                    if not time_avg:
                        ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].mean(axis=(1)).values) 
                    else:
                        ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0,1)).values)
                else:
                    if not time_avg:
                        ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].values[level])
                    else:
                        ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0))[level].values)
            ds_out[field].attrs['long_name']= exp.sim_data()[field].long_name    
            ds_out[field].attrs['units'] = exp.sim_data()[field].units

    elif year >= keys[-1]:
        exp = Experiment(gcm_type=gcm_type,land_year=keys[-1])
        ds_out['z'] = (('lat','lon'),exp.base()[height_key].values)
        if field is not None:
            if exp.sim_data()[field].ndim==2:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].values)
            if exp.sim_data()[field].ndim==3:
                if not time_avg:
                    ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].values)
                else:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=0).values)
            if exp.sim_data()[field].ndim==4:
                if level is None:
                    if not time_avg:
                        ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].mean(axis=(1)).values) 
                    else:
                        ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0,1)).values)
                else:
                    if not time_avg:
                        ds_out[field] = (('time','lat','lon'),exp.sim_data()[field].values[level])
                    else:
                        ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0))[level].values)
            ds_out[field].attrs['long_name']= exp.sim_data()[field].long_name    
            ds_out[field].attrs['units'] = exp.sim_data()[field].units

    else:
        for i in range(len(keys)):
            if keys[i] <= year <= keys[i+1]:
                exp0 = Experiment(gcm_type=gcm_type,land_year=keys[i])
                exp1 = Experiment(gcm_type=gcm_type,land_year=keys[i+1])
                tmp = interp(exp0.base()[height_key].values,
                             exp1.base()[height_key].values,
                             (year-keys[i])/(keys[i+1]-keys[i]))
                ds_out['z'] = (('lat','lon'),tmp)
                if field is not None:
                    if exp0.sim_data()[field].ndim==2:
                        a = exp0.sim_data()[field].values
                        b = exp1.sim_data()[field].values
                    if exp0.sim_data()[field].ndim==3:
                        if not time_avg:
                            a = exp0.sim_data()[field].values
                            b = exp1.sim_data()[field].values
                        else:
                            a = exp0.sim_data()[field].mean(axis=0).values
                            b = exp1.sim_data()[field].mean(axis=0).values
                    if exp0.sim_data()[field].ndim==4:
                        if level is None:
                            if not time_avg:
                                a = exp0.sim_data()[field].mean(axis=(1)).values
                                b = exp1.sim_data()[field].mean(axis=(1)).values
                            else:
                                a = exp0.sim_data()[field].mean(axis=(0,1)).values
                                b = exp1.sim_data()[field].mean(axis=(0,1)).values
                        else:
                            if not time_avg:
                                a = exp0.sim_data()[field].values[level]
                                b = exp1.sim_data()[field].values[level]
                            else:
                                a = exp0.sim_data()[field].mean(axis=(0)).values[level]
                                b = exp1.sim_data()[field].mean(axis=(0)).values[level]
                    tmp = interp(a,b,(year-keys[i])/(keys[i+1]-keys[i]))
                    if len(tmp.shape)==3:
                        ds_out[field] = (('time','lat','lon'),tmp)
                    else:
                        ds_out[field] = (('lat','lon'),tmp)
                    ds_out[field].attrs['long_name'] = exp0.sim_data()[field].long_name    
                    ds_out[field].attrs['units'] = exp0.sim_data()[field].units

    ds_new = xr.Dataset({'lat': (['lat'], basefile['lat'].values),
                         'lon': (['lon'], [0.0]+list(basefile['lon'].values)+[360.0])})
    ds_new['z'] = (ds_out['z'].dims,close_lon_gap(ds_out,'z'))
    ds_new['PHIS'] = (ds_out['z'].dims,9.8*ds_new['z'].values)
    ds_new['mask'] = (ds_out['z'].dims,np.where(ds_new['z'].values>0,1,0))
    if field is not None:
        ds_new[field] = (ds_out[field].dims,smooth_n_point(close_lon_gap(ds_out,field),5))
        ds_new[field].attrs['long_name'] = ds_out[field].long_name
        ds_new[field].attrs['units'] = ds_out[field].units
    if 'time' not in ds_new:
        ds_new.to_netcdf(interp_file)
    return ds_new

def potential_intensity(t_surf):
    A = 28.2
    B = 55.8
    C = 0.1813
    T0 = 30.0+273.15
    return A+B*np.exp(C*(t_surf-T0))

def define_land_colormap():

    ncolors = 256
    color_array = list(plt.get_cmap('GnBu_r')(range(ncolors))[0:128]) \
                + list(plt.get_cmap('gist_earth')(range(ncolors))[128:128+16]) \
                + list(plt.get_cmap('terrain')(range(ncolors))[160:160+16])[::-1] \
                + list(plt.get_cmap('gist_earth')(range(ncolors))[192:192+16]) \
                + list(plt.get_cmap('terrain')(range(ncolors))[128:128+16]) \
                + list(plt.get_cmap('terrain')(range(ncolors))[224:224+32]) \
                + list(plt.get_cmap('gist_gray')(range(ncolors))[256-32:])
    map_object = LinearSegmentedColormap.from_list(name='custom_earth',colors=color_array)

    # register this new colormap with matplotlib
    unregister_cmap('custom_earth')
    plt.register_cmap(cmap=map_object)

def mpl_to_plotly(cmap, pl_entries=11, rdigits=2):
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3]*255).astype(np.uint8)
    pl_colorscale = [[round(s, rdigits), f'rgb{tuple(color)}'] for s, color in zip(scale, colors)]
    return pl_colorscale

def define_cloud_colormap():

    ncolors = 256
    ramped_alpha = list([0]*128)+list(np.linspace(0.5,1,ncolors-128))
    color_array = []
    for i in range(ncolors):
        color_array.append((1,1,1,ramped_alpha[i]))
    map_object = LinearSegmentedColormap.from_list(name='custom_clouds',colors=color_array)

    # register this new colormap with matplotlib
    unregister_cmap('custom_clouds')
    plt.register_cmap(cmap=map_object)

def get_data(experiment,field='t_surf',level=None,decode_times=True,anomaly=False):
    data = xr.open_mfdataset(experiment.files,decode_times=decode_times)
    land = xr.open_mfdataset(os.path.join(os.environ['TOPO_DIR'],experiment.topo_file),
            decode_times=False)
    
    if field=='potential_intensity':
        pi = potential_intensity(data['t_surf'].values)
        data[field] = (data['t_surf'].dims,pi)
        data[field].attrs['long_name'] = 'potential intensity'
        data[field].attrs['units'] = 'm/s'

    if anomaly:
        #tmp = xr.open_mfdataset(Experiment(land_year=0).files,decode_times=decode_times) 
        tmp = get_data(Experiment(land_year=0),field=field,level=level,anomaly=False)
        data[field] = (tmp[field].dims,data[field].values-tmp[field].values.mean())
    
    if field=='potential_intensity':
        data[field] = (data[field].dims,np.multiply(data[field].values,np.array(land['land_mask'].values < 1.0, dtype=float)))        

    if 'co2' in data:
        tmp = data[[field,'co2']]
    else:
        tmp = data[[field]]
    
    tmp['land_mask'] = (land['land_mask'].dims,land['land_mask'].values)

    if 'pfull' in tmp[field].dims:
        if level is None:
            level = 250
        levels = np.array(tmp['pfull'].values)
        level = levels[np.argmin(np.abs(levels-level))]
        
        tmp[field] = tmp[field].sel(pfull = level)
    
    return tmp

def get_avg_field(exp,field='t_surf',level=None,vmin=None,vmax=None,anomaly=False):

    data = get_data(exp,field=field,level=level,decode_times=True,anomaly=anomaly)

    land = data['land_mask']
# Setup the initial plot
    if 'co2' in data:
        co2 = data['co2']

    variable = data[field]

    lons = sorted(variable.lon.values)

    fig = plt.figure(figsize=(10,5))
    proj = ccrs.PlateCarree(central_longitude=180.0)
    ax = plt.axes(projection=proj)

    if 'time' in variable.dims:
        avg = variable.mean(dim='time')
    else:
        avg = variable

    image = avg.plot.imshow(ax=ax, transform=proj,
                            interpolation='bilinear',cmap="coolwarm",
                            animated=True, add_colorbar=False,
                            )

    if land is not None:
        land_img = land.plot.contour(ax=ax, transform=ccrs.PlateCarree(),
                                     cmap="land_cmap",
                                     add_colorbar=False,
                                     alpha=1.0,
                                    )
    else:
        ax.coastlines()

    try:
        cb = plt.colorbar(image, ax=ax, orientation='horizontal', pad=0.05, label=f'{variable.long_name} ({variable.units})')
    except:
        cb = plt.colorbar(image, ax=ax, orientation='horizontal', pad=0.05, label=f'{variable.name}')

    if vmin is None or vmax is None:
        image.set_clim(variable.values.min(),variable.values.max())
    else:
        image.set_clim(vmin,vmax)

    text = cb.ax.xaxis.label
    font = matplotlib.font_manager.FontProperties(size=16)
    text.set_font_properties(font)

    image.set_array(avg.sel(lon=lons))

    if 'co2' in data:
        ax.set_title(f'Time Average, CO2 = {sig_round(co2.values.mean(),4)} ({co2.units}), solar constant = {sig_round(solar_constant(exp.land_year),4)} (W/m**3)',fontsize=20)
    else:
        ax.set_title(f'Time Average',fontsize=20)

def get_hires_topo_and_polar_coords(land_year=0):#,rstride=1,cstride=1):

    hires_topo = hires_interp(land_year)['z'].values
    hires_lons = hires_interp(land_year)['lon'].values
    hires_lats = hires_interp(land_year)['lat'].values
    
    if len(hires_lats.shape)==1:
        hires_lats = np.tile(hires_lats,(hires_topo.shape[1],1)).transpose()
    if len(hires_lons.shape)==1:
        hires_lons = np.tile(hires_lons,(hires_topo.shape[0],1))
    
    xs,ys,zs = mapping_map_to_sphere(hires_lons,hires_lats)#[::rstride,::cstride],hires_lats[::rstride,::cstride])
    #hires_topo = smooth_n_point(hires_topo,9)
    #xs,ys,zs = mapping_map_to_sphere(hires_lons,hires_lats)
    ratio_topo = 1.0 + hires_topo*1e-5
    xs = xs*ratio_topo
    ys = ys*ratio_topo
    zs = zs*ratio_topo
    return smooth_n_point(hires_topo,5),xs,ys,zs

def get_lowres_topo_and_polar_coords(land_year=0):

    lowres_topo = field_interp(land_year)['z'].values
    lowres_lons = field_interp(land_year)['lon'].values
    lowres_lats = field_interp(land_year)['lat'].values

    lowres_lats = np.tile(lowres_lats,(lowres_topo.shape[1],1)).transpose()
    lowres_lons = np.tile(lowres_lons,(lowres_topo.shape[0],1))
    
    xs,ys,zs = mapping_map_to_sphere(lowres_lons,lowres_lats)
    #hires_topo = smooth_n_point(hires_topo,9)
    #xs,ys,zs = mapping_map_to_sphere(hires_lons,hires_lats)
    ratio_topo = 1.0 + lowres_topo*1e-5
    xs = xs*ratio_topo
    ys = ys*ratio_topo
    zs = zs*ratio_topo

    return smooth_n_point(lowres_topo,5),xs,ys,zs


def get_field_and_polar_coords(land_year=0,field='TS',gcm_type='cesm',level=None,plevel=None):

    interp_call = field_interp(land_year,gcm_type=gcm_type,field=field,level=level,plevel=plevel)
    
    field_array = interp_call[field]
    topo = interp_call['z'].values
    lons = interp_call['lon'].values
    lats = interp_call['lat'].values

    lats = np.tile(lats,(field_array.shape[1],1)).transpose()
    lons = np.tile(lons,(field_array.shape[0],1))
    
    xt,yt,zt = mapping_map_to_sphere(lons,lats)
    xt = xt*(1.025+topo*1e-5)
    yt = yt*(1.025+topo*1e-5)
    zt = zt*(1.025+topo*1e-5)

    return field_array,xt,yt,zt

def get_interactive_globe(land_year=0,field='RELHUM',
                          gcm_type='cesm',
                          level=None,
                          plevel=None,
                          save_html=False,
                          save_fig=False,
                          fig_name=None,
                          vmin=None,vmax=None,
                          fast=False):

    define_noaa_colormap()
    titlecolor = 'white'
    bgcolor = 'black'
    
    start_time = time.time()
    field_array,xt,yt,zt = get_field_and_polar_coords(land_year,
                                                      field=field,
                                                      gcm_type=gcm_type,
                                                      level=level,
                                                      plevel=plevel)
    #logger.info(f'get_field_and_polar_coords time: {time.time()-start_time}')

    #start_time = time.time()
    if fast:
        rstride=10
        cstride=20
        topo,xs,ys,zs = get_lowres_topo_and_polar_coords(land_year)
        #topo,xs,ys,zs = get_hires_topo_and_polar_coords(land_year,rstride=rstride,cstride=cstride)
    else:
        rstride=1
        cstride=2
        topo,xs,ys,zs = get_hires_topo_and_polar_coords(land_year)#,rstride=rstride,cstride=cstride)
    #logger.info(f'get_hires_topo_and_polar_coords: {time.time()-start_time}')
    #logger.info(f'Prep time: {time.time()-start_time}')
    

    start_time = time.time()
    Ctopo = mpl_to_plotly(plt.get_cmap('custom_noaa'),255)
    if field=='RELHUM' or field=='rh' or 'CLOUD' in field:
        Cfield = [[0.0, 'rgba(255,255,255,0.0)'], [0.1, 'rgba(255,255,255,0.1)'],
                  [0.2, 'rgba(255,255,255,0.2)'], [0.3, 'rgba(255,255,255,0.3)'],
                  [0.4, 'rgba(255,255,255,0.4)'], [0.5, 'rgba(255,255,255,0.5)'],
                  [0.6, 'rgba(255,255,255,0.6)'], [0.7, 'rgba(255,255,255,0.7)'],
                  [0.8, 'rgba(255,255,255,0.8)'], [0.9, 'rgba(255,255,255,0.9)'],
                  [1.0, 'rgba(255,255,255,1.0)']]
        field_alpha = 1.0
    else:
        Cfield = mpl_to_plotly(plt.get_cmap('coolwarm'),255)
        field_alpha = 0.6

    topo_sphere=dict(type='surface',
                     x=xs,
                     y=ys,
                     z=zs,
                     colorscale=Ctopo,
                     surfacecolor=topo,
                     opacity=1.0,
                     cmin=-10000,
                     cmax=10000,
                     showscale=False)

    field_sphere=dict(type='surface',
                      x=xt,
                      y=yt,
                      z=zt,
                      colorscale=Cfield,#Ctopo,
                      surfacecolor=field_array.values,
                      opacity=field_alpha,
                      showscale=True,
                      cmin=vmin,
                      cmax=vmax,
                      colorbar=dict(thickness=20,
                                    title=f'{field} ({field_array.units})',
                                    titleside='right',
                                    tickfont=dict(family='Courier New', color=titlecolor),
                                    titlefont=dict(family='Courier New', color=titlecolor)))
    

    noaxis=dict(showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                ticks='',
                title='',
                zeroline=False)
    
    title = f'Time: {str(round(land_year,2))}Ma BP, '
    title += f'Average {field_array.long_name}: {str(round(field_array.values.mean(),2))} ({field_array.units})'
    layout = go.Layout(
                       autosize=False, width=1800, height=750,
                       title = title,
                       title_x = 0.5,
                       title_y = 0.95,
                       titlefont = dict(family='Courier New', color=titlecolor),
                       showlegend = True,
                       margin=dict(l=20, r=50, t=80, b=20),
                       scene = dict(xaxis = noaxis,
                                    yaxis = noaxis,
                                    zaxis = noaxis,
                                    aspectmode='manual',
                                    aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1)),
                       scene_camera=dict(eye=polar_to_cartesian(radius=1.4,lat=20,lon=320)),
                       paper_bgcolor = bgcolor,
                       plot_bgcolor = bgcolor)

    fig = go.Figure([topo_sphere,field_sphere], layout=layout)
    if level is not None:
        outfile = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_lev{level}_{land_year}Ma_interactive_globe.html'
    else:
        outfile = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_{land_year}Ma_interactive_globe.html'
    if save_html:
        start_time = time.time()
        logger.info(f'Saving webpage: {outfile}')
        po.plot(fig,validate=False,filename=outfile,auto_open=False) 
        #logger.info(f'po.plot time: {time.time()-start_time}')
    if fig_name is None:
        fig_name = outfile.rstrip('.html')+'.png'
    else:
        fig_name = f'{os.environ["USER_ANIMS_DIR"]}/{fig_name.split("/")[-1]}'
    if save_fig:
        start_time = time.time()
        logger.info(f'Saving figure: {fig_name}')
        pio.write_image(fig,fig_name,format='png')
        #logger.info(f'pio.write_image time: {time.time()-start_time}')
    #logger.info(f'Plotting time: {time.time()-start_time}')
    return fig

def get_field_time_animation(year,stored_years=stored_years,
                             field='TS',level=None,
                             level_num=10,vmin=None,
                             vmax=None,color_map='coolwarm',
                             globe=False,
                             gcm_type='cesm'):

    define_land_colormap()
    define_cloud_colormap()
    define_noaa_colormap()

    fig = plt.figure(figsize=(12,7))
    if globe:
        proj = ccrs.Orthographic(330, 20)
    else:
        proj = ccrs.PlateCarree(central_longitude=180.0)
    ax = plt.axes(projection=proj)
    ax.gridlines(color='black', linestyle='dotted')
    field_alpha=0.6
    if field=='RELHUM':
        level_num=5
    
    init_field = field_interp(year,stored_years,field=field,level=level,gcm_type=gcm_type,time_avg=False)[field]
    
    land_img = hires_interp(year,stored_years)['z'].plot.imshow(ax=ax, 
                                                                transform=ccrs.PlateCarree(), 
                                                                cmap='custom_noaa',
                                                                add_colorbar=False, 
                                                                alpha=1.0)
    border_img = hires_interp(year,stored_years)['mask'].plot.contour(ax=ax, levels=2, 
                                                                      transform=ccrs.PlateCarree(), 
                                                                      colors="black", 
                                                                      add_colorbar=False, 
                                                                      alpha=1.0)
    image = init_field[0].plot.imshow(ax=ax, levels=level_num,
                                      transform=ccrs.PlateCarree(), 
                                      interpolation='bilinear',
                                      cmap=color_map, 
                                      add_colorbar=False,
                                      alpha=0.6)
    
    field_border = init_field[0].plot.contour(ax=ax, levels=level_num,
                                           transform=ccrs.PlateCarree(), 
                                           interpolation='bilinear',
                                           cmap=color_map, 
                                           add_colorbar=False,
                                           alpha=0.8)
     
    cb = plt.colorbar(image, ax=ax, orientation='horizontal', pad=0.05, label=f'{init_field.long_name} ({init_field.units})')
    if vmin is not None and vmax is not None:
        image.set_clim(vmin,vmax)
    
    text = cb.ax.xaxis.label
    font = matplotlib.font_manager.FontProperties(size=16)
    text.set_font_properties(font)
    
    def update(i):
        logger.info(f'Animation step: {i}')

        fig.suptitle(f'Time: {year} Ma BP, Step: {i}', fontsize=20)       
        ax.clear()
        
        land_img = hires_interp(year,stored_years)['z'].plot.imshow(ax=ax, 
                                                                 transform=ccrs.PlateCarree(), 
                                                                 cmap='custom_noaa',
                                                                 add_colorbar=False, 
                                                                 alpha=1.0)
        border_img = hires_interp(year,stored_years)['mask'].plot.contour(ax=ax,levels=2, 
                                                                       transform=ccrs.PlateCarree(), 
                                                                       colors="black", 
                                                                       add_colorbar=False, 
                                                                       alpha=1.0)
        image = init_field[i].plot.imshow(ax=ax,levels=level_num,
                                                 transform=ccrs.PlateCarree(), 
                                                 interpolation='bilinear',
                                                 cmap=color_map,
                                                 add_colorbar=False,
                                                 alpha=field_alpha)
        
        field_border = init_field[i].plot.contour(ax=ax,levels=level_num,
                                                         transform=ccrs.PlateCarree(), 
                                                         interpolation='bilinear',
                                                         cmap=color_map, 
                                                         add_colorbar=False,
                                                         alpha=0.8)
        return image
    
    plt.close()
    animation = anim.FuncAnimation(fig, update, frames=range(init_field.shape[0]), blit=False)
    writervideo = anim.FFMpegWriter(fps=5) 
    if level is not None:
        if globe:
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_lev{level}_{year}Ma_globe_animation.mp4'
        else:    
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_lev{level}_{year}Ma_animation.mp4'
    else:    
        if globe:
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_{year}Ma_globe_animation.mp4'
        else:    
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_{year}Ma_animation.mp4'
    animation.save(anim_file, writer=writervideo)
    print(anim_file)
    #return HTML(animation.to_jshtml())        

def get_field_animation(times,stored_years=stored_years,
                        field='TS',level=None,
                        level_num=10,vmin=None,
                        vmax=None,color_map='coolwarm',
                        globe=False,
                        gcm_type='cesm'):

    define_land_colormap()
    define_cloud_colormap()
    define_noaa_colormap()

    fig = plt.figure(figsize=(12,7))
    if globe:
        proj = ccrs.Orthographic(330, 20)
    else:
        proj = ccrs.PlateCarree(central_longitude=0.0)
    ax = plt.axes(projection=proj)
    ax.gridlines(color='black', linestyle='dotted')
    field_alpha=0.6
    
    init_field = field_interp(0,stored_years,field=field,level=level,gcm_type=gcm_type)[field]
    
    land_img = hires_interp(0,stored_years)['z'].plot.imshow(ax=ax, 
                                                             transform=ccrs.PlateCarree(), 
                                                             cmap='custom_noaa',
                                                             add_colorbar=False, 
                                                             alpha=1.0)
    border_img = hires_interp(0,stored_years)['mask'].plot.contour(ax=ax, levels=2, 
                                                                   transform=ccrs.PlateCarree(), 
                                                                   colors="black", 
                                                                   add_colorbar=False, 
                                                                   alpha=1.0)
    image = init_field.plot.imshow(ax=ax, levels=level_num,
                                   transform=ccrs.PlateCarree(), 
                                   interpolation='bilinear',
                                   cmap=color_map, 
                                   add_colorbar=False,
                                   alpha=0.6)
    
    field_border = init_field.plot.contour(ax=ax, levels=level_num,
                                           transform=ccrs.PlateCarree(), 
                                           interpolation='bilinear',
                                           cmap=color_map, 
                                           add_colorbar=False,
                                           alpha=0.8)
     
    cb = plt.colorbar(image, ax=ax, orientation='horizontal', pad=0.05, label=f'{init_field.long_name} ({init_field.units})')
    if vmin is not None and vmax is not None:
        image.set_clim(vmin,vmax)
    
    text = cb.ax.xaxis.label
    font = matplotlib.font_manager.FontProperties(size=16)
    text.set_font_properties(font)
    
    def update(i):
        logger.info(f'Animation step: {i}')

        t = times[i]
        current_field = field_interp(t,stored_years,field=field,level=level,gcm_type=gcm_type)
        fig.suptitle(f'Time: {str(round(t,2))} Ma BP, Average {field}: {str(round(current_field[field].values.mean(),2))} {init_field.units}', fontsize=20)       
        ax.clear()
        
        land_img = hires_interp(t,stored_years)['z'].plot.imshow(ax=ax, 
                                                                 transform=ccrs.PlateCarree(), 
                                                                 cmap='custom_noaa',
                                                                 add_colorbar=False, 
                                                                 alpha=1.0)
        border_img = hires_interp(t,stored_years)['mask'].plot.contour(ax=ax,levels=2, 
                                                                       transform=ccrs.PlateCarree(), 
                                                                       colors="black", 
                                                                       add_colorbar=False, 
                                                                       alpha=1.0)
        image = current_field[field].plot.imshow(ax=ax,levels=level_num,
                                                 transform=ccrs.PlateCarree(), 
                                                 interpolation='bilinear',
                                                 cmap=color_map,
                                                 add_colorbar=False,
                                                 alpha=field_alpha)
        
        field_border = current_field[field].plot.contour(ax=ax,levels=level_num,
                                                         transform=ccrs.PlateCarree(), 
                                                         interpolation='bilinear',
                                                         cmap=color_map, 
                                                         add_colorbar=False,
                                                         alpha=0.8)
        return image
    
    plt.close()
    animation = anim.FuncAnimation(fig, update, frames=range(len(times)), blit=False)
    writervideo = anim.FFMpegWriter(fps=5) 
    if level is not None:
        if globe:
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_lev{level}_globe_animation.mp4'
        else:    
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_lev{level}_animation.mp4'
    else:    
        if globe:
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_globe_animation.mp4'
        else:    
            anim_file = f'{os.environ["USER_FIGS_DIR"]}/{gcm_type}_{field}_animation.mp4'
    animation.save(anim_file, writer=writervideo)
    print(anim_file)
    #return HTML(animation.to_jshtml())        

def get_continent_animation(times,stored_years,globe=False):

    define_land_colormap()
    define_noaa_colormap()

    fig = plt.figure(figsize=(12,7))

    if globe:
        proj = ccrs.Orthographic(320, 20)
    else:
        proj = ccrs.PlateCarree(central_longitude=0.0)
    proj = ccrs.PlateCarree(central_longitude=0.0)
    ax = plt.axes(projection=proj)
    
    image = hires_interp(0,stored_years)['z'].plot.imshow(ax=ax, 
                                                          transform=ccrs.PlateCaree(), 
                                                          interpolation='bilinear',
                                                          cmap='custom_noaa',
                                                          animated=True,
                                                          add_colorbar=False)
    border_img = hires_interp(0,stored_years)['mask'].plot.contour(ax=ax, levels=2, 
                                                                   transform=ccrs.PlateCaree(), 
                                                                   colors="black", 
                                                                   add_colorbar=False, 
                                                                   alpha=1.0)
    cb = plt.colorbar(image, ax=ax, orientation='horizontal', pad=0.05, label=f'altitude (m)')
        
    text = cb.ax.xaxis.label
    font = matplotlib.font_manager.FontProperties(size=16)
    text.set_font_properties(font)
    
    def update(i):
        logger.info(f'Animation step: {i}')

        t = times[i]
        
        ax.clear()
        #current_field = field_interp(t,stored_years)
        current_field = hires_interp(t,stored_years)
        fig.suptitle(f'Time: {str(round(t,2))} Ma BP', fontsize=20)       
        ax.clear()
        image = current_field['z'].plot.imshow(ax=ax,
                                               transform=ccrs.PlateCarree(),
                                               interpolation='bilinear',
                                               cmap='custom_noaa',
                                               add_colorbar=False)
        border_img = hires_interp(t,stored_years)['mask'].plot.contour(ax=ax, levels=2, 
                                                                       transform=ccrs.PlateCarree(), 
                                                                       colors="black", 
                                                                       add_colorbar=False, 
                                                                       alpha=1.0)
        return image
    
    plt.close()
    animation = anim.FuncAnimation(fig, update, frames=range(len(times)), blit=False)
    writervideo = anim.FFMpegWriter(fps=5)
    if globe:
        anim_file = f'{os.environ["USER_FIGS_DIR"]}/continent_globe_animation.mp4'
    else:
        anim_file = f'{os.environ["USER_FIGS_DIR"]}/continent_animation.mp4'
    animation.save(anim_file, writer=writervideo)
    logger.info(f'Saving: {anim_file}')
