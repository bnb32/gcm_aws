import ecrlgcm.environment
from ecrlgcm.misc import sig_round, interp, land_years, get_logger
from ecrlgcm.preprocessing import solar_constant
from ecrlgcm.experiment import Experiment

import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.animation as anim
from IPython.display import HTML
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import unregister_cmap
from metpy.calc import smooth_n_point
import os

logger = get_logger()

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

def hires_interp(land_year,stored_years):
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
    
    ds_out['z'] = (ds_out['z'].dims,smooth_n_point(ds_out['z'].values,9))
    ds_out['PHIS'] = (ds_out['z'].dims,9.8*ds_out['z'].values)
    ds_out['mask'] = (ds_out['z'].dims,np.where(ds_out['z'].values>0,1,0))
    return ds_out

def field_interp(land_year,stored_years,field=None,level=None):
    year = float(land_year)
    keys = sorted(stored_years)
    basefile = xr.open_mfdataset(os.environ['ORIG_CESM_TOPO_FILE'])
    ds_out = xr.Dataset({'lat': (['lat'], basefile['lat'].values),
                         'lon': (['lon'], basefile['lon'].values)})
    
    if land_year in keys:
        exp = Experiment(land_year=keys[keys.index(land_year)])
        ds_out['z'] = (('lat','lon'),exp.base()['z'].values)
        if field is not None:
            if exp.sim_data()[field].ndim==2:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].values)
            if exp.sim_data()[field].ndim==3:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=0).values)
            if exp.sim_data()[field].ndim==4:
                if level is None:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0,1)).values)
                else:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0))[level].values)
            ds_out[field].attrs['long_name']= exp.sim_data()[field].long_name    
            ds_out[field].attrs['units'] = exp.sim_data()[field].units

    elif year <= keys[0]:
        exp = Experiment(land_year=keys[0])
        ds_out['z'] = (('lat','lon'),exp.base()['z'].values)
        if field is not None:
            if exp.sim_data()[field].ndim==2:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].values)
            if exp.sim_data()[field].ndim==3:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=0).values)
            if exp.sim_data()[field].ndim==4:
                if level is None:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0,1)).values)
                else:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0)).values[level])
            ds_out[field].attrs['long_name']= exp.sim_data()[field].long_name    
            ds_out[field].attrs['units'] = exp.sim_data()[field].units

    elif year >= keys[-1]:
        exp = Experiment(land_year=keys[-1])
        ds_out['z'] = (('lat','lon'),exp.base()['z'].values)
        if field is not None:
            if exp.sim_data()[field].ndim==2:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].values)
            if exp.sim_data()[field].ndim==3:
                ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=0).values)
            if exp.sim_data()[field].ndim==4:
                if level is None:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0,1)).values)
                else:
                    ds_out[field] = (('lat','lon'),exp.sim_data()[field].mean(axis=(0)).values[level])
            ds_out[field].attrs['long_name']= exp.sim_data()[field].long_name    
            ds_out[field].attrs['units'] = exp.sim_data()[field].units

    else:
        for i in range(len(keys)):
            if keys[i] <= year <= keys[i+1]:
                exp0 = Experiment(land_year=keys[i])
                exp1 = Experiment(land_year=keys[i+1])
                tmp = interp(exp0.base()['z'].values,
                             exp1.base()['z'].values,
                             (year-keys[i])/(keys[i+1]-keys[i]))
                ds_out['z'] = (('lat','lon'),tmp)
                if field is not None:
                    if exp0.sim_data()[field].ndim==2:
                        a = exp0.sim_data()[field].values
                        b = exp1.sim_data()[field].values
                    if exp0.sim_data()[field].ndim==3:
                        a = exp0.sim_data()[field].mean(axis=0).values
                        b = exp1.sim_data()[field].mean(axis=0).values
                    if exp0.sim_data()[field].ndim==4:
                        if level is None:
                            a = exp0.sim_data()[field].mean(axis=(0,1)).values
                            b = exp1.sim_data()[field].mean(axis=(0,1)).values
                        else:
                            a = exp0.sim_data()[field].mean(axis=(0)).values[level]
                            b = exp1.sim_data()[field].mean(axis=(0)).values[level]
                    tmp = interp(a,b,(year-keys[i])/(keys[i+1]-keys[i]))
                    ds_out[field] = (('lat','lon'),tmp)
                    ds_out[field].attrs['long_name']= exp0.sim_data()[field].long_name    
                    ds_out[field].attrs['units'] = exp0.sim_data()[field].units
    
    ds_out['PHIS'] = (ds_out['z'].dims,9.8*ds_out['z'].values)
    ds_out['mask'] = (ds_out['z'].dims,np.where(ds_out['z'].values>0,1,0))
    return ds_out

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

def define_cloud_colormap():

    ncolors = 256
    ramped_alpha = list([0]*1)+list(np.linspace(0.0,1,255))
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

def get_field_animation(times,stored_years,field='TS',level=None,
                        level_num=10,vmin=None,vmax=None,
                        color_map='coolwarm'):

    define_land_colormap()
    define_cloud_colormap()
    define_noaa_colormap()

    fig = plt.figure(figsize=(12,7))
    proj = ccrs.PlateCarree(central_longitude=0.0)
    ax = plt.axes(projection=proj)
    if field != 'RELHUM': field_alpha=0.6
    else: field_alpha=1.0
    
    init_field = field_interp(0,stored_years,field=field,level=level)[field]
    
    land_img = hires_interp(0,stored_years)['z'].plot.imshow(ax=ax, 
                                                             transform=proj,#ccrs.PlateCarree(), 
                                                             #cmap='custom_earth',
                                                             cmap='custom_noaa',
                                                             add_colorbar=False, 
                                                             alpha=1.0)
    border_img = hires_interp(0,stored_years)['mask'].plot.contour(ax=ax, levels=2, 
                                                                   transform=proj,#ccrs.PlateCarree(), 
                                                                   colors="black", 
                                                                   add_colorbar=False, 
                                                                   alpha=1.0)
    image = init_field.plot.imshow(ax=ax, levels=level_num,
                                   transform=proj, 
                                   interpolation='bilinear',
                                   cmap=color_map, 
                                   #animated=True,
                                   add_colorbar=False,
                                   alpha=field_alpha)
    
    field_border = init_field.plot.contour(ax=ax, levels=level_num,
                                           transform=proj, 
                                           interpolation='bilinear',
                                           cmap=color_map, 
                                           #animated=True,
                                           add_colorbar=False,
                                           alpha=0.9)
     
    cb = plt.colorbar(image, ax=ax, orientation='horizontal', pad=0.05, label=f'{init_field.long_name} ({init_field.units})')
    if vmin is not None and vmax is not None:
        image.set_clim(vmin,vmax)
    
    text = cb.ax.xaxis.label
    font = matplotlib.font_manager.FontProperties(size=16)
    text.set_font_properties(font)
    
    def update(i):
        logger.info(f'Animation step: {i}')

        t = times[i]
        current_field = field_interp(t,stored_years,field=field,level=level)
        fig.suptitle(f'Time: {str(round(t,2))} Ma B.P., Average {field}: {str(round(current_field[field].values.mean(),2))} {init_field.units}', fontsize=20)       
        ax.clear()
        
        land_img = hires_interp(t,stored_years)['z'].plot.imshow(ax=ax, 
                                                                 transform=proj,#ccrs.PlateCarree(), 
                                                                 #cmap='custom_earth', 
                                                                 cmap='custom_noaa',
                                                                 add_colorbar=False, 
                                                                 alpha=1.0)
        border_img = hires_interp(t,stored_years)['mask'].plot.contour(ax=ax,levels=2, 
                                                                       transform=proj,#ccrs.PlateCarree(), 
                                                                       colors="black", 
                                                                       add_colorbar=False, 
                                                                       alpha=1.0)
        image = current_field[field].plot.imshow(ax=ax,levels=level_num,
                                                 transform=proj,
                                                 interpolation='bilinear',
                                                 cmap=color_map,
                                                 #animated=True,
                                                 add_colorbar=False,
                                                 alpha=field_alpha)
        
        field_border = current_field[field].plot.contour(ax=ax,levels=level_num,
                                                         transform=proj, 
                                                         interpolation='bilinear',
                                                         cmap=color_map, 
                                                         #animated=True,
                                                         add_colorbar=False,
                                                         alpha=0.9)
        return image
    
    plt.close()
    animation = anim.FuncAnimation(fig, update, frames=range(len(times)), blit=False)
    writervideo = anim.FFMpegWriter(fps=5) 
    if level is not None:
        anim_file = os.path.join(os.environ['GCM_REPO_DIR'],f'ecrlgcm/data/figs/{field}_lev{level}_animation.mp4')
    else:    
        anim_file = os.path.join(os.environ['GCM_REPO_DIR'],f'ecrlgcm/data/figs/{field}_animation.mp4')
    animation.save(anim_file, writer=writervideo)
    print(anim_file)
    #return HTML(animation.to_jshtml())        

def get_continent_animation(times,stored_years):

    define_land_colormap()
    define_noaa_colormap()

    fig = plt.figure(figsize=(12,7))
    proj = ccrs.PlateCarree(central_longitude=0.0)
    ax = plt.axes(projection=proj)
    
    image = hires_interp(0,stored_years)['z'].plot.imshow(ax=ax, transform=proj, 
                                                        interpolation='bilinear',
                                                        #cmap="custom_earth", 
                                                        cmap='custom_noaa',
                                                        animated=True,
                                                        add_colorbar=False)
    border_img = hires_interp(0,stored_years)['mask'].plot.contour(ax=ax, levels=2, 
                                                                   transform=proj,#ccrs.PlateCarree(), 
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
        fig.suptitle(f'Time: {str(round(t,2))} Ma B.P.', fontsize=20)       
        ax.clear()
        image = current_field['z'].plot.imshow(ax=ax,transform=proj,
                                                 interpolation='bilinear',
                                                 #cmap="custom_earth",
                                                 cmap='custom_noaa',
                                                 animated=True,
                                                 add_colorbar=False)
        border_img = hires_interp(t,stored_years)['mask'].plot.contour(ax=ax, levels=2, 
                                                                       transform=proj,#ccrs.PlateCarree(), 
                                                                       colors="black", 
                                                                       add_colorbar=False, 
                                                                       alpha=1.0)
        return image
    
    plt.close()
    animation = anim.FuncAnimation(fig, update, frames=range(len(times)), blit=False)
    writervideo = anim.FFMpegWriter(fps=5) 
    anim_file = os.path.join(os.environ['GCM_REPO_DIR'],f'ecrlgcm/data/figs/continent_animation.mp4')
    animation.save(anim_file, writer=writervideo)
    print(anim_file)

def get_animation(exp,field='t_surf',level=None,vmin=None,vmax=None,anomaly=False):

    data = get_data(exp,field=field,level=level,decode_times=True,anomaly=anomaly)
    land = data['land_mask']
    
# Setup the initial plot
    if 'co2' in data:
        co2 = data['co2']

    variable = data[field]

    fig = plt.figure(figsize=(12,7))
    proj = ccrs.PlateCarree(central_longitude=180.0)
    ax = plt.axes(projection=proj)

    image = variable.mean(dim='time').plot.imshow(ax=ax, transform=proj, 
                                                  interpolation='bilinear',cmap="coolwarm", 
                                                  animated=True, add_colorbar=False)
    
    
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
        avg = variable.mean(dim=['time'])
        std = np.std(variable.values)
        
        vmin = variable.values.min()#avg.values.min()-2*std
        vmax = variable.values.max()#avg.values.max()+2*std

        print(f"plotting with vmin={vmin}, vmax={vmax}")
        image.set_clim(vmin,vmax)
    else:
        image.set_clim(vmin,vmax)
        
    text = cb.ax.xaxis.label
    font = matplotlib.font_manager.FontProperties(size=16)
    text.set_font_properties(font)
    
    def update(i):
        t = variable.time.values[i]
        if 'co2' in data:
            ax.set_title(f'time = {t.strftime("%B %Y")}, co2 = {sig_round(co2[i].values.mean(),4)} ({co2.units}), solar constant = {sig_round(solar_constant(exp.land_year),4)} (W/m**3)',fontsize=20)
        else:
            ax.set_title(f'time = {t.strftime("%B %Y")}',fontsize=20)
            
        image.set_array(variable.sel(time=t))
        return image
    
    plt.close()
    animation = anim.FuncAnimation(fig, update, frames=range(len(variable.time)), blit=False)
    writervideo = anim.FFMpegWriter(fps=5) 
    anim_file = os.path.join(os.environ['GCM_REPO_DIR'],f'ecrlgcm/postprocessing/anims/{exp.path_format}_{field}')
    if anomaly:
        anim_file += '_anomaly.mp4'
    else:
        anim_file += '.mp4'
    animation.save(anim_file, writer=writervideo)
    print(anim_file)
    #return HTML(animation.to_jshtml())        
