import ecrlgcm.environment
import xarray as xr

import os
import glob

class Experiment:
    def __init__(self,gcm_type='cesm',multiplier=1,land_year=0,co2_value=None,res='T42_T42',exp_type='cam'):
        self.multiplier = float(multiplier)
        self.land_year = land_year
        self.res = res
        self.exp_type = exp_type
        self.gcm_type = gcm_type
        self.high_res_file = f'{land_year}Ma_high_res.nc'
        self.remapped_f19 = f'remapped_{land_year}Ma_f19.nc'
        self.remapped_g16 = f'remapped_{land_year}Ma_g16.nc'
        self.remapped_1x1d = f'remapped_{land_year}Ma_1x1d.nc'
        self.topo_file = f'topo_{land_year}Ma_{res}.nc'
        self.solar_file = f'solar_{land_year}Ma.nc'
        self.landfrac_file = f'landfrac_{land_year}Ma_{res}.nc'
        self.oceanfrac_file = f'oceanfrac_{land_year}Ma_{res}.nc'
        self.oceanmask_file = f'oceanmask_{land_year}Ma_{res}.ieeei4'
        self.oceantopo_file = f'oceantopo_{land_year}Ma_{res}.ieeei4'
        self.landplant_file = f'landplant_{land_year}Ma_{res}.nc'
        #self.init_ocean_file = f'init_ocean_{land_year}Ma_{res}.nc'
        self.init_ocean_file = f'init_ocean_{land_year}Ma_{res}.ieeer8'
        
        if co2_value is not None:
            self.co2_file = f'co2_{co2_value}ppm_continents_{land_year}Ma_{res}.nc'
            self.name = self.path_format = f'{exp_type}_co2_{co2_value}ppm_continents_{land_year}Ma_experiment'
        else:
            self.co2_file = f'co2_{multiplier}x_continents_{land_year}Ma_{res}.nc'
            self.name = self.path_format = f'{exp_type}_co2_{multiplier}x_continents_{land_year}Ma_experiment'
        
        if gcm_type=='isca':
            self.file_path = os.path.join(os.environ['GFDL_DATA'],self.path_format)
            self.files = sorted(glob.glob(os.path.join(self.file_path,'run*/atmos_monthly.nc')))
        elif gcm_type=='cesm':
            self.file_path = os.path.join(os.environ['CIME_OUT_DIR'],self.path_format)
            self.files = sorted(glob.glob(os.path.join(self.file_path,f'atm/hist/{self.name}.cam.h0.*.nc')))
            self.solar_file = f'{os.environ["CESM_SOLAR_DIR"]}/{self.solar_file}'
            self.topo_file = f'{os.environ["CESM_TOPO_DIR"]}/{self.topo_file}'
            self.co2_file = f'{os.environ["CESM_CO2_DIR"]}/{self.co2_file}'
            self.landfrac_file = f'{os.environ["CESM_LANDFRAC_DIR"]}/{self.landfrac_file}'
            self.oceanfrac_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.oceanfrac_file}'
            self.oceanmask_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.oceanmask_file}'
            self.oceantopo_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.oceantopo_file}'
            self.init_ocean_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.init_ocean_file}'
            self.landplant_file = f'{os.environ["CESM_LANDFRAC_DIR"]}/{self.landplant_file}'
            self.high_res_file = f'{os.environ["HIGH_RES_TOPO_DIR"]}/{self.high_res_file}'
            self.remapped_f19 = f'{os.environ["REMAPPED_LAND_DIR"]}/{self.remapped_f19}'
            self.remapped_g16 = f'{os.environ["REMAPPED_LAND_DIR"]}/{self.remapped_g16}'
            self.remapped_1x1d = f'{os.environ["REMAPPED_LAND_DIR"]}/{self.remapped_1x1d}'

    def topo_data(self):
        return xr.open_mfdataset(self.topo_file,decode_times=False)
    
    def solar_data(self):
        return xr.open_mfdataset(self.solar_file,decode_times=False)
    
    def co2_data(self):
        return xr.open_mfdataset(self.co2_file,decode_times=False)
