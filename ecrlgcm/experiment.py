import ecrlgcm.environment

import os
import glob

class Experiment:
    def __init__(self,type='isca',multiplier=1,land_year=0,co2_value=None,res='T42_T42'):
        self.multiplier = float(multiplier)
        self.land_year = land_year
        self.res = res
        self.land_file = f'continents_{land_year}Ma_{res}.nc'
        self.path_format = f'variable_co2_{self.multiplier}x_continents_{self.land_year}Ma_experiment'
        self.name = f'variable_co2_{multiplier}x_continents_{land_year}Ma_experiment'
        
        if type=='isca':
            self.file_path = os.path.join(os.environ['GFDL_DATA'],self.path_format)
            self.files = sorted(glob.glob(os.path.join(self.file_path,'run*/atmos_monthly.nc')))
        elif type=='cesm':
            self.file_path = os.path.join(os.environ['CIME_OUT_DIR'],self.path_format)
            self.files = sorted(glob.glob(os.path.join(self.file_path,f'atm/hist/{self.name}.cam.h0.*.nc')))

        if co2_value is not None:
            self.co2_file = f'co2_{co2_value}ppm_continents_{land_year}Ma_{res}.nc'
        else:
            self.co2_file = f'co2_{multiplier}x_continents_{land_year}Ma_{res}.nc'
        
