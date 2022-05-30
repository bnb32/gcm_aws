"""Setup GCM experiments"""
import xarray as xr
import os
import glob


class Experiment:
    """GCM experiment class"""
    def __init__(self, config, gcm_type='cesm', multiplier=1.0, land_year=0,
                 co2_value=None, res='f19_f19_mg17',
                 exp_type='cam_clmCN_docnDOM', sea_level=0,
                 max_depth=1000):
        self.multiplier = float(multiplier)
        self.land_year = land_year
        self.co2_value = co2_value
        if land_year == '0.02':
            self.sea_level = -100
        else:
            self.sea_level = sea_level
        self.max_depth = max_depth
        self.res = res
        self.exp_type = exp_type
        self.gcm_type = gcm_type
        self.high_res_file = f'{land_year}Ma_high_res.nc'
        self.remapped_f19 = f'remapped_{land_year}Ma_f19.nc'
        self.remapped_g16 = f'remapped_{land_year}Ma_g16.nc'
        self.remapped_f1 = f'remapped_{land_year}Ma_f1.nc'
        self.topo_file = f'topo_{land_year}Ma_{res}.nc'
        self.solar_file = f'solar_{land_year}Ma.nc'
        self.landfrac_file = f'landfrac_{land_year}Ma_{res}.nc'
        self.oceanfrac_file = f'oceanfrac_{land_year}Ma_{res}.nc'
        self.oceanmask_file = f'oceanmask_{land_year}Ma_{res}.ieeei4'
        self.oceantopo_file = f'oceantopo_{land_year}Ma_{res}.ieeei4'
        self.landplant_file = f'landplant_{land_year}Ma_{res}.nc'
        self.init_ocean_file = f'init_ocean_{land_year}Ma_{res}.ieeer8'
        self.docn_som_file = f'docn_som_{land_year}Ma_{res}.nc'
        self.docn_ocnfrac_file = f'docn_ocnfrac_{land_year}Ma_{res}.nc'
        self.docn_sst_file = f'docn_sst_{land_year}Ma_{res}.nc'
        self.init_atm_file = f'init_atm_{land_year}Ma_{res}.nc'
        self.high_res_file = os.path.join(config.HIGH_RES_TOPO_DIR,
                                          f'{land_year}Ma_high_res.nc')

        if gcm_type == 'isca':
            self.land_file = f'continents_{land_year}Ma.nc'
            self.base_file = f'{config.ISCA_TOPO_DIR}/{self.land_file}'
            self.topo_file = self.base_file
            if co2_value is not None:
                self.co2_file = f'co2_{co2_value}ppm_continents_'
                self.co2_file += f'{land_year}Ma.nc'
                self.name = f'variable_co2_{co2_value}ppm_continents_'
                self.name += f'{land_year}Ma_experiment'
                self.path_format = self.name
            else:
                self.co2_file = f'co2_{self.multiplier}x_continents_'
                self.co2_file += f'{land_year}Ma.nc'
                self.name = f'variable_co2_{self.multiplier}x_continents_'
                self.name += f'{land_year}Ma_experiment'
                self.path_format = self.name
            self.file_path = os.path.join(config.GFDL_DATA, self.path_format)
            self.files = sorted(glob.glob(
                os.path.join(self.file_path, 'run*/atmos_monthly.nc')))
            self.co2_file = f'{config.ISCA_CO2_DIR}/{self.co2_file}'

        elif gcm_type == 'cesm':
            if co2_value is not None:
                self.co2_file = f'co2_{co2_value}ppm_continents_'
                self.co2_file += f'{land_year}Ma_{res}.nc'
                self.name = f'{self.exp_type}_co2_{co2_value}ppm_continents_'
                self.name += f'{land_year}Ma_experiment'
                self.path_format = self.name
            else:
                self.co2_file = f'co2_{self.multiplier}x_continents_'
                self.co2_file += f'{land_year}Ma_{res}.nc'
                self.name = f'{self.exp_type}_co2_{self.multiplier}x_'
                self.name += f'continents_{land_year}Ma_experiment'
                self.path_format = self.name
            self.file_path = os.path.join(config.CIME_OUT_DIR,
                                          self.path_format)
            self.files = sorted(
                glob.glob(os.path.join(self.file_path,
                          f'atm/hist/{self.name}.cam.h0.*.nc')))
            self.run_files = sorted(glob.glob(
                os.path.join(self.file_path, f'run/{self.name}.cam.h0.*.nc')))

            self.solar_file = f'{config.CESM_SOLAR_DIR}/{self.solar_file}'
            self.topo_file = f'{config.CESM_TOPO_DIR}/{self.topo_file}'
            self.co2_file = f'{config.CESM_CO2_DIR}/{self.co2_file}'
            self.landfrac_file = os.path.join(config.CESM_LANDFRAC_DIR,
                                              self.landfrac_file)
            self.oceanfrac_file = os.path.join(config.CESM_OCEANFRAC_DIR,
                                               self.oceanfrac_file)
            self.oceanmask_file = os.path.join(config.CESM_OCEANFRAC_DIR,
                                               self.oceanmask_file)
            self.oceantopo_file = os.path.join(config.CESM_OCEANFRAC_DIR,
                                               self.oceantopo_file)
            self.docn_som_file = os.path.join(config.CESM_OCEANFRAC_DIR,
                                              self.docn_som_file)
            self.docn_sst_file = os.path.join(config.CESM_OCEANFRAC_DIR,
                                              self.docn_sst_file)
            self.docn_ocnfrac_file = os.path.join(config.CESM_OCEANFRAC_DIR,
                                                  self.docn_ocnfrac_file)
            self.docn_domain_file = os.path.join(
                config.CESM_INPUT_DATA_DIR,
                '/ocn/docn7/domain.ocn.1x1.111007.nc')
            self.init_ocean_file = os.path.join(config.CESM_OCEANFRAC_DIR,
                                                self.init_ocean_file)
            self.landplant_file = os.path.join(config.CESM_LANDFRAC_DIR,
                                               self.landplant_file)
            self.remapped_f19 = os.path.join(config.REMAPPED_LAND_DIR,
                                             self.remapped_f19)
            self.remapped_g16 = os.path.join(config.REMAPPED_LAND_DIR,
                                             self.remapped_g16)
            self.remapped_f1 = os.path.join(config.REMAPPED_LAND_DIR,
                                            self.remapped_f1)
            self.init_atm_file = os.path.join(config.INIT_CONDITIONS_DIR,
                                              self.init_atm_file)
            self.base_file = self.remapped_f19

    def base(self):
        """Get base data without manually introduced forcing"""
        return xr.open_mfdataset(self.base_file)

    def hires(self):
        """Get high resolution data"""
        return xr.open_mfdataset(self.high_res_file)

    def sim_data(self, **kwargs):
        """Get simulation output"""
        return xr.open_mfdataset(self.files, **kwargs)

    def run_data(self, **kwargs):
        """Get current simulation output"""
        return xr.open_mfdataset(self.run_files, **kwargs)

    def topo(self):
        """Get topography data"""
        return xr.open_mfdataset(self.topo_file, decode_times=False)

    def land(self):
        """Get land fraction data"""
        return xr.open_mfdataset(self.landfrac_file, decode_times=False)

    def ocean(self):
        """Get ocean fraction data"""
        return xr.open_mfdataset(self.oceanfrac_file, decode_times=False)

    def docn(self):
        """Get ocean fraction data for data ocean module"""
        return xr.open_mfdataset(self.docn_ocnfrac_file, decode_times=False)

    def solar_data(self):
        """Get solar data"""
        return xr.open_mfdataset(self.solar_file, decode_times=False)

    def co2_data(self):
        """Get co2 data"""
        return xr.open_mfdataset(self.co2_file, decode_times=False)


def Configuration(cesmexp, args):
    """Get simulation configuration"""
    if args.exp == 'cam_clmCN_docnDOM':
        sim_config = {}
        sim_config['atm'] = [f'bnd_topo="{cesmexp.topo_file}"',
                             f"solar_irrad_data_file='{cesmexp.solar_file}'",
                             f"co2vmr={args.co2_value}e-6",
                             f"co2vmr_rad={args.co2_value}e-6"]

        if args.step_type == 'ndays':
            sim_config['atm'].append('nhtfrq=-24')
            sim_config['atm'].append('mfilt=365')
        else:
            sim_config['atm'].append('nhtfrq=0')
            sim_config['atm'].append('mfilt=12')

        sim_config['lnd'] = ["urban_hac='OFF'",
                             f'fsurdat="{cesmexp.landplant_file}"']

        sim_config['docn'] = [f'domainfile="{cesmexp.oceanfrac_file}"']
        sim_config['change_som_stream'] = False
        sim_config['change_dom_stream'] = True

        sim_config['ice'] = ["ice_ic='none'", "xndt_dyn=2"]

        sim_config['cpl'] = ['orb_mode="fixed_year"', 'orb_iyear=1850',
                             'orb_eccen={eccentricity(args.year)}',
                             'orb_obliq={obliquity(args.year)}']

        sim_config['xml_changes'] = [
            f'NTASKS={args.ntasks}',
            f'STOP_OPTION={args.step_type}',
            f'STOP_N={args.nsteps}',
            f'REST_N={args.nsteps//5+1}',
            '--file env_build.xml DEBUG=TRUE',
            f'CCSM_CO2_PPMV={args.co2_value}',
            "CLM_CO2_TYPE='constant'",
            'EPS_FRAC=0.1',
            'ATM_DOMAIN_PATH=""',
            f'ATM_DOMAIN_FILE="{cesmexp.landfrac_file}"',
            'LND_DOMAIN_PATH=""',
            f'LND_DOMAIN_FILE="{cesmexp.landfrac_file}"',
            'OCN_DOMAIN_PATH=""',
            f'OCN_DOMAIN_FILE="{cesmexp.oceanfrac_file}"',
            'ICE_DOMAIN_PATH=""',
            f'ICE_DOMAIN_FILE="{cesmexp.oceanfrac_file}"',
            f'SSTICE_GRID_FILENAME="{cesmexp.docn_ocnfrac_file}"',
            f'SSTICE_DATA_FILENAME="{cesmexp.docn_sst_file}"']

        if args.timing:
            sim_config['xml_changes'].append("SAVE_TIMING='TRUE'")
            sim_config['xml_changes'].append("REST_OPTION='never'")
            sim_config['xml_changes'].append('STOP_N=5')
            sim_config['xml_changes'].append("STOP_OPTION='ndays'")
            sim_config['atm'].append("nhtfrq=0")

        sim_config['ocn'] = []

        sim_config['change_xml_cmd'] = ';'.join(
            f'./xmlchange {line}' for line in sim_config["xml_changes"])

    return sim_config
