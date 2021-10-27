import ecrlgcm.environment
from ecrlgcm.preprocessing import eccentricity,obliquity

import xarray as xr
import os
import glob

class Experiment:
    def __init__(self,gcm_type='cesm',multiplier=1,land_year=0,co2_value=None,res='f19_f19_mg17',exp_type='cam'):
        self.multiplier = float(multiplier)
        self.land_year = land_year
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
        #self.init_ocean_file = f'init_ocean_{land_year}Ma_{res}.nc'
        self.init_ocean_file = f'init_ocean_{land_year}Ma_{res}.ieeer8'
        self.docn_som_file = f'docn_som_{land_year}Ma_{res}.nc'
        self.docn_ocnfrac_file = f'docn_ocnfrac_{land_year}Ma_{res}.nc'
        self.docn_sst_file = f'docn_sst_{land_year}Ma_{res}.nc'

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
            self.docn_som_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.docn_som_file}'
            self.docn_sst_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.docn_sst_file}'
            self.docn_ocnfrac_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.docn_ocnfrac_file}'
            self.docn_domain_file = f'{os.environ["CESM_INPUT_DATA_DIR"]}/ocn/docn7/domain.ocn.1x1.111007.nc'
            self.init_ocean_file = f'{os.environ["CESM_OCEANFRAC_DIR"]}/{self.init_ocean_file}'
            self.landplant_file = f'{os.environ["CESM_LANDFRAC_DIR"]}/{self.landplant_file}'
            self.high_res_file = f'{os.environ["HIGH_RES_TOPO_DIR"]}/{self.high_res_file}'
            self.remapped_f19 = f'{os.environ["REMAPPED_LAND_DIR"]}/{self.remapped_f19}'
            self.remapped_g16 = f'{os.environ["REMAPPED_LAND_DIR"]}/{self.remapped_g16}'
            self.remapped_f1 = f'{os.environ["REMAPPED_LAND_DIR"]}/{self.remapped_f1}'

    def topo_data(self):
        return xr.open_mfdataset(self.topo_file,decode_times=False)
    
    def solar_data(self):
        return xr.open_mfdataset(self.solar_file,decode_times=False)
    
    def co2_data(self):
        return xr.open_mfdataset(self.co2_file,decode_times=False)

def Configuration(cesmexp,args):

    if args.exp=='cam_clmCN_docnDOM':
        sim_config = {}
        sim_config['atm'] = [
        f'bnd_topo="{cesmexp.topo_file}"',
        f"solar_irrad_data_file='{cesmexp.solar_file}'",
        f"co2vmr={args.co2_value}e-6",
        f"co2vmr_rad={args.co2_value}e-6",
        ]

        if args.step_type=='ndays':
            sim_config['atm'].append('nhtfrq=-24')
            sim_config['atm'].append('mfilt=365')
        else:
            sim_config['atm'].append('nhtfrq=0')
            sim_config['atm'].append('mfilt=12')

        sim_config['lnd'] = [
                f"urban_hac='OFF'",
                f'fsurdat="{cesmexp.landplant_file}"',
                ]

        sim_config['docn'] = [
                f'domainfile="{cesmexp.oceanfrac_file}"',
                ]
        sim_config['change_som_stream'] = False
        sim_config['change_dom_stream'] = True

        sim_config['ice'] = [
                f"ice_ic='none'",
                f"xndt_dyn=2",
                ]

        sim_config['cpl'] = [
                f'orb_mode="fixed_year"',
                f'orb_iyear=1850',
                f'orb_eccen={eccentricity(args.land_year)}',
                f'orb_obliq={obliquity(args.land_year)}',
                ]

        sim_config['xml_changes'] = [
                f'NTASKS={args.ntasks}',
                f'STOP_OPTION={args.step_type}',
                f'STOP_N={args.nsteps}',
                f'REST_N={args.nsteps//5+1}',
                f'--file env_build.xml DEBUG=TRUE',
                f'CCSM_CO2_PPMV={args.co2_value}',
                f"CLM_CO2_TYPE='constant'",
                f'EPS_FRAC=0.1',
                #f'ATM_GRID="1.9x2.5"',
                #f'LND_GRID="1.9x2.5"',
                #f'OCN_GRID="1.9x2.5"',
                #f'ICE_GRID="1.9x2.5"',
                #f'MASK_GRID="1.9x2.5"',
                #f'DOCN_SOM_FILENAME="{cesmexp.docn_som_file}"',
                f'ATM_DOMAIN_PATH=""',
                f'ATM_DOMAIN_FILE="{cesmexp.landfrac_file}"',
                f'LND_DOMAIN_PATH=""',
                f'LND_DOMAIN_FILE="{cesmexp.landfrac_file}"',
                f'OCN_DOMAIN_PATH=""',
                f'OCN_DOMAIN_FILE="{cesmexp.oceanfrac_file}"',
                f'ICE_DOMAIN_PATH=""',
                f'ICE_DOMAIN_FILE="{cesmexp.oceanfrac_file}"',
                f'SSTICE_GRID_FILENAME="{cesmexp.docn_ocnfrac_file}"',
                f'SSTICE_DATA_FILENAME="{cesmexp.docn_sst_file}"',
                ]

        sim_config['ocn'] = [
        #f'overflows_on=.false.',
        #f'overflows_interactive=.false.',
        #f'ltidal_mixing=.false.',
        #f'lhoriz_varying_bckgrnd=.false.',
        #f'region_mask_file="{cesmexp.oceanmask_file}"',
        #f'region_info_file="{os.environ["OCEAN_REGION_MASK_FILE"]}"',
        #f'topography_file="{cesmexp.oceantopo_file}"',
        #f"lat_aux_grid_type='user-specified'",
        #f"lat_aux_begin=-90.0",
        #f"lat_aux_end=90.0",
        #f"n_lat_aux_grid=180",
        #f"n_heat_trans_requested=.true.",
        #f"n_salt_trans_requested=.true.",
        #f"n_transport_reg=1",
        #f"moc_requested=.true.",
        #f"dt_count=30",
        #f'ldiag_velocity=.false.',
        #f'bckgrnd_vdc1=0.524',
        #f'bckgrnd_vdc2=0.313',
        #f"sw_absorption_type='jerlov'",
        #f'jerlov_water_type=3',
        #f"chl_option='file'",
        #f"chl_filename='unknown-chl'",
        #f"chl_file_fmt='bin'",
        #f"init_ts_option='mean'",
        #f"init_ts_file_fmt='bin'",
        #f"init_ts_option='ccsm_startup'",
        #f'init_ts_file="{cesmexp.init_ocean_file}"',
        ]

        sim_config['change_xml_cmd'] = ';'.join(f'./xmlchange {l}' for l in sim_config["xml_changes"])

    return sim_config    
