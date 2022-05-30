"""Ecrlgcm preprocessing"""
import os
import netCDF4 as nc
import numpy as np
import glob
import xarray as xr
import xesmf as xe
import warnings
from tqdm import tqdm
from metpy.calc import smooth_n_point

from ecrlgcm.data import co2_series, ecc_series, obl_series
from ecrlgcm.utilities import interp, get_logger, sliding_std
warnings.filterwarnings("ignore")

logger = get_logger()


def eccentricity(land_year):
    """Get orbital eccentricity for a requested year"""
    return interpolate_ecc(land_year)


def obliquity(land_year):
    """Get orbital obliquity for a requested year"""
    return interpolate_obl(land_year)


def solar_fraction(land_year):
    """Get fraction of current solar constant for a requested year"""
    time = -float(land_year) / 4700.0
    return 1.0 / (1 - 0.4 * time)


def solar_constant(land_year):
    """Get solar constant for a requested year"""
    return 1370.0 * solar_fraction(land_year)


def interpolate_series(land_year, series):
    """Interpolate time series to needed simulation year"""
    year = float(land_year)
    keys = sorted(series)
    if land_year in keys:
        return series[keys[keys.index(land_year)]]

    elif year <= keys[0]:
        return series[keys[0]]

    elif year >= keys[-1]:
        return series[keys[-1]]

    else:
        for i in range(len(keys)):
            if keys[i] <= year <= keys[i + 1]:
                return interp(series[keys[i]],
                              series[keys[i + 1]],
                              (year - keys[i]) / (keys[i + 1] - keys[i]))


def interpolate_co2(land_year):
    """Interpolate co2 for a requested year"""
    return interpolate_series(land_year, co2_series)


def interpolate_ecc(land_year):
    """Interpolate orbital eccentricity for a requested year"""
    return interpolate_series(land_year, ecc_series)


def interpolate_obl(land_year):
    """Interpolate orbital obliquity for a requested year"""
    return interpolate_series(land_year, obl_series)


class PreProcessing:
    """PreProcessing class"""
    def __init__(self, config):
        self.config = config
        out = self.get_global_vars()
        self.land_years, self.stored_years = out[:2]
        self.min_land_year, self.max_land_year = out[2:]

    def modify_isca_input_files(self, ecrlexp, remap=False):
        """Modify input data file for requested experiment"""
        multiplier = ecrlexp.multiplier
        land_year = ecrlexp.land_year
        sea_level = ecrlexp.sea_level
        co2_value = ecrlexp.co2_value

        if not os.path.exists(ecrlexp.co2_file) or remap:
            logger.info('Modifying co2_file')
            os.system(
                f'cp {self.config.ORIG_ISCA_CO2_FILE} {ecrlexp.co2_file}')
            ncfile = nc.Dataset(ecrlexp.co2_file, 'r+')
            co2 = ncfile.variables['co2']

            if co2_value is None:
                co2[:, :, :, :] = interpolate_co2(land_year)
                co2 *= float(multiplier)
            else:
                co2[:, :, :, :] = float(co2_value)
            ncfile.variables['co2'][:, :, :, :] = co2[:, :, :, :]
            ncfile.close()
            logger.info(f'Saving {ecrlexp.co2_file}')

        if not os.path.exists(ecrlexp.topo_file) or remap:
            logger.info('Modifying topo_file')
            land = self.interpolate_land(land_year)
            base = xr.open_mfdataset(self.config.ORIG_ISCA_TOPO_FILE)

            ds_out = xr.Dataset({'lat': (['lat'], base['lat'].values),
                                 'lon': (['lon'], base['lon'].values)})

            if not os.path.exists(ecrlexp.high_res_file):
                land.to_netcdf(ecrlexp.high_res_file)
                logger.info(
                    f'Saving high res map file: {ecrlexp.high_res_file}')

            regridder = xe.Regridder(land, ds_out, 'bilinear')
            land['z'] = (land['z'].dims,
                         np.where(land['z'].values > sea_level,
                         land['z'].values, 0))
            logger.info('Regridding topo_file')
            ds_out = regridder(land)
            ds_out['land_mask'] = (ds_out['z'].dims,
                                   np.where(ds_out['z'].values > 0.0,
                                   1.0, 0.0))
            ds_out = ds_out.rename({'z': 'zsurf'})
            ds_out = ds_out.fillna(0)
            ds_out.to_netcdf(ecrlexp.topo_file)
            logger.info(f'Saving map file: {ecrlexp.topo_file}')

    def regrid_high_res_data(self, cesmexp, land, remap=True):
        """Regrid high res data on grid used for simulation"""
        sea_level = cesmexp.sea_level
        max_depth = cesmexp.max_depth

        if not os.path.exists(cesmexp.remapped_f19) or remap:
            basefile = xr.open_mfdataset(self.config.ORIG_CESM_TOPO_FILE)
            ds_out = xr.Dataset({'lat': (['lat'], basefile['lat'].values),
                                 'lon': (['lon'], basefile['lon'].values)})

            logger.info('Regridding high res file to fv1.9x2.5')
            regridder = xe.Regridder(land, ds_out, 'bilinear',
                                     ignore_degenerate=True)
            land = regridder(land)

            landfrac = smooth_n_point(land['landfrac'].values, 5)
            landmask = np.where(landfrac > 0.0, 1.0, 0.0)
            oceanfrac = 1-landfrac
            oceanfrac = np.where(oceanfrac > 1.0, 1.0, oceanfrac)
            oceanfrac = np.where(oceanfrac < 0.0, 0.0, oceanfrac)
            oceanmask = np.where(oceanfrac > 0.0, 1.0, 0.0)
            height = np.where(land['z'].values > sea_level,
                              land['z'].values, 0)
            depth = np.where(land['z'].values <= sea_level,
                             -land['z'].values, 0)
            depth = np.where(depth > max_depth, max_depth, depth)

            land['landmask'] = (land['z'].dims, landmask)
            land['landfrac'] = (land['z'].dims, landfrac)
            land['oceanmask'] = (land['z'].dims, oceanmask)
            land['oceanfrac'] = (land['z'].dims, oceanfrac)
            land['height'] = (land['z'].dims, height)
            land['PHIS'] = (land['z'].dims, 9.8 * land['z'].values)
            land['depth'] = (land['z'].dims, depth)
            z_attrs = land['z'].attrs
            land['z'] = (land['z'].dims, land['z'].values)
            land['z'].attrs = z_attrs

            land.to_netcdf(cesmexp.remapped_f19)
            logger.info(f'Saved {cesmexp.remapped_f19}')

    def regrid_high_res_data_ncl(self, cesmexp, in_shape=(1801, 3601),
                                 remap=True):
        """Regrid high res data on grid used for simulation. Use ncl script"""

        cmd = f'export NCL_POP_REMAP="{self.config.USER_DIR}'
        cmd += '/inputdata/mapping_files"; '
        cmd += f'rm -f "%s"; ncl infile=\'"{cesmexp.high_res_file}"\' '
        cmd += 'outfile=\'"%s"\' '
        cmd += f'res=\'"%s"\' n_lat_in={in_shape[0]} n_lon_in={in_shape[1]} '
        cmd += f'{self.config.NCL_SCRIPTS}/remap_high_res.ncl'

        if not os.path.exists(cesmexp.remapped_f19) or remap:
            logger.info('Regridding high res file to fv1.9x2.5')
            logger.info(
                cmd, (cesmexp.remapped_f19, cesmexp.remapped_f19, "f19"))
            os.system(
                cmd, (cesmexp.remapped_f19, cesmexp.remapped_f19, "f19"))
            logger.info(f'Saved {cesmexp.remapped_f19}')

    def get_original_map_data(self, land_year):
        """Get original map for requested year"""
        land_year = str(land_year)
        file = glob.glob(self.config.RAW_TOPO_DIR + f'/Map*_{land_year}Ma.nc')
        land = xr.open_mfdataset(file)
        return land

    def modify_solar_file(self, cesmexp, remap):
        """Modify input solar file with forcing specified in cesmexp"""
        if not os.path.exists(cesmexp.solar_file) or remap:
            logger.info('Modifying solar_file')
            solar_frac = solar_fraction(cesmexp.land_year)
            f = xr.open_mfdataset(self.config.ORIG_CESM_SOLAR_FILE,
                                  decode_times=False)
            f['ssi'] = (f['ssi'].dims, solar_frac * f['ssi'].values)
            f.to_netcdf(cesmexp.solar_file)
            logger.info(f'Saved solar_file: {cesmexp.solar_file}')

    def interpolate_land(self, land_year):
        year = float(land_year)
        keys = sorted(self.land_years)

        if land_year in keys:
            ds_out = self.get_original_map_data(keys[keys.index(land_year)])

        elif year <= keys[0]:
            ds_out = self.get_original_map_data(keys[0])

        elif year >= keys[-1]:
            ds_out = self.get_original_map_data(keys[-1])

        else:
            for i, _ in enumerate(keys):
                if keys[i] <= year <= keys[i + 1]:
                    ds_out = self.get_original_map_data(keys[i])
                    tmp = interp(
                        self.get_original_map_data(keys[i])['z'].values,
                        self.get_original_map_data(keys[i + 1])['z'].values,
                        (year - keys[i]) / (keys[i + 1] - keys[i]))
                    ds_out['z'] = (ds_out['z'].dims, tmp)

        ds_out = ds_out.rename({'latitude': 'lat', 'longitude': 'lon'})
        return ds_out

    def modify_cesm_input_files(self, cesmexp, remap=True, remap_hires=True):

        land_year = cesmexp.land_year
        sea_level = cesmexp.sea_level
        max_depth = cesmexp.max_depth
        base_topofile = self.config.ORIG_CESM_TOPO_FILE

        if not os.path.exists(cesmexp.high_res_file) or remap_hires:
            logger.info("Computing land and ocean masks")
            raw_topo_data = compute_land_ocean_properties(
                self.interpolate_land(land_year), sea_level=sea_level,
                max_depth=max_depth)
            raw_topo_data.to_netcdf(cesmexp.high_res_file)
            logger.info(f'Saving high res map file: {cesmexp.high_res_file}')
            self.regrid_high_res_data(cesmexp, raw_topo_data, remap=remap)

        if not os.path.exists(cesmexp.topo_file) or remap:

            logger.info('Modifying topo_file')
            f19_data = xr.open_mfdataset(cesmexp.remapped_f19)
            topo_data = xr.open_mfdataset(base_topofile)
            topo_data['PHIS'] = (topo_data['PHIS'].dims,
                                 9.8 * f19_data['height'].values)
            topo_data['LANDFRAC'] = (topo_data['LANDFRAC'].dims,
                                     f19_data['landfrac'].values)
            topo_data['LANDM_COSLAT'] = (topo_data['LANDM_COSLAT'].dims,
                                         f19_data['landmask'].values)
            topo_data['SGH'] = (topo_data['SGH'].dims,
                                sliding_std(f19_data['height'].values))
            topo_data['SGH30'] = (topo_data['SGH30'].dims,
                                  sliding_std(f19_data['height'].values))
            topo_data.to_netcdf(cesmexp.topo_file)
            logger.info(f'Saved topo_file: {cesmexp.topo_file}')

        if not os.path.exists(cesmexp.landfrac_file) or remap:
            logger.info('Modifying landfrac_file')
            orig_land = xr.open_mfdataset(self.config.ORIG_CESM_LANDFRAC_FILE)
            orig_mask = orig_land['mask'].values.copy()
            landfrac = topo_data['LANDFRAC'].values
            landfrac = np.where(landfrac > 0.0, landfrac, 0.0)
            orig_land['frac'] = (orig_land['frac'].dims, landfrac)
            orig_land['mask'] = (
                orig_land['mask'].dims,
                np.where(orig_land['frac'] > 0, 1, 0).astype(np.int32))
            orig_land.to_netcdf(cesmexp.landfrac_file)
            logger.info(f'Saving landfrac_file: {cesmexp.landfrac_file}')

        if not os.path.exists(cesmexp.oceanfrac_file) or remap:
            logger.info('Modifying oceanfrac_file')
            orig_ocean = xr.open_mfdataset(
                self.config.ORIG_CESM_OCEANFRAC_FILE)
            oceanfrac = 1 - orig_land['frac'].values
            oceanfrac = np.where(oceanfrac > 1, 1, oceanfrac)
            oceanfrac = np.where(oceanfrac < 0, 0, oceanfrac)
            orig_ocean['frac'] = (orig_ocean['frac'].dims, oceanfrac)
            orig_ocean['mask'] = (
                orig_ocean['mask'].dims,
                np.where(
                    orig_ocean['frac'].values > 0.0, 1, 0).astype(np.int32))
            orig_ocean.to_netcdf(cesmexp.oceanfrac_file)
            logger.info(f'Saving oceanfrac_file: {cesmexp.oceanfrac_file}')

        if not os.path.exists(cesmexp.docn_ocnfrac_file) or remap:
            logger.info('Modifying docn_ocnfrac_file')
            orig_docn = xr.open_mfdataset(self.config.ORIG_DOCN_OCNFRAC_FILE)
            orig_docn['frac'] = (orig_docn['frac'].dims,
                                 orig_ocean['frac'].values)
            orig_docn['mask'] = (orig_docn['mask'].dims,
                                 orig_ocean['mask'].values)
            orig_docn.to_netcdf(cesmexp.docn_ocnfrac_file)
            logger.info(
                f'Saving docn_ocnfrac_file: {cesmexp.docn_ocnfrac_file}')

        if not os.path.exists(cesmexp.docn_sst_file) or remap:
            logger.info('Modifying docn_sst_file')
            orig_sst = xr.open_mfdataset(self.config.ORIG_DOCN_SST_FILE)
            orig_sst['SST_cpl'] = (
                orig_sst['SST_cpl'].dims, modify_array_with_time(
                    orig_sst['SST_cpl'].values, orig_mask,
                    orig_land['mask'].values))
            orig_sst['ice_cov'] = (
                orig_sst['ice_cov'].dims, modify_array_with_time(
                    orig_sst['ice_cov'].values, orig_mask,
                    orig_land['mask'].values))
            orig_sst.to_netcdf(cesmexp.docn_sst_file)
            logger.info(f'Saving docn_sst_file: {cesmexp.docn_sst_file}')

        if not os.path.exists(cesmexp.landplant_file) or remap:
            logger.info('Modifying landplant_file')
            landplant_data = xr.open_mfdataset(
                self.config.ORIG_CESM_LANDPLANT_FILE)
            frac_attrs = landplant_data['LANDFRAC_PFT'].attrs
            mask_attrs = landplant_data['PFTDATA_MASK'].attrs

            landplant_data = modify_all_arrays_with_mask(
                landplant_data, landplant_data['PFTDATA_MASK'].values,
                orig_land['mask'].values)
            landplant_data = modify_all_arrays_with_mask(
                landplant_data, landplant_data['PFTDATA_MASK'].values,
                orig_land['mask'].values)

            landplant_data['LANDFRAC_PFT'] = (
                landplant_data['LANDFRAC_PFT'].dims, orig_land['frac'].values)
            landplant_data['LANDFRAC_PFT'].attrs = frac_attrs
            landplant_data['PFTDATA_MASK'] = (
                landplant_data['PFTDATA_MASK'].dims, orig_land['mask'].values)
            landplant_data['PFTDATA_MASK'].attrs = mask_attrs

            landplant_data.to_netcdf(cesmexp.landplant_file)
            logger.info(f'Saved landplant_file: {cesmexp.landplant_file}')

        self.modify_solar_file(cesmexp, remap)

    def regrid_domain_files(self, cesmexp):

        logger.info("Modifying landfrac_file")
        orig_land = xr.open_mfdataset(self.config.ORIG_CESM_LANDFRAC_FILE)
        orig_land['frac'] = (orig_land['frac'].dims, orig_land['mask'].values)
        orig_land = orig_land.fillna(0)
        orig_land.to_netcdf(cesmexp.landfrac_file)
        logger.info(f'Saving landfrac_file: {cesmexp.landfrac_file}')

        logger.info("Modifying oceanfrac_file")
        orig_ocean = xr.open_mfdataset(self.config.ORIG_CESM_OCEANFRAC_FILE)
        orig_ocean['area'] = (
            orig_ocean['area'].dims, np.array(orig_land['area'].values))
        orig_ocean['mask'] = (
            orig_ocean['mask'].dims,
            np.array(1 - orig_land['mask'].values, dtype=np.int32))
        orig_ocean['frac'] = (
            orig_ocean['mask'].dims,
            np.array(1-orig_land['mask'].values, dtype=np.int32))
        orig_ocean = orig_ocean.fillna(0)
        orig_ocean.to_netcdf(cesmexp.oceanfrac_file)
        logger.info(f'Saving oceanfrac_file: {cesmexp.oceanfrac_file}')

        logger.info('Modifying docn_ocnfrac_file')
        orig_docn = xr.open_mfdataset(self.config.ORIG_DOCN_OCNFRAC_FILE)
        orig_docn['mask'] = (
            orig_docn['mask'].dims,
            np.array(1 - orig_land['mask'].values, dtype=np.int32))
        orig_docn = orig_docn.fillna(0)
        orig_docn.to_netcdf(cesmexp.docn_ocnfrac_file)
        logger.info(f'Saving docn_ocnfrac_file: {cesmexp.docn_ocnfrac_file}')

        logger.info('Modifying docn_sst_file')
        orig_sst = xr.open_mfdataset(self.config.ORIG_DOCN_SST_FILE)
        orig_sst['SST_cpl'] = (orig_sst['SST_cpl'].dims,
                               orig_sst['SST_cpl'].values)
        orig_sst['ice_cov'] = (orig_sst['ice_cov'].dims,
                               orig_sst['ice_cov'].values)
        orig_sst = orig_sst.fillna(0)
        orig_sst.to_netcdf(cesmexp.docn_sst_file)
        logger.info(f'Saving docn_sst_file: {cesmexp.docn_sst_file}')

    def modify_co2_file(self, cesmexp, remap):

        if not os.path.exists(cesmexp.co2_file) or remap:
            logger.info('Modifying co2_file')
            co2value = interpolate_co2(cesmexp.land_year)
            f=xr.open_mfdataset(self.config.ORIG_CESM_CO2_FILE,
                                decode_times=False)
            tmp = np.full(f['co2vmr'].shape,
                          cesmexp.multiplier * co2value * 1.0e-6)
            f['co2vmr'] = (f['co2vmr'].dims, tmp)
            f.to_netcdf(cesmexp.co2_file)
            logger.info(f'Saved co2_file: {cesmexp.co2_file}')


    def get_landfrac_dict(self, orig_data=None, source_data=None,
                          land_year=None, sea_level=0):

        landfrac_dict = {}
        if land_year is not None:
            source_data = self.interpolate_land(land_year)
        raw_landmask = np.array(source_data['z'].values > sea_level,
                                dtype=float)
        logger.info('Calculating landfrac_dict')
        outlat = orig_data['yc'].values[:, 0]
        outlon = orig_data['xc'].values[0, :]

        if len(source_data['lat'].shape) == 2:
            inlat = source_data['lat'][:, 0].values
            inlon = source_data['lon'][0, :].values
        else:
            inlat = source_data['lat'].values
            inlon = source_data['lon'].values

        return source_data, overlap_fraction(inlat, inlon, outlat, outlon,
                                             raw_landmask)

    def edit_namelists(self, experiment, configuration):
        """Edit namelists for simulations"""
        case = f'{self.config.CESM_REPO_DIR}/cases/{experiment.name}'
        nl_cam_file = f"{case}/user_nl_cam"
        nl_cpl_file = f"{case}/user_nl_cpl"
        nl_clm_file = f"{case}/user_nl_clm"
        nl_pop_file = f"{case}/user_nl_pop"
        nl_docn_file = f"{case}/user_nl_docn"
        nl_ice_file = f"{case}/user_nl_ice"

        logger.info(f"**Changing namelist file: {nl_cam_file}**")
        with open(nl_cam_file, 'w') as f:
            for l in configuration['atm']:
                f.write(f'{l}\n')
        f.close()

        logger.info(f"**Changing namelist file: {nl_cpl_file}**")
        with open(nl_cpl_file, 'w') as f:
            for l in configuration['cpl']:
                f.write(f'{l}\n')
        f.close()

        logger.info(f"**Changing namelist file: {nl_clm_file}**")
        with open(nl_clm_file, 'w') as f:
            for l in configuration['lnd']:
                f.write(f'{l}\n')
        f.close()

        logger.info(f"**Changing namelist file: {nl_docn_file}**")
        with open(nl_docn_file, 'w') as f:
            for l in configuration['docn']:
                f.write(f'{l}\n')
        f.close()

        logger.info(f"**Changing namelist file: {nl_pop_file}**")
        with open(nl_pop_file, 'w') as f:
            for l in configuration['ocn']:
                f.write(f'{l}\n')
        f.close()

        logger.info(f"**Changing namelist file: {nl_ice_file}**")
        with open(nl_ice_file, 'w') as f:
            for l in configuration['ocn']:
                f.write(f'{l}\n')
        f.close()

        if configuration['change_som_stream']:
            logger.info("**Changing docn dom stream file**")
            with open(f'{self.config.GCM_REPO_DIR}'
                      '/templates/user_docn.streams.txt.som') as f:
                docn_file = f.read()

            os.system(f'cp {self.config.GCM_REPO_DIR}'
                      '/templates/user_docn.streams.txt.som {case}')
            with open(f'{case}/user_docn.streams.txt.som', 'w') as f:
                tmp_file = experiment.docn_som_file.split('/')[-1]
                tmp_path = experiment.docn_som_file.split('/')[:-1]
                tmp_path = '/'.join(tmp_path)
                f.write(docn_file.replace('%DOCN_SOM_FILE%',
                                          tmp_file).replace(
                                              '%DOCN_SOM_DIR%', tmp_path))

        if configuration['change_dom_stream']:
            logger.info("**Changing docn dom stream file**")
            tmp = os.path.join(self.config.GCM_REPO_DIR,
                               '/templates/user_docn.streams.txt.prescribed')
            with open(tmp) as f:
                docn_file = f.read()

            os.system(f'cp {self.config.GCM_REPO_DIR}'
                      '/templates/user_docn.streams.txt.prescribed {case}')
            with open(f'{case}/user_docn.streams.txt.prescribed', 'w') as f:
                sst_file = experiment.docn_sst_file.split('/')[-1]
                sst_path = experiment.docn_sst_file.split('/')[:-1]
                sst_path = '/'.join(sst_path)
                ocnfrac_file = experiment.docn_ocnfrac_file.split('/')[-1]
                ocnfrac_path = experiment.docn_ocnfrac_file.split('/')[:-1]
                ocnfrac_path = '/'.join(ocnfrac_path)
                f.write(docn_file.replace(
                    '%DOCN_SST_FILE%', sst_file).replace(
                        '%DOCN_SST_DIR%', sst_path).replace(
                            '%DOCN_OCNFRAC_FILE%', ocnfrac_file).replace(
                                '%DOCN_OCNFRAC_DIR%', ocnfrac_path))

    def get_base_topofile(self, res):
        tmp_res = res.replace('.', '').split('_')[0]
        if '42' in tmp_res:
            self.config.ORIG_CESM_TOPO_FILE = self.config.T42_TOPO_FILE
        if '19' in tmp_res:
            self.config.ORIG_CESM_TOPO_FILE = self.config.f19_TOPO_FILE
        if '09' in tmp_res:
            self.config.ORIG_CESM_TOPO_FILE = self.config.f09_TOPO_FILE
        return self.config.ORIG_CESM_TOPO_FILE

    def get_global_vars(self):
        """Get global vars which depend on previous runs

        Returns
        -------
        land_years: list
            List of year with saved maps
        stored_years : list
            List of previously simulated years
        min_land_year : float
            Min year for available map
        max_land_year : float
            Max year for available map
        """
        land_years = glob.glob(self.config.RAW_TOPO_DIR + '/Map*.nc')
        land_years = sorted([float(line.strip('Ma.nc').split('_')[-1])
                             for line in land_years])
        land_years = [int(x) if int(x) == float(x) else float(x)
                      for x in land_years]

        stored_years = glob.glob(self.config.CIME_OUT_DIR
                                 + '/*/atm/hist/*cam.*.nc')
        stored_years = sorted([float(line.split('_')[-2].strip('Ma'))
                               for line in stored_years])
        stored_years = [int(x) if int(x) == float(x) else float(x)
                        for x in stored_years]

        min_land_year = min(land_years)
        max_land_year = max(land_years)
        return land_years, stored_years, min_land_year, max_land_year


def regrid_continent_maps(remap_file, basefile='', outfile='', sea_level=0,
                          max_depth=1000):
    land = xr.open_mfdataset(remap_file)
    ds_out = regrid_continent_data(land, basefile=basefile,
                                   sea_level=sea_level,
                                   max_depth=max_depth)

    os.system(f'rm -f {outfile}')
    ds_out.to_netcdf(outfile)
    print(f'{outfile}')

def shift_longitude(data):

    logger.info("Shifting longitude in high res data")
    tmp = np.zeros(data['z'].shape)
    lons = [x + 360.0 if x < 0 else x for x in data['lon'].values]
    lats = sorted(data['lat'].values)
    sorted_lons = sorted(lons)
    for i, lon in enumerate(sorted_lons):
        idx = lons.index(lon)
        tmp[:, i] = data['z'].values[::-1, idx]

    lon_attrs = data['lon'].attrs
    lat_attrs = data['lat'].attrs
    z_attrs = data['z'].attrs

    data['lon'] = (data['lon'].dims, sorted_lons)
    data['lon'].attrs = lon_attrs
    data['lat'] = (data['lat'].dims, lats)
    data['lat'].attrs = lat_attrs
    data['z'] = (data['z'].dims, tmp)
    data['z'].attrs = z_attrs
    return data

def compute_land_ocean_properties(land, sea_level=0, max_depth=1000):

    if 'latitude' in land:
        land = land.rename({'latitude': 'lat'})
    if 'longitude' in land:
        land = land.rename({'longitude': 'lon'})

    land = shift_longitude(land)
    land['z_non_smooth'] = (land['z'].dims, land['z'].values)
    land['z'] = (land['z'].dims, smooth_n_point(land['z'].values, 9))
    landfrac = np.where(land['z'].values > sea_level, 1.0, 0.0)
    landmask = np.where(landfrac > 0.0, 1.0, 0.0)
    oceanfrac = 1 - landfrac
    oceanfrac = np.where(oceanfrac > 1.0, 1.0, oceanfrac)
    oceanfrac = np.where(oceanfrac < 0.0, 0.0, oceanfrac)
    oceanmask = np.where(oceanfrac > 0.0, 1.0, 0.0)
    height = np.where(land['z'].values > sea_level, land['z'].values, 0)
    depth = np.where(land['z'].values <= sea_level, -land['z'].values, 0)
    depth = np.where(depth>max_depth, max_depth, depth)

    land['landmask'] = (land['z'].dims, landmask)
    land['landfrac'] = (land['z'].dims, landfrac)
    land['oceanmask'] = (land['z'].dims, oceanmask)
    land['oceanfrac'] = (land['z'].dims, oceanfrac)
    land['height'] = (land['z'].dims, height)
    land['PHIS'] = (land['z'].dims, 9.8 * land['z'].values)
    land['depth'] = (land['z'].dims, depth)
    z_attrs = land['z'].attrs
    land['z'] = (land['z'].dims, land['z'].values)
    land['z'].attrs = z_attrs
    return land


def regrid_continent_data(land, basefile='', sea_level=0, max_depth=1000):

    base = xr.open_mfdataset(basefile)

    if all(-np.pi < l < np.pi for l in base['lat'].values):
        lats = 180.0 / np.pi * base['lat'].values
    else:
        lats = base['lat'].values
    if all(-2 * np.pi < l < 2 * np.pi for l in base['lon'].values):
        lons = 180.0 / np.pi * base['lon'].values
    else:
        lons = base['lon'].values

    if len(land['lat'].shape) == 2:
        raw_lats = land['lat'][:, 0].values
        raw_lons = land['lon'][0, :].values
    else:
        raw_lats = land['lat'].values
        raw_lons = land['lon'].values

    raw_landmask = np.where(land['z'].values > sea_level, 1.0, 0.0)

    ds_out = xr.Dataset({'lat': (['lat'], lats),
                         'lon': (['lon'], lons)})

    logger.info('Regridding continent data')
    regridder = xe.Regridder(land, ds_out, 'bilinear')
    ds_out = regridder(land)

    logger.info('Calculating landfrac_dict')
    landfrac_dict = overlap_fraction(raw_lats, raw_lons, lats, lons,
                                     raw_landmask)
    landfrac = get_landfrac(shape=ds_out['z'].shape,
                            landfrac_dict=landfrac_dict)
    landmask = np.array(landfrac > 0, dtype=np.int32)

    landmask[landmask > 1] = 1
    landfrac[landfrac > 1] = 1
    landmask[landmask < 0] = 0
    landfrac[landfrac < 0] = 0

    ds_out['landfrac'] = (ds_out['z'].dims, landfrac)
    ds_out['landmask'] = (ds_out['z'].dims, landmask.astype(np.int32))
    ds_out['land_mask'] = (ds_out['z'].dims, landmask.astype(np.int32))

    ds_out['oceanfrac'] = (ds_out['z'].dims, 1 - ds_out['landfrac'].values)
    ds_out['oceanmask'] = (ds_out['z'].dims,
                           np.array(1 - ds_out['landmask'].values,
                                    dtype=np.int32))

    height = np.where(ds_out['z'].values > sea_level, ds_out['z'].values, 0)
    depth = np.where(ds_out['z'] <= sea_level, -ds_out['z'].values, 0)
    depth = np.where(depth > max_depth, max_depth, depth)
    ds_out['z'] = (ds_out['z'].dims, height)
    ds_out['zsurf'] = (ds_out['z'].dims, height)
    ds_out['depth'] = (ds_out['z'].dims, depth)
    ds_out['PHIS'] = (ds_out['z'].dims, 9.8 * ds_out['z'].values)
    return ds_out


def zonal_band_anomaly_squared_distance(y, y0):
    return (y - y0)**2

def meridional_band_anomaly_squared_distance(r, x, x0, y):
    ymin = np.sqrt(r) + 1
    ymax = 90 - ymin - 1
    if ymin < y < ymax:
        return (x - x0)**2
    elif y <= ymin:
        return (x - x0)**2 + (y - ymin)**2
    elif y >= ymax:
        return (x - x0)**2 + (y - ymax)**2


def disk_anomaly_squared_distance(x, x0, y, y0):
    return (x - x0)**2 + (y - y0)**2


def anomaly_smoothing(max_val, d, r):
    if d <= r:
        return max_val * (r - d) / r
    return 0


def fill_poles(data):
    tmp = data['land_mask'].values
    for i in range(len(data['lat'].values)):
        for j in range(len(data['lon'].values)):
            if np.abs(data['lat'].values[i] - 90.0) < 1:
                tmp[i, j] = 1.0
    return tmp


def modify_array_with_time(data, old_mask, new_mask):
    tmp = np.zeros(data.shape)
    for i in range(tmp.shape[0]):
        data[i, :, :] = modify_arrays_with_mask(data[i, :, :], old_mask,
                                                new_mask)
    return data


def modify_array_with_time_and_level(data, old_mask, new_mask):
    tmp = np.zeros(data.shape)
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            data[i, j, :, :] = modify_arrays_with_mask(data[i, j, :, :],
                                                       old_mask, new_mask)
    return data


def modify_arrays_with_mask(data, old_mask, new_mask):
    tmp = np.zeros(new_mask.shape)
    mask_mean = data[old_mask > 0].mean()
    nomask_mean = data[old_mask==0].mean()

    tmp[new_mask > 0] = mask_mean
    tmp[new_mask == 0] = nomask_mean
    return tmp


def modify_all_arrays_with_mask(data, old_mask, new_mask):
    for e in tqdm(data):
        if data[e].shape == 3:
            tmp_attrs = data[e].attrs
            tmp_vals = modify_array_with_time(data[e].values, old_mask,
                                              new_mask)
            data[e] = (data[e].dims, tmp_vals)
            data[e].attrs = tmp_attrs
        if data[e].shape == 4:
            tmp_attrs = data[e].attrs
            tmp_vals = modify_array_with_time_and_level(data[e].values,
                                                        old_mask, new_mask)
            data[e] = (data[e].dims, tmp_vals)
            data[e].attrs = tmp_attrs
    return data


def cell_overlap(lats, lons, lat, lon, dx, dy, landmask):

    lons_in = ((lon - dx / 2 < lons) & (lons < lon + dx / 2))
    lats_in = ((lat - dy / 2 < lats) & (lats < lat + dy / 2))

    lat_count = lats_in.sum()
    lon_count = lons_in.sum()
    total_count = lat_count*lon_count
    mask_count = landmask[np.outer(lats_in, lons_in)].sum()

    return {'mask_count': mask_count, 'total_count': total_count}


def overlap_fraction(inlat, inlon, outlat, outlon, landmask):
    """Compute overlap fraction between sets of coordinates"""
    tmp = {}
    inlon = [x + 360.0 if x < 0 else x for x in inlon]
    outlon = [x + 360.0 if x < 0 else x for x in outlon]

    for i, lat in tqdm(enumerate(inlat)):
        for j, lon in enumerate(inlon):
            lat_idx = np.argmin(np.abs(outlat - lat))
            lon_idx = np.argmin(np.abs(outlon - lon))

            key = (lat_idx, lon_idx)
            if key not in tmp:
                tmp[key] = {'mask_count': landmask[i, j], 'total_count': 1.0}
            else:
                tmp[key]['mask_count'] += landmask[i, j]
                tmp[key]['total_count'] += 1.0
    return tmp


def get_landfrac(shape=None, landfrac_dict=None):

    landfrac = np.zeros(shape)
    logger.info('Calculating landfrac')
    for i in tqdm(range(landfrac.shape[0])):
        for j in range(landfrac.shape[1]):
            key = (i, j)
            landfrac[i, j] = landfrac_dict[key]['mask_count']
            landfrac[i, j] /= landfrac_dict[key]['total_count']
    return landfrac

def get_oceanfrac(shape=None, landfrac_dict=None):

    oceanfrac = np.zeros(shape)
    logger.info('Calculating oceanfrac')
    for i in tqdm(range(oceanfrac.shape[0])):
        for j in range(oceanfrac.shape[1]):
            key = (i, j)
            oceanfrac[i, j] = 1 - landfrac_dict[key]['mask_count']
            oceanfrac[i, j] /= landfrac_dict[key]['total_count']
    return oceanfrac

def anomaly_value(max_val, r, x, x0, y, y0, anomaly_type='disk'):
    if x >= 180.0:
        x -= 360.0
    if x0 >= 180.0:
        x0 -= 360.0
    if anomaly_type == 'disk':
        d = disk_anomaly_squared_distance(x, x0, y, y0)
    if anomaly_type == 'zonal_band':
        d = zonal_band_anomaly_squared_distance(y, y0)
    if anomaly_type == 'meridional_band':
        d = meridional_band_anomaly_squared_distance(r, x, x0, y)
    if anomaly_type == 'none':
        return 0
    return anomaly_smoothing(max_val, np.sqrt(d), np.sqrt(r))

def inject_anomaly(basefile='', anomaly_type='disk', variable='PHIS',
                   exp_type='dry_hs', max_anomaly=0, squared_radius=1,
                   anomaly_lon=0, anomaly_lat=0, use_lapse_rate=True,
                   just_surface=False):

    base = xr.open_mfdataset(basefile)
    data = base[variable].values
    lats = base['lat'].values
    lons = base['lon'].values

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = anomaly_value(max_anomaly, squared_radius, lon,
                                  anomaly_lon, lat, anomaly_lat,
                                  anomaly_type=anomaly_type)
            if exp_type == 'aqua':
                data[:, i, j] += value
            elif exp_type == 'dry_hs':
                if use_lapse_rate:
                    for k in range(len(data[0, :, 0, 0])):
                        lapse_rate = value / (len(data[0, :, 0, 0]) - 1)
                        data[:, k, i, j] += k * lapse_rate
                elif just_surface:
                    data[:, -1, i, j] += value
                else:
                    data[:, :, i, j] += value
    return data
