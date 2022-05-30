"""Variable CO2 and continents ISCA simulation"""

from ecrlgcm import run_isca_variable_co2_continents_argparse
from ecrlgcm.environment import EnvironmentConfig
from ecrlgcm.preprocessing import (PreProcessing, solar_constant,
                                   eccentricity, obliquity)
from ecrlgcm.experiment import Experiment as ecrlExp
from ecrlgcm.utilities.utilities import get_logger

import os
import f90nml
from isca import IscaCodeBase, DiagTable, Experiment, GFDL_BASE


if __name__ == '__main__':
    parser = run_isca_variable_co2_continents_argparse()
    args = parser.parse_args()

    config = EnvironmentConfig(args.config)
    pre_proc = PreProcessing(config)

    logger = get_logger()
    NCORES = int(args.ncores)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    cb = IscaCodeBase.from_directory(GFDL_BASE)
    cb.compile()
    ecrlexp = ecrlExp(config, gcm_type='isca',
                      multiplier=args.multiplier,
                      land_year=args.land_year,
                      max_depth=0,
                      co2_value=args.co2)

    if args.overwrite:
        logger.info(f'Removing {ecrlexp.file_path}')
        os.system(f'rm -rf {ecrlexp.file_path}')

    exp = Experiment(ecrlexp.name, codebase=cb)

    co2_file = ecrlexp.co2_file.rstrip('.nc')
    land_file = ecrlexp.land_file

    pre_proc.modify_isca_input_files(ecrlexp, remap=args.remap)

    exp.inputfiles = [
        ecrlexp.co2_file, ecrlexp.topo_file,
        os.path.join(
            os.environ['GFDL_BASE'],
            'exp/test_cases/realistic_continents/input/siconc_clim_amip.nc')]

    diag = DiagTable()
    diag.add_file('atmos_monthly', 30, 'days', time_units='days')

    diag.add_field('dynamics', 'ps', time_avg=True)
    diag.add_field('dynamics', 'bk')
    diag.add_field('dynamics', 'pk')
    diag.add_field('atmosphere', 'bucket_depth', time_avg=True)
    diag.add_field('dynamics', 'zsurf')
    diag.add_field('atmosphere', 'precipitation', time_avg=True)
    diag.add_field('atmosphere', 'cape', time_avg=True)
    diag.add_field('atmosphere', 'convection_rain', time_avg=True)
    diag.add_field('atmosphere', 'condensation_rain', time_avg=True)
    diag.add_field('mixed_layer', 't_surf', time_avg=True)
    diag.add_field('dynamics', 'sphum', time_avg=True)
    diag.add_field('atmosphere', 'rh', time_avg=True)
    diag.add_field('dynamics', 'ucomp', time_avg=True)
    diag.add_field('dynamics', 'vcomp', time_avg=True)
    diag.add_field('dynamics', 'temp', time_avg=True)
    diag.add_field('dynamics', 'vor', time_avg=True)
    diag.add_field('dynamics', 'div', time_avg=True)
    diag.add_field('two_stream', 'co2', time_avg=True)

    exp.diag_table = diag
    exp.clear_rundir()

    namelist_name = os.path.join(base_dir, 'variable_co2_and_continents.nml')
    nml = f90nml.read(namelist_name)
    exp.namelist = nml

    exp.update_namelist({
        'spectral_init_cond_nml': {
            'topog_file_name': f'{ecrlexp.topo_file.split("/")[-1]}',
            'topography_option': 'input'
        },

        'two_stream_gray_rad_nml': {
            'rad_scheme': 'byrne',
            'atm_abs': 0.2,
            'do_seasonal': True,
            'equinox_day': 0.75,
            'do_read_co2': True,
            'co2_file': f'{ecrlexp.co2_file.rstrip(".nc").split("/")[-1]}',
            'solar_constant': solar_constant(args.land_year),
        },

        'idealized_moist_phys_nml': {
            'do_damping': True,
            'turb': True,
            'mixed_layer_bc': True,
            'do_virtual': False,
            'do_simple': True,
            'roughness_mom': 3.21e-05,
            'roughness_heat': 3.21e-05,
            'roughness_moist': 3.21e-05,
            'two_stream_gray': True,
            'convection_scheme': 'SIMPLE_BETTS_MILLER',
            'land_option': 'input',
            'land_file_name': f'INPUT/{ecrlexp.topo_file.split("/")[-1]}',
        },

        'astronomy_nml': {
            'obliq': obliquity(args.land_year),
            'ecc': eccentricity(args.land_year),
        },

    })

    exp.run(1, use_restart=False, num_cores=NCORES,
            overwrite_data=args.overwrite)
    for i in range(2, int(args.nyears) + 1):
        exp.run(i, num_cores=NCORES, overwrite_data=args.overwrite)
