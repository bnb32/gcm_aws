"""Ecrlgcm"""
import os
import argparse

from ecrlgcm.utilities.utilities import (none_or_str, none_or_int,
                                         none_or_float)

ECRLGCM_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.dirname(ECRLGCM_DIR), 'ecrlgcm', 'data')

config_arg_description = ('Path to configuration file. Needs to have the '
                          'parameters defined in environment/config.py. Needs '
                          'to be in either json format or a .py file.')
year_arg_description = ("Years prior to current era in units of Ma.")

EXPERIMENT_DICTIONARY = {
    'cam_clmCN_docnSOM':
        {'compset': 'F1850_CAM60_CLM50%CN_SICE_DOCN%SOM_SROF_SGLC_SWAV',
         'res': 'f19_f19_mg17',
         'custom': True},
    'cam_clmCN_docnDOM':
        {'compset': 'F1850_CAM60_CLM50%CN_SICE_DOCN%DOM_SROF_SGLC_SWAV',
         'res': 'f19_f19_mg17',
         'custom': True},
    'aqua':
        {'compset': 'QSC6',
         'res': 'f19_f19_mg17',
         'custom': True}}


def isca_argparse():
    """Parse args for ISCA run"""
    parser = argparse.ArgumentParser(description="Run ISCA")
    parser.add_argument('-multiplier', default=1, help="CO2 Multiplier")
    parser.add_argument('-co2', default=None, help="CO2 Value")
    parser.add_argument('-year', default=0, help=year_arg_description)
    parser.add_argument('-sea_level', default=0, type=float,
                        help='Sea level to use for simulation. Relative to '
                             'current era. So to lower the level use a '
                             'negative number.')
    parser.add_argument('-ncores', default=32, type=int,
                        help='Number of cores to use for simulation.')
    parser.add_argument('-nyears', default=10, type=int,
                        help='Number of years to run simulation.')
    parser.add_argument('-overwrite', action='store_true',
                        help='Whether to overwrite existing data.')
    parser.add_argument('-remap', action='store_true')
    parser.add_argument('-config', required=True, help=config_arg_description)
    return parser


def cesm_argparse():
    """Parse args for CESM run"""
    parser = argparse.ArgumentParser(description="Run CESM")
    parser.add_argument('-exp', default='cam_clmCN_docnDOM',
                        choices=list(EXPERIMENT_DICTIONARY))
    parser.add_argument('-multiplier', default=1.0, type=float,
                        help="CO2 Multiplier")
    parser.add_argument('-co2_value', default=None, help="CO2 Value")
    parser.add_argument('-sea_level', default=0, type=float, help="Sea level")
    parser.add_argument('-max_depth', default=1000, type=float,
                        help="Max ocean depth. This is the number of meters "
                             "below sea level at which the ocean depth will "
                             "be capped for the simulation.")
    parser.add_argument('-year', default=0, help=year_arg_description)
    parser.add_argument('-case', default='test', type=str)
    parser.add_argument('-ntasks', default=96, type=int)
    parser.add_argument('-nthrds', default=8, type=int)
    parser.add_argument('-start_date', default="0001-01-01")
    parser.add_argument('-step_type', default='ndays')
    parser.add_argument('-nsteps', default=300, type=int,
                        help='Number of steps to run simulation. If step_type '
                             'is days then this will be the number of days to '
                             'run')
    parser.add_argument('-restart', default=False, action='store_true')
    parser.add_argument('-setup', default=False, action='store_true')
    parser.add_argument('-build', default=False, action='store_true')
    parser.add_argument('-run', default=False, action='store_true')
    parser.add_argument('-run_all', default=False, action='store_true')
    parser.add_argument('-remap', default=False, action='store_true')
    parser.add_argument('-remap_hires', default=False, action='store_true')
    parser.add_argument('-timing', default=False, action='store_true')
    parser.add_argument('-config', required=True, help=config_arg_description)
    return parser


def run_isca_variable_co2_continents_argparse():
    """Parse args for variable co2 ISCA simulation"""
    parser = argparse.ArgumentParser(description="Run variable co2 experiment")
    parser.add_argument('-multiplier', default=1)
    parser.add_argument('-co2', default=None, type=none_or_str)
    parser.add_argument('-land_year', default=0, help=year_arg_description)
    parser.add_argument('-sea_level', default=0, type=float)
    parser.add_argument('-nyears', default=5, type=int)
    parser.add_argument('-ncores', default=32, type=int)
    parser.add_argument('-overwrite', action='store_true')
    parser.add_argument('-remap', action='store_true')
    parser.add_argument('-config', required=True, help=config_arg_description)
    return parser


def app_argparse():
    """Parse args for dahsboard app"""
    parser = argparse.ArgumentParser(description="Initialize dashboard app")
    parser.add_argument('-config', required=True, help=config_arg_description)
    return parser


def animation_argparse():
    """Parse args for creating animation"""
    parser = argparse.ArgumentParser(description="Make animation")
    parser.add_argument('-field')
    parser.add_argument('-level', default=None, type=none_or_int)
    parser.add_argument('-plevel', default=None, type=none_or_float)
    parser.add_argument('-model', default='cesm', choices=['cesm', 'isca'])
    parser.add_argument('-globe', default=False, action='store_true')
    parser.add_argument('-time_avg', default=False, action='store_true')
    parser.add_argument('-year', default=751, type=float,
                        help=year_arg_description)
    parser.add_argument('-config', required=True, help=config_arg_description)
    return parser


def figures_argparse():
    """Parse args for creating animation figures"""
    parser = argparse.ArgumentParser(description="Make interactive globe")
    parser.add_argument('-field', default='RELHUM')
    parser.add_argument('-year', default=751, type=float,
                        help=year_arg_description)
    parser.add_argument('-level', default=None, type=none_or_int)
    parser.add_argument('-model', default='cesm', choices=['cesm', 'isca'])
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('-time_avg', default=False, action='store_true')
    parser.add_argument('-config', required=True, help=config_arg_description)
    return parser


def globe_argparse():
    """Parse args for creating interactive globe"""
    parser = argparse.ArgumentParser(description="Make interactive globe")
    parser.add_argument('-field', default='RELHUM')
    parser.add_argument('-level', default=None, type=none_or_int)
    parser.add_argument('-save_html', default=False, action='store_true')
    parser.add_argument('-model', default='cesm', choices=['cesm', 'isca'])
    parser.add_argument('-year', default=0, help=year_arg_description)
    parser.add_argument('-config', required=True, help=config_arg_description)
    return parser
