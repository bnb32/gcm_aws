"""Ecrlgcm"""
import os
import argparse

from ecrlgcm.utilities import (none_or_str)

ECRLGCM_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.dirname(ECRLGCM_DIR), 'ecrlgcm', 'data')

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
    parser.add_argument('-year', default=0,
                        help="Years prior to current era in units of Ma")
    parser.add_argument('-sea_level', default=0, type=float)
    parser.add_argument('-ncores', default=32, type=int)
    parser.add_argument('-nyears', default=10, type=int)
    parser.add_argument('-overwrite', action='store_true')
    parser.add_argument('-remap', action='store_true')
    parser.add_argument('-config', required=True)
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
                        help="Max ocean depth")
    parser.add_argument('-year', default=0,
                        help="Years prior to current era in units of Ma")
    parser.add_argument('-case', default='test', type=str)
    parser.add_argument('-ntasks', default=96, type=int)
    parser.add_argument('-nthrds', default=8, type=int)
    parser.add_argument('-start_date', default="0001-01-01")
    parser.add_argument('-step_type', default='ndays')
    parser.add_argument('-nsteps', default=300, type=int)
    parser.add_argument('-restart', default=False, action='store_true')
    parser.add_argument('-setup', default=False, action='store_true')
    parser.add_argument('-build', default=False, action='store_true')
    parser.add_argument('-run', default=False, action='store_true')
    parser.add_argument('-run_all', default=False, action='store_true')
    parser.add_argument('-remap', default=False, action='store_true')
    parser.add_argument('-remap_hires', default=False, action='store_true')
    parser.add_argument('-timing', default=False, action='store_true')
    parser.add_argument('-config', required=True)
    return parser


def run_isca_variable_co2_continents_argparse():
    """Parse args for variable co2 ISCA simulation"""
    parser = argparse.ArgumentParser(description="Run variable co2 experiment")
    parser.add_argument('-multiplier', default=1)
    parser.add_argument('-co2', default=None, type=none_or_str)
    parser.add_argument('-land_year', default=0,
                        help="Years prior to current era in units of Ma")
    parser.add_argument('-sea_level', default=0, type=float)
    parser.add_argument('-nyears', default=5, type=int)
    parser.add_argument('-ncores', default=32, type=int)
    parser.add_argument('-overwrite', action='store_true')
    parser.add_argument('-remap', action='store_true')
    parser.add_argument('-config', required=True)
    return parser
