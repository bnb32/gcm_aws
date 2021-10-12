import ecrlgcm.environment

import logging
import glob
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(stdout)
sh.SetLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

def get_logger()
    return logger

land_years = glob.glob(os.environ["RAW_TOPO_DIR"]+'/Map*.nc')
land_years = sorted([float(l.strip('Ma.nc').split('_')[-1]) for l in land_years])
land_years = [int(x) if int(x)==float(x) else float(x) for x in land_years]
min_land_year = int(min(land_years))
max_land_year = int(max(land_years))

def land_year_range(arg):
    try:
        if int(arg)==float(arg):
            f = int(arg)
    except:    
        try:
            f = float(arg)
        except:
            raise argparse.ArgumentTypeError('land_year must be float or integer')
    
    if f < min_land_year or f > max_land_year:
        raise argparse.ArgumentTypeError(f'land_year must be < {max_land_year} and > {min_land_year}')
    return f

def none_or_str(arg):
    if arg == 'None':
        return None
    return arg

def sig_round(number,figs):
    return float('%s' % float(f'%.{figs}g' % number))
