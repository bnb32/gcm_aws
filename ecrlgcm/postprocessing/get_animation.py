from ecrlgcm.postprocessing import get_continent_animation, get_field_animation
from ecrlgcm.misc import land_years, stored_years, none_or_int

import numpy as np
import argparse

parser=argparse.ArgumentParser(description="Make animation")
parser.add_argument('-field')
parser.add_argument('-level',default=None,type=none_or_int)
parser.add_argument('-model',default='cesm',choices=['cesm','isca'])
parser.add_argument('-globe',default=False,action='store_true')
args=parser.parse_args()

times = np.linspace(500,0,2)

if args.field=='z':
    get_continent_animation(times,land_years)
else:
    if args.field=='RELHUM':
        get_field_animation(times,stored_years,level=args.level,
                            field='RELHUM',color_map='custom_clouds',
                            gcm_type=args.model,globe=args.globe)
    else:
        get_field_animation(times,stored_years,level=args.level,
                            field=args.field,color_map='bwr',
                            gcm_type=args.model,globe=args.globe)
