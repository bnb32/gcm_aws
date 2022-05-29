from ecrlgcm.postprocessing import get_continent_animation, get_field_animation, get_field_time_animation
from ecrlgcm.utilities import land_years, stored_years, none_or_int, none_or_float

import numpy as np
import argparse

def animation_argparse():
    parser=argparse.ArgumentParser(description="Make animation")
    parser.add_argument('-field')
    parser.add_argument('-level',default=None,type=none_or_int)
    parser.add_argument('-plevel',default=None,type=none_or_float)
    parser.add_argument('-model',default='cesm',choices=['cesm','isca'])
    parser.add_argument('-globe',default=False,action='store_true')
    parser.add_argument('-time_avg',default=False,action='store_true')
    parser.add_argument('-year',default=751,type=float)
    return parser

if __name__ == '__main__':
    parser = animation_argparse()
    args=parser.parse_args()

    times = np.linspace(500,0,2)

    if args.time_avg:
        if args.field=='z':
            get_continent_animation(times,land_years)
        else:
            if args.field=='RELHUM' or 'CLOUD' in args.field:
                get_field_animation(times,stored_years,level=args.level,
                                field='RELHUM',color_map='custom_clouds',
                                gcm_type=args.model,globe=args.globe)
            else:
                get_field_animation(times,stored_years,level=args.level,
                                field=args.field,color_map='bwr',
                                gcm_type=args.model,globe=args.globe)

    if not args.time_avg:

        if args.field=='RELHUM' or 'CLOUD' in args.field:
            get_field_time_animation(args.year,stored_years,level=args.level,
                                 field=args.field,color_map='custom_clouds',
                                 level_num=5,plevel=args.plevel,
                                 gcm_type=args.model,
                                 globe=args.globe)
        else:
            get_field_time_animation(args.year,stored_years,level=args.level,
                                 field=args.field,color_map='bwr',
                                 plevel=args.plevel,gcm_type=args.model,
                                 globe=args.globe)
