from ecrlgcm.postprocessing import get_interactive_globe
from ecrlgcm.utilities import land_year_range, min_land_year, max_land_year, none_or_int

import argparse

def globe_argparse():
    parser=argparse.ArgumentParser(description="Make interactive globe")
    parser.add_argument('-field',default='RELHUM')
    parser.add_argument('-level',default=None,type=none_or_int)
    parser.add_argument('-save_html',default=False,action='store_true')
    parser.add_argument('-model',default='cesm',choices=['cesm','isca'])
    parser.add_argument('-year',default=0,type=land_year_range,
                        metavar=f'[{min_land_year}-{max_land_year}]',
                        help="Years prior to current era in units of Ma")
    return parser

parser = globe_argparse()
args=parser.parse_args()

get_interactive_globe(land_year=args.year,
                      field=args.field,
                      level=args.level,
                      save_html=args.save_html,
                      gcm_type=args.model,
                      vmin=0,vmax=100)
