from ecrlgcm.postprocessing import get_interactive_globe, get_interactive_globe_time
from ecrlgcm.utilities import land_year_range, min_land_year, max_land_year, none_or_int
import ecrlgcm.environment

import argparse
import numpy as np
import os
from joblib import Parallel, delayed


parser=argparse.ArgumentParser(description="Make interactive globe")
parser.add_argument('-field',default='RELHUM')
parser.add_argument('-year',default=751,type=float)
parser.add_argument('-level',default=None,type=none_or_int)
parser.add_argument('-model',default='cesm',choices=['cesm','isca'])
parser.add_argument('-overwrite',default=False,action='store_true')
parser.add_argument('-time_avg',default=False,action='store_true')
args=parser.parse_args()

if args.overwrite:
    cmd = f'rm -f {os.environ["USER_ANIMS_DIR"]}/{args.field}_*.png;'
    os.system(cmd)

if args.field == 'RELHUM' or 'CLOUD' in args.field:
    vmin = 0
    vmax = 100
elif args.field == 'TS':
    vmin = 240
    vmax = 300
else:
    vmin=vmax=None

if args.time_avg:
    times = np.linspace(500,0,501)
else:
    times = [i for i in range(100)]

def tmp(i):
    fig_name = f'{args.field}_{i:04}.png'
    if not os.path.exists(f'{os.environ["USER_ANIMS_DIR"]}/{fig_name}'):

        if args.time_avg:
            get_interactive_globe(times[i],field=args.field,
                              level=args.level,save_fig=True,
                              fig_name=fig_name,
                              gcm_type=args.model,
                              vmin=vmin,
                              vmax=vmax)
        else:
            get_interactive_globe_time(args.year,time_step=times[i],field=args.field,
                              level=args.level,save_fig=True,
                              fig_name=fig_name,
                              gcm_type=args.model,
                              vmin=vmin,
                              vmax=vmax)

Parallel(n_jobs=24)(delayed(tmp)(i) for i in range(len(times)))

if args.time_avg:
    animation_name=f'{args.model}_{args.field}.mp4'
else:
    animation_name=f'{args.model}_{args.field}_{args.year}.mp4'


cmd = f'ffmpeg -r 5 -f image2 -s 1920x1080 -i {os.environ["USER_ANIMS_DIR"]}/{args.field}_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {os.environ["USER_FIGS_DIR"]}/{animation_name}'
os.system(cmd)
