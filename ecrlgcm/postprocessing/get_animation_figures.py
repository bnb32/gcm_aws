from ecrlgcm.postprocessing import get_interactive_globe
from ecrlgcm.misc import land_year_range, min_land_year, max_land_year, none_or_int
import ecrlgcm.environment

import argparse
import numpy as np
import os
from joblib import Parallel, delayed


parser=argparse.ArgumentParser(description="Make interactive globe")
parser.add_argument('-field',default='RELHUM')
parser.add_argument('-level',default=None,type=none_or_int)
parser.add_argument('-model',default='cesm',choices=['cesm','isca'])
parser.add_argument('-overwrite',default=False,action='store_true')
args=parser.parse_args()

if args.overwrite:
    cmd = f'rm -f {os.environ["USER_ANIMS_DIR"]}/{args.field}_*.png;'
    os.system(cmd)

if args.field == 'RELHUM':
    vmin = 0
    vmax = 100
if args.field == 'TS':
    vmin = 240
    vmax = 300

times = np.linspace(500,0,501)

def tmp(i):
    fig_name = f'{args.field}_{i:04}.png'
    if not os.path.exists(f'{os.environ["USER_ANIMS_DIR"]}/{fig_name}'):
        get_interactive_globe(times[i],field=args.field,
                              level=args.level,save_fig=True,
                              fig_name=fig_name,
                              gcm_type=args.model,
                              vmin=vmin,
                              vmax=vmax)

Parallel(n_jobs=24)(delayed(tmp)(i) for i in range(len(times)))

'''
Parallel(n_jobs=24)(delayed(get_interactive_globe)(times[i],
                                                   field=args.field,
                                                   level=args.level,
                                                   save_fig=True,
                                                   fig_name=f'{args.field}_{i:04}.png',
                                                   vmin=vmin,vmax=vmax) for i in range(len(times)))
'''
cmd = f'ffmpeg -r 20 -f image2 -s 1920x1080 -i {os.environ["USER_ANIMS_DIR"]}/{args.field}_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {os.environ["USER_FIGS_DIR"]}/{args.model}_{args.field}.mp4'   
os.system(cmd)
