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
args=parser.parse_args()

cmd = f'rm -f {os.environ["USER_ANIMS_DIR"]}/{args.field}_*.png;'
os.system(cmd)

times = np.linspace(500,0,100)
Parallel(n_jobs=32)(delayed(get_interactive_globe)(times[i],
                                                   field=args.field,
                                                   level=args.level,
                                                   save_fig=True,
                                                   fig_name=f'{args.field}_{i:04}.png',
                                                   vmin=0,vmax=100) for i in range(len(times)))

'''        
for i,t in enumerate(times):
    get_interactive_globe(land_year=t,
                          field=args.field,
                          level=args.level,
                          gcm_type=args.model,
                          save_fig=True,
                          fig_name=f'{args.field}_{i:04}.png',
                          vmin=0,vmax=100)

cmd = f'ffmpeg -r 5 -f image2 -s 1920x1080 -i {os.environ["USER_ANIMS_DIR"]}/{args.field}_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {os.environ["USER_FIGS_DIR"]}/{args.field}.mp4'   
os.system(cmd)
'''
