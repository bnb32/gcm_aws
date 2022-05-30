"""Get animation figures from simulation output"""
import numpy as np
import os
from joblib import Parallel, delayed

from ecrlgcm.environment import EnvironmentConfig
from ecrlgcm.postprocessing import (PostProcessing, figures_argparse)


def parallel_run(i, args, config):
    """Run interactive globe creation in parallel"""
    fig_name = f'{args.field}_{i:04}.png'
    if not os.path.exists(f'{config.USER_ANIMS_DIR}/{fig_name}'):

        post_proc = PostProcessing(config)

        if args.time_avg:
            post_proc.get_interactive_globe(
                times[i], field=args.field, level=args.level, save_fig=True,
                fig_name=fig_name, gcm_type=args.mode, vmin=vmin, vmax=vmax)
        else:
            post_proc.get_interactive_globe_time(
                args.year, time_step=times[i], field=args.field,
                level=args.level, save_fig=True, fig_name=fig_name,
                gcm_type=args.model, vmin=vmin, vmax=vmax)


if __name__ == '__main__':
    parser = figures_argparse()
    args = parser.parse_args()

    config = EnvironmentConfig(args.config)

    if args.overwrite:
        cmd = f'rm -f {config.USER_ANIMS_DIR}/{args.field}_*.png;'
        os.system(cmd)

    if args.field == 'RELHUM' or 'CLOUD' in args.field:
        vmin = 0
        vmax = 100
    elif args.field == 'TS':
        vmin = 240
        vmax = 300
    else:
        vmin = vmax = None

    if args.time_avg:
        times = np.linspace(500, 0, 501)
    else:
        times = list(range(100))

    Parallel(n_jobs=24)(
        delayed(parallel_run)(i, args, config) for i in range(len(times)))

    if args.time_avg:
        animation_name = f'{args.model}_{args.field}.mp4'
    else:
        animation_name = f'{args.model}_{args.field}_{args.year}.mp4'

    cmd = 'ffmpeg -r 5 -f image2 -s 1920x1080 -i '
    cmd += f'{config.USER_ANIMS_DIR}/{args.field}_%04d.png -vcodec '
    cmd += 'libx264 -crf 25  -pix_fmt yuv420p '
    cmd += f'{config.USER_FIGS_DIR}/{animation_name}'
    os.system(cmd)
