"""Get animation from simulation output"""
import numpy as np
from ecrlgcm.environment import EnvironmentConfig

from ecrlgcm.postprocessing import (PostProcessing, animation_argparse)


if __name__ == '__main__':
    parser = animation_argparse()
    args = parser.parse_args()

    config = EnvironmentConfig(args.config)
    post_proc = PostProcessing(config)

    times = np.linspace(500, 0, 2)

    if args.time_avg:
        if args.field == 'z':
            post_proc.get_continent_animation(times, post_proc.land_years)
        else:
            if args.field == 'RELHUM' or 'CLOUD' in args.field:
                post_proc.get_field_animation(
                    times, post_proc.stored_years, level=args.level,
                    field='RELHUM', color_map='custom_clouds',
                    gcm_type=args.model, globe=args.globe)
            else:
                post_proc.get_field_animation(
                    times, post_proc.stored_years, level=args.level,
                    field=args.field, color_map='bwr', gcm_type=args.model,
                    globe=args.globe)

    else:
        if args.field == 'RELHUM' or 'CLOUD' in args.field:
            post_proc.get_field_time_animation(
                args.year, post_proc.stored_years, level=args.level,
                field=args.field, color_map='custom_clouds', level_num=5,
                plevel=args.plevel, gcm_type=args.model, globe=args.globe)
        else:
            post_proc.get_field_time_animation(
                args.year, post_proc.stored_years, level=args.level,
                field=args.field, color_map='bwr', plevel=args.plevel,
                gcm_type=args.model, globe=args.globe)
