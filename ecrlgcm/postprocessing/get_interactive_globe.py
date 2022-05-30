"""Run interactive globe creation"""
from ecrlgcm.environment import EnvironmentConfig
from ecrlgcm.postprocessing import PostProcessing
from ecrlgcm import globe_argparse


if __name__ == '__main__':
    parser = globe_argparse()
    args = parser.parse_args()

    config = EnvironmentConfig(args.config)

    post_proc = PostProcessing(config)
    post_proc.get_interactive_globe(land_year=args.year, field=args.field,
                                    level=args.level, save_html=args.save_html,
                                    gcm_type=args.model, vmin=0, vmax=100)
