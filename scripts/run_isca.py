"""Run ISCA simulation"""
import os

from ecrlgcm import isca_argparse


if __name__ == '__main__':
    parser = isca_argparse()
    args = parser.parse_args()

    cmd = 'python experiments/variable_co2_and_continents.py'
    cmd += f' -config {args.config}'
    cmd += f' -multiplier {args.multiplier}'
    cmd += f' -land_year {args.year}'
    cmd += f' -sea_level {args.sea_level}'
    cmd += f' -nyears {args.nyears}'
    cmd += f' -ncores {args.ncores}'
    cmd += f' -co2 {args.co2}'
    cmd += f' -config {args.config}'
    if args.overwrite:
        cmd += ' -overwrite'
    if args.remap:
        cmd += ' -remap'

    os.system(cmd)
