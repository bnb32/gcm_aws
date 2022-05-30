"""Get maps for simluations"""
import os
import warnings
import argparse

from ecrlgcm.environment import EnvironmentConfig
warnings.filterwarnings("ignore")


def get_maps_argparse():
    """Parse args for getting maps"""
    parser = argparse.ArgumentParser(
        description="Download paleo-continent maps")
    parser.add_argument('-config', required=True)
    return parser


if __name__ == '__main__':
    parser = get_maps_argparse()
    args = parser.parse_args()
    config = EnvironmentConfig(args.config)

    zip_file = 'Scotese_Wright_2018_Maps_1-88_6minX6min_PaleoDEMS_nc.zip'
    data_source = f'https://zenodo.org/record/5460860/files/{zip_file}'
    cmd = f'rm -rf {config.RAW_TOPO_DIR}'
    cmd += f'; wget {data_source}'
    cmd += f'; unzip {zip_file}'
    cmd += f'; mv {zip_file.strip(".zip")} {config.RAW_TOPO_DIR}'
    cmd += f'; rm {zip_file}'

    os.system(cmd)
