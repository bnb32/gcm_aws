"""Initialize ISCA"""
import os, sys
import argparse

from ecrlgcm.environment import EnvironmentConfig


def init_isca_argparse():
    """Parse args for ISCA initialization"""
    parser = argparse.ArgumentParser(description="Initialize ISCA")
    parser.add_argument('-config', required=True)
    return parser


if __name__ == '__main__':
    parser = init_isca_argparse()
    args = parser.parse_args()

    config = EnvironmentConfig(args.config)
    cmd = f'git clone https://github.com/ExeClim/Isca {config.GFDL_BASE}'
    cmd += f'; cd {config.GFDL_BASE}'
    cmd += '; conda env create -f ci/environment-py3.9.yml'
    cmd += '; conda activate isca_env'

    os.system(cmd)

    cmd = f'cd {config.GFDL_BASE}'
    cmd += '; cd src/extra/python/'
    cmd += '; pip install -e .'

    os.system(cmd)
