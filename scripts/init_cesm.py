"""Initialize CESM"""
import os
import argparse

from ecrlgcm.environment import EnvironmentConfig


def init_cesm_argparse():
    """Parse args for CESM initialization"""
    parser = argparse.ArgumentParser(description="Initialize CESM")
    parser.add_argument('-config_only', default=False, action='store_true')
    parser.add_argument('-config', required=True)
    return parser


if __name__ == '__main__':
    parser = init_cesm_argparse()
    args = parser.parse_args()
    config = EnvironmentConfig(args.config)

    cmd = f"rm -rf {config.CESM_DIR}; "
    cmd += "svn ls https://svn-ccsm-models.cgd.ucar.edu/ww3/release_tags; "
    cmd += f"git clone git@github.com:ESCOMP/cesm.git {config.CESM_DIR}; "
    cmd += f"cd {config.CESM_DIR}; "
    cmd += "git checkout release-cesm2.2.0; "
    cmd += "./manage_externals/checkout_externals; "

    if not args.config_only:
        os.system(cmd)

    cmd = f" cp {config.CESM_REPO_DIR}/SrcMods/*xml "
    cmd += f"{config.CESM_DIR}/cime/config/cesm/machines/; "
    cmd += f"mkdir -p {config.RUN_DIR}/inputdata; "

    os.system(cmd)

    with open(f"{config.CESM_REPO_DIR}/SrcMods/config_machines.xml", 'r') as f:
        CONFIG_MACHINES = f.read()

    config_machines_file = f"{config.CESM_DIR}"
    config_machines_file += "/cime/config/cesm/machines/config_machines.xml"
    with open(config_machines_file, 'w') as f:
        f.write(CONFIG_MACHINES.replace(
            '%RUN_DIR%',
            config.RUN_DIR).replace(
                '%SCRATCH_DIR%',
                config.SCRATCH_DIR).replace(
                    '%INPUT_DATA_DIR%', config.CESM_INPUT_DATA_DIR))
        f.close()
