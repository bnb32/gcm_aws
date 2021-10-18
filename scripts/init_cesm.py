import ecrlcesm.environment as env

import os
import argparse

parser=argparse.ArgumentParser(description="Initialize CESM")
parser.add_argument('-config_only',default=False,action='store_true')
args=parser.parse_args()


cmd=f"rm -rf {env.CESM_DIR}; "
cmd+="svn ls https://svn-ccsm-models.cgd.ucar.edu/ww3/release_tags; "
cmd+=f"git clone git@github.com:ESCOMP/cesm.git {env.CESM_DIR}; "
cmd+=f"cd {env.CESM_DIR}; "
cmd+="git checkout release-cesm2.2.0; "
cmd+="./manage_externals/checkout_externals; "

if not args.config_only:
    os.system(cmd)

cmd=f" cp {env.MAIN_DIR}/SrcMods/*xml {env.CESM_DIR}/cime/config/cesm/machines/; "
cmd+=f"mkdir -p {env.RUN_DIR}/inputdata; "

os.system(cmd)

with open(f"{env.MAIN_DIR}/SrcMods/config_machines.xml", 'r') as f:
    CONFIG_MACHINES = f.read()

with open(f"{env.CESM_DIR}/cime/config/cesm/machines/config_machines.xml", 'w') as f:
    f.write(CONFIG_MACHINES.replace('%RUN_DIR%',env.RUN_DIR).replace('%SCRATCH_DIR%',env.SCRATCH_DIR).replace('%INPUT_DATA_DIR%',env.INPUT_DATA_DIR))
    f.close()

