Clone repo (recommended for developers)
---------------------------------------

1. from home dir, ``git clone git@github.com:bnb32/gcm_aws.git``

2. Create ``ecrlgcm`` environment and install package
    1) Create a conda env: ``conda create -n ecrlgcm``
    2) Run the command: ``conda activate ecrlgcm``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``ecrlgcm`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)
    6) Install xesmf and cartopy with conda (``conda install xesmf cartopy``)