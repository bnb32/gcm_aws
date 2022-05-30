*************
Ecrlgcm
*************
Tools for setting up gcms (isca and cesm) and running on aws. For use with
isca_aws and cesm_aws repos.

Installation
============

Edit environment configuration:

.. code-block:: bash

    cd gcm_aws
    cp ecrlgcm/environment/config.py my_config.py
    vim my_config.json

Configuration can be in either .py or .json format. Follow the required
variables from config.py. Easiest is just to edit the my_config.py file and
not convert to json.

Install package:

.. code-block:: bash

    cd gcm_aws
    pip install -e .
    bash ./go.sh

This go script kicks off scripts from the scripts directory and requires
the my_config.py file. Pip has trouble installing xesmf and cartopy so these
packages may need to be installed manually with conda.
