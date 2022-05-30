*************
Ecrlgcm
*************
Tools for setting up gcms (isca and cesm) and running on aws.

Initialization
==============

Edit environment configuration:

.. code-block:: bash

    cd gcm_aws
    cp environment/config.py my_config.py
    vim my_config.json

Configuration can be in either .py or .json format. Follow the required
variables from config.py. Easiest is just to edit the my_config.py file and
not convert to json.

After following the installation instructions `here <https://bnb32.github.io/gcm_aws/install.html>`_:

.. code-block:: bash

    cd gcm_aws
    bash ./go.sh

This go script kicks off scripts from the scripts directory and requires
the my_config.py file. Pip has trouble installing xesmf and cartopy so these
packages may need to be installed manually with conda.
