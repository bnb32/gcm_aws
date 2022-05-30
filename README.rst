*************
Ecrlgcm
*************
Tools for setting up gcms (isca and cesm) and running on aws. For use with
isca_aws and cesm_aws repos.

Installation
============

Install package:

.. code-block:: bash

    cd gcm_aws
    pip install -e .
    bash ./go.sh

Edit environment configuration:

.. code-block:: bash

    cd gcm_aws
    cp environment/config.py my_config.json
    vim my_config.json

Configuration needs to be in json format. Follow the required variables from
config.py
