********
Examples
********

Run Simulations
===============

Run a CESM simulation for 300Ma BP:

.. code-block:: bash

    python scripts/run_cesm.py -year 300.0 -run_all -config my_config.json


Run an Isca simulation for 300Ma BP:

.. code-block:: bash

    python scripts/run_isca.py -year 300.0 -config my_config.json


Visualization
=============

Run interactive dashboard:

.. code-block:: bash

    python postprocessing/app.py -config my_config.json