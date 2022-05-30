#!/bin/bash

echo "Mounting storage"

bash ./scripts/storage.sh

echo "Downloading cesm"

python ./scripts/init_cesm.py -config my_config.json

echo "Downloading and installing isca"

python ./scripts/init_isca.py -config my_config.json

echo "Downloading paleo-continent maps"

python ./scripts/get_maps.py -config my_config.json

echo "Installing packages"

python ./scripts/init_env.py -config my_config.json
