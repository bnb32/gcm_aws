#!/bin/bash

echo "Mounting storage"

bash ./scripts/storage.sh

echo "Downloading cesm"

python ./scripts/init_cesm.py

echo "Downloading and installing isca"

python ./scripts/init_isca.py

echo "Downloading paleo-continent maps"

python ./scripts/get_maps.py

echo "Installing packages"

python ./scripts/init_env.py
