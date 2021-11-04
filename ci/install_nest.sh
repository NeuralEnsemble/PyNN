#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.9" ]; then
    echo -e "\n========== Installing NEST ==========\n"

    sudo apt-get install nest

    python -c "import nest";
fi
