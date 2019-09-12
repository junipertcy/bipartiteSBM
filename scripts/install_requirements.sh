#!/bin/bash

pip3 install --upgrade pip
pip3 install -r requirements.txt

# only install test requirements if explicitly specified
if [[ "$INSTALL_TEST_REQUIREMENTS" == "true" ]]; then
    pip3 install -r requirements_test.txt
fi

pip3 list
