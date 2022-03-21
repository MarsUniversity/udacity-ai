#!/bin/bash

set -e

PYTORCH="pytorch"
# . ~/venv/bin/activate
# deactivate
# . ./pytorch/bin/activate
# which jupyter-lab
# echo "$PATH"

clean_dir () {
    # if pytorch exists then delete it
    if [[ -d "$PYTORCH" ]]; then
        rm -fr "./$PYTORCH"
        echo "*** Deleted pytorch folder ***"
    fi
}

build () {
    clean_dir

    python3 -m venv "./$PYTORCH"
    . "./$PYTORCH/bin/activate"
    pip install -U pip setuptools wheel colorama
    pip install -U jupyterlab matplotlib numpy pandas
    pip install -U torch torchvision
    # pip install opencv-contrib-python
}

if [[ $# -ne 0 ]] ; then
    if [[ $1 = "build" ]]; then
        echo ">> Building environment"
        build
        exit 0
    fi
    if [[ $1 = "test" ]]; then
        echo ">> Test"
        exit 0
    fi
fi

if [[ ! -d "$PYTORCH" ]]; then
    echo "*** Missing pytorch folder ... will build ***"
    build
else
    echo ">> Good to go!"
    . "./$PYTORCH/bin/activate"
fi


# . ./pytorch/bin/activate
jupyter-lab
