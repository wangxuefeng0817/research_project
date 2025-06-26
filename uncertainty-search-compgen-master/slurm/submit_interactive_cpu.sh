#!/bin/bash

cd $WRKDIR
module load scicomp-python-env

BASEDIR=${BASEDIR:-$WRKDIR}
VENV=${VENV:-$WRKDIR/venv/activate}
PORT=${PORT:-8099}

source ${VENV}

cd $BASEDIR; jupyter notebook --port $PORT --NotebookApp.password='' --NotebookApp.token='' --ip='*'
