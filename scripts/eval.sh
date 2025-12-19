#!/bin/bash

# Note: You should also update the fid.cache_ref in configs/eval_config.yml
export LOG_DIR="YOUR_LOG_DIR"

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=${now}_${salt}_$1
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

sudo mkdir -p ${LOG_DIR}
sudo chmod 777 -R ${LOG_DIR}

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:eval \
    2>&1 | tee -a $LOG_DIR/output.log
