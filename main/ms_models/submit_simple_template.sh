#!/bin/sh
#BSUB -q hpc
#BSUB -J ms_simple_template
#BSUB -R "span[hosts=1]"
#BSUB -n 4 
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s164222@student.dtu.dk
#BSUB -o HPC_output/output_%J.out
#BSUB -e Error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
module swap python3/3.8.2

python3 simple_template.py \
    --save_path simple_template/ \
    --eta 0.98 \
    --delta 0.01 \
    --lmbda 1.0 \
    --epsilon 0.001 \
    --time_step 0.001 \
    --t0 0.0 \
    --T 1.0 \
    --max_iter 20000 \
    --save_step 500
