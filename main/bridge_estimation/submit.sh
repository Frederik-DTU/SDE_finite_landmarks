#!/bin/sh
#BSUB -q hpc
#BSUB -J ellipse_ahs_theta
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

python3 ellipse.py \
    --save_path ellipse_saved/ \
    --model ahs \
    --eta 0.98 \
    --delta 0.001 \
    --epsilon 0.001 \
    --theta 0.2 \
    --update_theta 1 \
    --max_iter 20000 \
    --save_step 500
