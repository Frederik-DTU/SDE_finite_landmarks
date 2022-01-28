#!/bin/sh
#BSUB -q hpc
#BSUB -J cc_ahs_theta
#BSUB -R "span[hosts=1]"
#BSUB -n 1 
#BSUB -W 5:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s164222@student.dtu.dk
#BSUB -o HPC_output/output_%J.out
#BSUB -e Error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
module swap python3/3.8.2

python3 corpus_callosum.py \
    --save_path corpus_callosum_models/ \
    --model ahs \
    --eta 0.98 \
    --delta 0.001 \
    --epsilon 0.001 \
    --theta 0.2 \
    --update_theta 1 \
    --max_iter 2500 \
    --save_step 50
