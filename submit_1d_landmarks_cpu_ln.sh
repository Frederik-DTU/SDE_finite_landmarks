#!/bin/sh
#BSUB -q hpc
#BSUB -J 1d_landmarks
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

python3 1d_landmarks_ms.py \
    --save_path Model_output/low_noise/1d_landmarks \
    --eta 0.1 \
    --delta 0.1 \
    --lambda_ 1.0 \
    --epsilon 0.001 \
    --time_step 0.001 \
    --t0 0.0 \
    --T 1.0 \
    --seed 2712 \
    --max_iter 20000 \
    --save_hours 1.0 \
    --load_model_path Model_output/high_noise/1d_landmarks_ite_1000
