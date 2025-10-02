Model=$1

python -m Clf_Learner.main \
--dirname credit_exp_batch128_max_20/${Model} \
--datasets credit_scoring_normalised_class_balanced.csv \
--best_response lagrange \
--cost quadratic \
--loss naive_ssvm_hinge \
--model ${Model} \
--utility strategic \
--lr 0.01 \
--batch 128 \
--epochs 20 \
--seed 0 \
--validate \
--train \
--test \
--store \
--verbose \
--implicit \
--gpu \
--args '{"max_iterations": 20000, "lr":0.0007, "margin": 5e-2, "lagrange_mult_lr": 5e-2, "lagrange_mult_cost_lr": 1e-2}'

#--model parabolic \


#--dataset ball_half_ring_dataset_ball_rad_2_in_rad_3_out_rad_5.csv