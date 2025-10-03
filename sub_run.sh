Model=$1

python -m Clf_Learner.main \
--dirname credit_batch256_max_10_lr0001_t212_y11/${Model} \
--datasets credit_scoring_normalised_class_balanced.csv \
--best_response lagrange \
--cost quadratic \
--loss naive_ssvm_hinge \
--model ${Model} \
--utility strategic \
--lr 0.01 \
--batch 256 \
--epochs 50 \
--seed 123 \
--validate \
--train \
--test \
--store \
--verbose \
--implicit \
--gpu \
--args '{"best_response": {"max_iterations": 10000, "lr":0.001, "margin": 0.05, "lagrange_mult_lr": 0.0005, "lagrange_mult_cost_lr": 0.01}, "cost": {"radius": 2.0}, "model": {"hidden_layers": 1, "hidden_dim": 4}, "datasets": {"strat_cols": [0, 5, 7]}}'

#--args '{"max_iterations": 20000, "lr":0.001, "margin": 5e-2, "lagrange_mult_lr": 5e-2, "lagrange_mult_cost_lr": 1e-2}'


#--model parabolic \


#--args '{"max_iterations": 20000, "lr":0.0007, "margin": 5e-2, "lagrange_mult_lr": 5e-2, "lagrange_mult_cost_lr": 1e-2}'

#--dataset ball_half_ring_dataset_ball_rad_2_in_rad_3_out_rad_5.csv