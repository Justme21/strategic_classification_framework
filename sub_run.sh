python -m Clf_Learner.main --dirname results_holder \
--datasets ball_half_ring_dataset_ball_rad_2_in_rad_3_out_rad_5.csv \
--best_response projected_grad \
--cost quadratic \
--loss naive_ssvm_hinge \
--model icnn \
--utility hinge \
--lr 0.01 \
--batch 2048 \
--epochs 20 \
--seed 0 \
--train \
--test \
--store \
--verbose \
--args '{"best_response": {"max_iterations": 500, "lr":0.01}, "cost": {"radius": 2.0}, "loss": {"gamma": 0},  "utility": {"margin": 0.01}}'

#--model parabolic \
