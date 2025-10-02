TODO
How to run this code: (Requires Python version< 3.12.0)

#python3 -m Clf_Learner.main
--dirname <name to write the results to in results/ directory>

--datasets \ # e.g. twin_moons, ball, ring, ... 

--best_response <best_response_name> \ # lagrange, identity 

--cost quadratic \ # Currently no other costs of interest 

--loss naive_ssvm_hinge \ # Currently no other losses of interest 

--model \ # linear, quadratic, mlp 

--utility strategic \ # Currently no other utilities of interest 

--lr 0.01 \ # Outer loop learning rate 

--batch 128 \ # Outer loop batch size 

--epochs 50 \ # Number of epochs to run outer loop for 

--seed 0 \ # Supposed to help with reproducibility 

--gpu \ # Binary flag. Include to run on GPU 

--implicit \ # Binary flag, Include to run implicit gradient algorithm

--train \ # Binary flag. Include to run training 

--validate \ # Binary flag. Include to perform validation

--test \ # Binary flag. Include to perform testing 

--store \ # Binary flag. Include to store results 

--verbose \ # Binary flag. Include to produce debug text during run 

--args '{"best_response": {"max_iterations": 10000, "lr":0.005, "margin": 0.05, "lagrange_mult_lr": 0.0005, "lagrange_mult_cost_lr": 0.01}, "cost": {"radius": 2.0}}' # Hyperparamters specific to the various modules being run. Each args entry here gets mapped to the corresponding module during initialisation.

see python3 -m Clf_Learner.main --help for more details on argument meaning

Identified good hyperparameter settings for best response:

Model: Linear; {"max_iterations": 10000, "lr":0.005, "margin": 5e-2, "lagrange_mult_lr": 5e-4, "lagrange_mult_cost_lr": 1e-2}
Model: Quadratic/MLP; {"max_iterations": 20000, "lr":0.0007, "margin": 5e-2, "lagrange_mult_lr": 5e-2, "lagrange_mult_cost_lr": 1e-2}
