#--gen-small-hard env
#python train_dqn.py             tiny-gen policies/dqn_tiny-gen.pth --training_steps 50000
#python train_dqn.py             tiny-small policies/dqn_tiny-small.pth --training_steps 100000
#python train_dqn.py             tiny-hard policies/dqn_tiny-hard.pth --training_steps 100000

#python run_random.py             tiny-gen policies/random_tiny-gen.pth --training_steps 50000
#python run_random.py             tiny-small policies/random_tiny-small.pth --training_steps 100000
#python run_random.py             tiny-hard policies/random_tiny-hard.pth --training_steps 100000

#python run_rule_based.py          tiny-gen policies/rule_tiny-gen.pth --training_steps 50000
#python run_rule_based.py          tiny-small policies/rule_tiny-small.pth --training_steps 100000
#python run_rule_based.py          tiny-hard policies/rule_tiny-hard.pth --training_steps 100000

#--Scaler----
#python train_constrained_dqn.py tiny-gen Constrained_DQN policies/cdqn_tiny-gen.pth 1  --training_steps 50000
#python train_constrained_dqn.py tiny-small Constrained_DQN policies/cdqn_tiny-gen.pth 1  --training_steps 50000
#python train_constrained_dqn.py tiny-hard Constrained_DQN policies/cdqn_tiny-gen.pth 1  --training_steps 50000

#--Hybrid----
#python train_constrained_dqn_lambdaNet.py tiny-gen Constrained_DQN_LambdaNet policies/cdqnNet_tiny-small.pth 1 --training_steps 100000
#python train_constrained_dqn_lambdaNet.py tiny-small Constrained_DQN_LambdaNet policies/cdqnNet_tiny-small.pth 1 --training_steps 100000
#python train_constrained_dqn_lambdaNet.py tiny-hard Constrained_DQN_LambdaNet policies/cdqnNet_tiny-small.pth 1 --training_steps 100000

#--Pure----
#python train_constrained_dqn_lambdaNet.py tiny-gen Constrained_DQN_PureNet policies/cdqnPNet_tiny-small.pth 1 --training_steps 100000
#python train_constrained_dqn_lambdaNet.py tiny-small Constrained_DQN_PureNet policies/cdqnPNet_tiny-small.pth 1 --training_steps 100000
#python train_constrained_dqn_lambdaNet.py tiny-hard Constrained_DQN_PureNet policies/cdqnPNet_tiny-small.pth 1 --training_steps 100000
