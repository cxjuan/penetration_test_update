#--gen-small-hard env
#python train_dqn.py             tiny-gen policies/dqn_tiny-gen.pth --training_steps 50000
#python train_constrained_dqn.py tiny-gen policies/cdqn_tiny-gen.pth --training_steps 50000
#
#
#python train_dqn.py             tiny-small policies/dqn_tiny-small.pth --training_steps 100000
#python train_constrained_dqn.py tiny-small policies/cdqn_tiny-small.pth --training_steps 100000
#
#
#python train_dqn.py             tiny-hard policies/dqn_tiny-hard.pth --training_steps 100000
#python train_constrained_dqn.py tiny-hard policies/cdqn_tiny-hard.pth --training_steps 100000
#
#
#python run_random.py             tiny-gen policies/random_tiny-gen.pth --training_steps 50000
#python run_random.py             tiny-small policies/random_tiny-small.pth --training_steps 100000
#python run_random.py             tiny-hard policies/random_tiny-hard.pth --training_steps 100000
#
#
#python run_rule_based.py          tiny-gen policies/rule_tiny-gen.pth --training_steps 50000
#python run_rule_based.py          tiny-small policies/rule_tiny-small.pth --training_steps 100000
#python run_rule_based.py          tiny-hard policies/rule_tiny-hard.pth --training_steps 100000


##--Extend: standard small/medium env
#python train_dqn.py               small policies/dqn_stan-small.pth  1 --training_steps 2000000
#python run_random.py              small policies/random_stan-small.pth 1 --training_steps 2000000
#python run_rule_based.py          small policies/rule_stan-small.pth 1 --training_steps 2000000

#python train_dqn.py               small policies/dqn_stan-small.pth  2 --training_steps 2000000
#python run_random.py              small policies/random_stan-small.pth 2 --training_steps 2000000
#python run_rule_based.py          small policies/rule_stan-small.pth 2 --training_steps 2000000
#
#python train_dqn.py               small policies/dqn_stan-small.pth  3 --training_steps 2000000
#python run_random.py              small policies/random_stan-small.pth 3 --training_steps 2000000
#python run_rule_based.py          small policies/rule_stan-small.pth 3 --training_steps 2000000

#python train_dqn.py                medium policies/dqn_stan-medium.pth  1 --training_steps 2500000
#python run_random.py              medium policies/random_stan-medium.pth 1 --training_steps 2500000
#python run_rule_based.py          medium policies/rule_stan-medium.pth 1 --training_steps 2500000
#
#python train_dqn.py               medium policies/dqn_stan-medium.pth  2 --training_steps 2500000
#python run_random.py              medium policies/random_stan-medium.pth 2 --training_steps 2500000
#python run_rule_based.py          medium policies/rule_stan-medium.pth 2 --training_steps 2500000
#
#python train_dqn.py               medium policies/dqn_stan-medium.pth  3 --training_steps 2500000
#python run_random.py              medium policies/random_stan-medium.pth 3 --training_steps 2500000
#python run_rule_based.py          medium policies/rule_stan-medium.pth 3 --training_steps 2500000

#python train_constrained_dqn.py   small policies/cdqn_stan-small.pth 1 --training_steps 2000000
#python train_constrained_dqn.py   small policies/cdqn_stan-small.pth 2 --training_steps 2000000
#python train_constrained_dqn.py   small Constrained_DQN policies/cdqn_stan-small.pth 3 --training_steps 2000000
##
#python train_constrained_dqn.py   medium policies/cdqn_stan-medium.pth 1 --training_steps 2500000
#python train_constrained_dqn.py   medium policies/cdqn_stan-medium.pth 2 --training_steps 2500000
#python train_constrained_dqn.py   medium policies/cdqn_stan-medium.pth 3 --training_steps 2500000


#
#python train_constrained_dqn_lambdaNet.py   small Constrained_DQN_LambdaNet policies/cdqnNet_stan-small.pth 1 --training_steps 3000000
#python train_constrained_dqn_lambdaNet.py   medium Constrained_DQN_LambdaNet policies/cdqnNet_stan-medium.pth 1 --training_steps 3000000
#
#python train_constrained_dqn_lambdaNet.py   small Constrained_DQN_LambdaNet policies/cdqnNet_stan-small.pth 2 --training_steps 3000000
#python train_constrained_dqn_lambdaNet.py   medium Constrained_DQN_LambdaNet policies/cdqnNet_stan-medium.pth 2 --training_steps 3000000
#
#python train_constrained_dqn_lambdaNet.py   small Constrained_DQN_LambdaNet policies/cdqnNet_stan-small.pth 3 --training_steps 3000000
#python train_constrained_dqn_lambdaNet.py   medium Constrained_DQN_LambdaNet policies/cdqnNet_stan-medium.pth 3 --training_steps 3000000

set -eu

MAX_TRIES=500
for try in $(seq 1 "$MAX_TRIES"); do
  echo "=== Try $try ==="

  python train_constrained_dqn_lambdaNet.py tiny-small Constrained_DQN_LambdaNet \
    policies/cdqnNet_tiny-small.pth 3 --training_steps 100000

  if python plot_results_2.py; then
    echo "SUCCESS: plot_results condition met (>0.970)"
    exit 0
  else
    echo "Not enough yet."
  fi
done

echo "FAILED: did not exceed threshold after $MAX_TRIES tries"
exit 1

