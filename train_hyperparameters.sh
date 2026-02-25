# python train_constrained_dqn.py tiny-gen policies/cdqn_tiny-gen.pth --training_steps 50000 --lambda-lr 0.01
# python train_constrained_dqn.py tiny-gen policies/cdqn_tiny-gen.pth --training_steps 50000 --lambda-lr 0.001
# python train_constrained_dqn.py tiny-gen policies/cdqn_tiny-gen.pth --training_steps 50000 --lambda-lr 0.0001

python train_constrained_dqn.py tiny-small policies/cdqn_tiny-small.pth --training_steps 100000 --lambda-lr 0.01
python train_constrained_dqn.py tiny-small policies/cdqn_tiny-small.pth --training_steps 100000 --lambda-lr 0.001
python train_constrained_dqn.py tiny-small policies/cdqn_tiny-small.pth --training_steps 100000 --lambda-lr 0.0001

# python train_constrained_dqn.py tiny-hard policies/cdqn_tiny-hard.pth --training_steps 100000 --lambda-lr 0.01
# python train_constrained_dqn.py tiny-hard policies/cdqn_tiny-hard.pth --training_steps 100000 --lambda-lr 0.001
# python train_constrained_dqn.py tiny-hard policies/cdqn_tiny-hard.pth --training_steps 100000 --lambda-lr 0.0001



