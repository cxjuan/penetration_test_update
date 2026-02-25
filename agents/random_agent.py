"""An example DQN Agent.

It uses pytorch 1.5+ and tensorboard libraries (HINT: these dependencies can
be installed by running pip install nasim[dqn])

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python dqn_agent.py tiny

To see detailed results using tensorboard:

$ tensorboard --logdir runs/

To see available hyperparameters:

$ python dqn_agent.py --help

Notes
-----

This is by no means a state of the art implementation of DQN, but is designed
to be an example implementation that can be used as a reference for building
your own agents.
"""
import random
from pprint import pprint

from gymnasium import error
import numpy as np

import nasim
from nasim.envs.action import NoOp

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install dqn_agent dependencies by running "
        "'pip install nasim[dqn]'.)"
    )




class RandomAgent:
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.001,
                 training_steps=20000,
                 batch_size=32,
                 replay_size=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 hidden_sizes=[64, 64],
                 target_update_freq=1000,
                 verbose=True,
                 **kwargs):

        # This DQN implementation only works for flat actions
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning DQN with config:")
            pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(seed)

        # environment setup
        self.env = env

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        # logger setup
        self.logger = SummaryWriter()

        self.training_steps = training_steps
        self.exploration_steps = exploration_steps
        self.steps_done = 0
        self.testing_monitor = []
        self.epsilon=0.1



    def save(self, save_path):
        pass

    def load(self, load_path):
        pass


    def get_action(self):
        return random.randint(0, self.num_actions-1)


    def train(self):
        if self.verbose:
            print("\nStarting training")

        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)
            ep_return, ep_cost, ep_steps, goal = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)

            self.logger.add_scalar(
                "episode_return", ep_return, self.steps_done
            )
            self.logger.add_scalar(
                "episode_cost", ep_cost, self.steps_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.steps_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.steps_done
            )

            if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tcost = {ep_cost}")
                print(f"\tgoal = {goal}")

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tcost = {ep_cost}")
            print(f"\tgoal = {goal}")

    def run_train_episode(self, step_limit):

        o, _ = self.env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0
        episode_cost = 0

        action_sequence = []
        text_sequence = []
        testing_results = dict()

        while not done and not env_step_limit_reached and steps < step_limit:
            a = self.get_action()

            next_o, r, done, env_step_limit_reached, info = self.env.step(a)
            self.steps_done += 1

            o = next_o
            episode_return += r
            episode_cost += 1
            steps += 1

            real_action = self.env.action_space.get_action(a)
            action_sequence.append(a)
            text_sequence.append(str(real_action))

        testing_results["actions"] = action_sequence
        testing_results["texts"] = text_sequence
        testing_results["goal"] = self.env.goal_reached()
        self.testing_monitor.append(testing_results)

        episode_return = max(0, episode_return)
        return episode_return, episode_cost, steps, self.env.goal_reached()

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
                         render_mode="human"):
        if env is None:
            env = self.env

        original_render_mode = env.render_mode
        env.render_mode = render_mode

        o, _ = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render()
            input("Initial state. Press enter to continue..")

        while not done and not env_step_limit_reached:
            a = self.get_action()
            next_o, r, done, env_step_limit_reached, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render()
                print(f"Reward = {r}")
                print(f"Done = {done}")
                print(f"Step limit reached = {env_step_limit_reached}")
                input("Press enter to continue..")

                if done or env_step_limit_reached:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        env.render_mode = original_render_mode
        return episode_return, steps, env.goal_reached()