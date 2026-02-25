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
import collections

from gymnasium import error
import numpy as np

import nasim
from nasim.envs.host_vector import HostVector
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


class RuleBasedAgent:
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

        # This implementation only works for flat actions
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning RuleBasedAgent with config:")
            pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # environment setup
        self.env = env

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        # logger setup
        self.logger = SummaryWriter()

        self.training_steps = training_steps
        self.exploration_steps = exploration_steps
        self.steps_done = 0

        noop_index = None
        for i in range(self.env.action_space.n):
            a = self.env.action_space.get_action(i)
            if isinstance(a, NoOp) or a.name in {"noop", "NoOp"}:
                noop_index = i
                break
        if noop_index is None:
            # Inject manually noop action array number for actions failed to RuleBased logic
            noop_action = NoOp()
            env.action_space.actions.append(noop_action)
            noop_index = len(self.env.action_space.actions) - 1
            self.env.action_space.n = len(self.env.action_space.actions)

        action_map = self.build_action_map(self.env, noop_index)

        self.action_map = action_map
        self.known_exploits = self.extract_known_exploits()
        self.exploited_hosts = set()
        self.scanned = set()
        self.rooted_hosts = set()
        self.known_access = {}
        self.failed_exploit_attempts = collections.Counter()
        self.max_exploit_retries = max(2, len(self.env.network.hosts))
        self.testing_monitor = []


    def save(self, save_path):
        pass

    def load(self, load_path):
        pass


    def convert_np(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return obj

    def build_action_map(self, env, noop_index):
        reverse_action_map = {}

        for i in range(env.action_space.n):
            action = env.action_space.get_action(i)

            if action.is_exploit():
                key = ("Exploit", action.target, action.service)
            # Privilege Escalation
            elif action.is_privilege_escalation():
                key = ("PrivilegeEscalate", action.target, action.name[3:])
            # Scans
            elif action.name in {"service_scan", "os_scan", "subnet_scan", "process_scan"}:
                key = (action.name, action.target, None)
            elif action.name == "noop":
                noop_index = i
                key = (action.name, action.target, None)
            else:
                key = (action.name, action.target, None)

            reverse_action_map[key] = i

        def exploit_fn(host_id, vuln_name):
            action_tuple = ("Exploit", host_id, vuln_name)
            fallback_tuple = ("Exploit", host_id, None)
            if action_tuple in reverse_action_map:
                return reverse_action_map[action_tuple]
            elif fallback_tuple in reverse_action_map:
                return reverse_action_map[fallback_tuple]
            else:
                print(f"No exploit action found for: {action_tuple}")
                return None

        def scan_fn(scanned):
            preferred_scans = ["subnet_scan", "service_scan", "os_scan", "process_scan"]

            for host in env.network.hosts:
                h = env.network.hosts[host]
                if not h.reachable and not h.compromised:
                    continue

                for scan_type in preferred_scans:
                    key = (scan_type, host, None)
                    action_idx = reverse_action_map.get(key)

                    if action_idx is None or (host, scan_type) in scanned:
                        continue
                    if scan_type == "service_scan" and h.reachable:
                        continue
                    if scan_type == "os_scan" and any(h.os.values()):
                        continue
                    if scan_type == "process_scan" and any(h.processes.values()):
                        continue
                    if scan_type == "subnet_scan" and any(
                            n.reachable for n in env.network.hosts.values() if n.address != host):
                        continue

                    scanned.add((host, scan_type))
                    return action_idx

            return noop_index

        return {
            'exploit': exploit_fn,
            'scan': scan_fn,
            'noop': lambda: noop_index
        }

    def mark_all_services_exploited(self, agent, env, host_id):
        host = env.current_state.get_host(host_id)
        print(f'--mark_all_services_exploited----host_id = {host_id}--')
        if host.compromised:
            for s, active in host.services.items():
                if active:
                    agent.exploited_hosts.add((host_id, s))
                    print(f"Auto-marked exploited: ({host_id}, {s})")


    def extract_known_exploits(self):
        known_exploits = set()
        for exploit_def in self.env.scenario.exploits.values():
            known_exploits.add(exploit_def['service'])
        return list(known_exploits)


    def update_internal_state(self, obs):
        host_states = self.parse_state_vector(obs)
        for host_id, host in host_states.items():
            self.known_access[host_id] = host.get("Access", 0)
            if host.get("Access", 0) >= 2:
                self.rooted_hosts.add(host_id)

            # Backward-compatible scan detection
            if host.get("scanned_os") or any(k in host for k in ("linux", "windows")):
                self.scanned.add((host_id, "os_scan"))
            if host.get("scanned_services") or any(k in host for k in ("http", "ssh", "ftp", "samba", "smtp")):
                self.scanned.add((host_id, "service_scan"))
            if host.get("scanned_processes") or any(k in host for k in ("proc_0", "proc_1", "tomcat", "daclsvc", "schtask")):
                self.scanned.add((host_id, "process_scan"))
            if host.get("scanned_subnet"):
                self.scanned.add((host_id, "subnet_scan"))

    def reset(self):
        self.exploited_hosts.clear()
        self.scanned.clear()
        self.rooted_hosts.clear()
        self.known_access.clear()
        self.failed_exploit_attempts.clear()

    def reset_episode_memory(self):
        self.exploited_hosts.clear()
        self.rooted_hosts.clear()
        self.known_access.clear()
        self.failed_exploit_attempts.clear()


    def _expected_access(self, service):
        for k in self.env.action_space.actions:
            if k.name.startswith("e_") and k.service == service:
                return k.access
        return 0  # fallback

    allowed_procs = {"proc_0", "proc_1", "tomcat", "daclsvc", "schtask"}

    def parse_state_vector(self, state):
        host_states = {}
        num_hosts = len(self.env.network.hosts)
        state = np.array(state)

        try:
            state_matrix = state.reshape((num_hosts + 1, -1))
        except ValueError as e:
            raise ValueError(f"Cannot reshape obs of length {len(state)} to ({num_hosts + 1}, -1)") from e

        host_tensor = state_matrix[:-1]
        host_ids = list(self.env.network.hosts.keys())

        known_services = {"http", "ssh", "ftp", "samba", "smtp"}
        known_processes = {"tomcat", "daclsvc", "schtask"}
        known_os = {"linux", "windows"}

        for i, host_id in enumerate(host_ids):
            obs = host_tensor[i]
            readable = HostVector.get_readable(obs)

            host_state = {
                "reachable": readable.get("Reachable"),
                "discovered": readable.get("Discovered"),
                "access": int(readable.get("Access")),
                "value": float(readable.get("Value")),
                "compromised": readable.get("Compromised"),
                "scanned_os": readable.get("Scanned OS", False),
                "scanned_services": readable.get("Scanned Services", False),
                "scanned_processes": readable.get("Scanned Processes", False),
                "scanned_subnet": readable.get("Scanned Subnet", False),
            }

            for key, value in readable.items():
                if key.startswith("srv_") or key.startswith("os_") or key.startswith("proc_"):
                    host_state[key] = value
                    if key.startswith("srv_"):
                        host_state["scanned_services"] = value
                    if key.startswith("os_"):
                        host_state["scanned_os"] = value
                    if key.startswith("proc_"):
                        host_state["scanned_processes"] = value
                elif key in known_services or key in known_processes or key in known_os:
                    host_state[key] = value
                    if key in known_services:
                        host_state["scanned_services"] = value
                    if key in known_os:
                        host_state["scanned_os"] = value
                    if key in known_processes:
                        host_state["scanned_processes"] = value

            host_states[host_id] = host_state
        return host_states

    def _exploit_step(self, host_states, current_scanned):
        # Exploit
        for host_id in random.sample(list(host_states.keys()), len(host_states)):
            host = host_states[host_id]
            if not host["reachable"] or not host["discovered"]:
                continue
            for service in random.sample(self.known_exploits, len(self.known_exploits)):
                if not host.get("scanned_services", False):
                    continue
                if not (host.get(f"srv_{service}") or host.get(service)):
                    continue
                if (host_id, service) in self.exploited_hosts:
                    continue
                if self.failed_exploit_attempts[(host_id, service)] >= self.max_exploit_retries:
                    continue
                return self.action_map['exploit'](host_id, service)
        return None

    def _priv_esc_step(self, host_states, current_scanned):
        # Privilege Escalation
        action_indices = list(range(self.env.action_space.n))
        random.shuffle(action_indices)
        for i in action_indices:
            action = self.env.action_space.get_action(i)
            if not action.is_privilege_escalation():
                continue
            host = host_states.get(action.target)
            if not host or not host["reachable"] or not host["discovered"]:
                continue
            if host["access"] < action.req_access:
                continue
            if host["access"] >= action.req_access and host["access"] < action.access:
                if host.get(f"proc_{action.process}", False) or host.get(action.process, False):
                    return i
        return None

    def _subnet_scan_step(self, host_states, current_scanned):
        # Subnet Scan from privileged hosts
        action_indices = list(range(self.env.action_space.n))
        random.shuffle(action_indices)
        for i in action_indices:
            action = self.env.action_space.get_action(i)

            if action.name == "subnet_scan":
                key = (action.target, action.name)
                host = host_states.get(action.target)
                if host and host["access"] >= action.req_access and key not in current_scanned:
                    current_scanned.add(key)
                    return i
        return None

    def _os_proc_scan_step(self, host_states, current_scanned):
        # OS/Process Scan
        action_indices = list(range(self.env.action_space.n))
        random.shuffle(action_indices)
        for i in action_indices:
            action = self.env.action_space.get_action(i)

            if action.name in {"os_scan", "process_scan"}:
                key = (action.target, action.name)
                host = host_states.get(action.target)
                if host and host["reachable"] and key not in current_scanned:
                    current_scanned.add(key)
                    return i
        return None

    def _service_scan_step(self, host_states, current_scanned):
        # Fallback Service Scan
        action_indices = list(range(self.env.action_space.n))
        random.shuffle(action_indices)
        for i in action_indices:
            action = self.env.action_space.get_action(i)

            if action.name == "service_scan":
                key = (action.target, action.name)
                host = host_states.get(action.target)
                if host and host["reachable"] and key not in current_scanned:
                    current_scanned.add(key)
                    return i
        return None

    def _fallback_scan_step(self, host_states, current_scanned):
        # Global fallback using scan_fn
        scan_idx = self.action_map['scan'](current_scanned)
        if scan_idx != self.action_map['noop']():
            return scan_idx
        return None

    def get_action(self, state):
        host_states = self.parse_state_vector(state)
        current_scanned = set(self.scanned)

        for step_func in [self._os_proc_scan_step, self._service_scan_step, self._exploit_step, self._priv_esc_step, self._subnet_scan_step,
                          self._os_proc_scan_step, self._service_scan_step, self._fallback_scan_step]:
            action = step_func(host_states, current_scanned)

            if action is not None:
                return action

        # If the action failed in RuleBased logic, we return it as noop,
        # and be reselected as 50% chance
        if random.random() < 0.5:
            valid_actions = [i for i in range(self.env.action_space.n)
                             if self.env.action_space.get_action(i).name.lower() != "noop"]
            return random.choice(valid_actions)

        return self.action_map['noop']()


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
        self.reset_episode_memory()

        while not done and not env_step_limit_reached and steps < step_limit:

            a = self.get_action(o)
            real_action = self.env.action_space.get_action(a)

            next_o, r, done, env_step_limit_reached, info = self.env.step(a)
            self.steps_done += 1
            self.update_internal_state(next_o)

            if real_action.is_exploit() and not info.get("success"):
                self.failed_exploit_attempts[(real_action.target, real_action.service)] += 1
            """ Success -> mark host as [exploited] """
            if real_action.is_exploit() and info["success"]:
                self.exploited_hosts.add((real_action.target, real_action.service))

            if real_action.name == "subnet_scan" and info.get("discovered"):
                newly_found = [addr for addr, val in info["discovered"].items() if val]
                if newly_found:
                    self.scanned.add((real_action.target, "subnet_scan"))

            o = next_o
            episode_return += r
            episode_cost += 1
            steps += 1
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
            a = self.get_action(o)
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

