"""
Run a SoftwareBasedAgent and log its performance like DQNAgent.
"""

import os
import json
import argparse
import nasim
import numpy as np
from agents.software_agent import SoftwareBasedAgent

LINE = "-" * 60
results = []

def convert_np(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    return obj

def build_action_map(env, noop_index):
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
                # Skip subnet scan only if we've already scanned it (tracked by self.scanned)
                if (host, scan_type) in scanned:
                    continue

                scanned.add((host, scan_type))
                return action_idx

        return noop_index

    return {
        'exploit': exploit_fn,
        'scan': scan_fn,
        'noop': lambda: noop_index
    }

def mark_all_services_exploited(agent, env, host_id):
    host = env.current_state.get_host(host_id)
    print(f'--mark_all_services_exploited----host_id = {host_id}--')
    if host.compromised:
        for s, active in host.services.items():
            if active:
                agent.exploited_hosts.add((host_id, s))
                print(f"🧩 Auto-marked exploited: ({host_id}, {s})")
    # if host.compromised:
    #     for s in agent.known_exploits:
    #         agent.exploited_hosts.add((host_id, s))


def make_action_map(env, noop_index):
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

    # Fallback noop — pick a known safe scan or invalid action
    if noop_index is None:
        # Pick a safe default — the first scan found
        for i in range(env.action_space.n):
            action = env.action_space.get_action(i)
            # if action.name in {"service_scan", "os_scan", "subnet_scan", "process_scan"}:
            #     print(f"Warning: No real noop found, using '{action.name}' at index {i} as fallback.")
            #     noop_index = i
            if action.name == "noop":
                noop_index = i
                break
        if noop_index is None:
            raise RuntimeError("No noop and no valid scan to fake a noop fallback.")

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
        known_targets = sorted(set(key[1] for key in reverse_action_map if key[0] in preferred_scans))

        print(reverse_action_map)

        for scan_type in preferred_scans:
            for host in known_targets:
                key = (scan_type, host, None)
                action_idx = reverse_action_map.get(key)
                if action_idx is None:
                    continue

                action = env.action_space.get_action(action_idx)
                h = env.network.hosts.get(host)

                # Skip if host is not reachable or compromised
                if not h or (not h.reachable and not h.compromised):
                    continue

                # Skip if already scanned
                if (host, scan_type) in scanned:
                    continue

                # Skip scans that yield no new info
                if scan_type == "service_scan" and any(h.services.values()):
                    continue
                if scan_type == "os_scan" and any(h.os.values()):
                    continue
                if scan_type == "process_scan" and any(h.processes.values()):
                    continue
                if scan_type == "subnet_scan" and h.reachable:
                    continue

                scanned.add((host, scan_type))
                return action_idx

        return noop_index  # Safe fallback

    return {
        'exploit': exploit_fn,
        'scan': scan_fn,
        'noop': lambda: noop_index
    }




def extract_known_exploits(env):
    known_exploits = set()
    for exploit_def in env.scenario.exploits.values():
        known_exploits.add(exploit_def['service'])
    return list(known_exploits)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("save_path", type=str, help="save path for best result JSON")
    parser.add_argument("-r", "--result_path", type=str, default="results",
                        help="directory to store training logs (default=results/)")
    parser.add_argument("--training_steps", type=int, default=10000,
                        help="total training steps (default=10000)")
    parser.add_argument("--retry_limit", type=int, default=20, help="Max retry attempts for exploit actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--partially_obs", action="store_true")
    parser.add_argument("--save_file_as_", type=int, default=1)
    args = parser.parse_args()

    # Setup output paths
    data_result_dir = os.path.join(args.result_path, 'data', args.env_name)
    testing_result_dir = os.path.join(args.result_path, 'testing', args.env_name)
    os.makedirs(data_result_dir, exist_ok=True)
    os.makedirs(testing_result_dir, exist_ok=True)

    # Create NASim environment
    env = nasim.make_benchmark(
        args.env_name,
        args.seed,
        fully_obs=True,
        flat_actions=True,
        flat_obs=True,

    )


    from nasim.envs.action import NoOp
    # 1. Inject NoOp into the action space manually (without changing NASim)
    noop_action = NoOp()
    env.action_space.actions.append(noop_action)
    # 2. Store the noop index for convenience
    # Patch action_space.n to match actual actions
    if len(env.action_space.actions) > env.action_space.n:
        env.action_space.n = len(env.action_space.actions)
    noop_index = env.action_space.n - 1

    # Reset environment so this access is registered
    env.reset()




    # Init agent
    # action_map = make_action_map(env, noop_index)
    action_map = build_action_map(env, noop_index)

    known_exploits = extract_known_exploits(env)
    agent = SoftwareBasedAgent(env, action_map, known_exploits, max_exploit_retries=args.retry_limit)

    # Training loop
    best_result = None
    best_reward = -float("inf")
    testing_monitor = []
    steps_total = 0
    episode_count = 0


    while steps_total < args.training_steps:
        obs, _ = env.reset()
        agent.reset()
        done = False
        env_step_limit_reached = False

        reward = 0
        cost = 0
        steps = 0
        r = 0
        action_sequence = []
        text_sequence = []

        # for k, v in env.scenario.exploits.items():
        #     print(k, v)

        while not done and not env_step_limit_reached and steps_total < args.training_steps:

            """ get action """
            action = agent.select_action(obs)
            action_obj = env.action_space.get_action(action)
            # if action_obj.name == "noop":
            #     break

            """ penetration """
            obs, r, done, env_step_limit_reached, info = env.step(action)

            """ Failed -> mark current exploits as [unexploited], and retry """
            if action_obj.is_exploit() and not info.get("success", False):
                agent.failed_exploit_attempts[(action_obj.target, action_obj.service)] += 1

            """ Success -> update current new state to host """
            agent.update_internal_state(obs)
            """ Success -> mark host as [exploited] """
            if action_obj.is_exploit() and info["success"]:
                agent.exploited_hosts.add((action_obj.target, action_obj.service))

            reward += r
            cost += 1
            steps += 1
            steps_total += 1

            action_sequence.append(action)
            text_sequence.append(str(action_obj))

        goal = env.goal_reached()

        results.append([steps_total, float(reward), float(cost), float(steps), goal])
        testing_monitor.append({
            "actions": action_sequence,
            "texts": text_sequence,
            "goal": goal
        })

        if goal and reward > best_reward:
            best_reward = reward
            best_result = {
                "episode": episode_count,
                "steps_done": steps_total,
                "reward": reward,
                "cost": cost,
                "goal": goal,
                "path": text_sequence
            }

        episode_count += 1

        # print(f"[Episode {episode_count}] reward={reward}, cost={cost}, goal={goal}")

    # Save results
    file_No = args.save_file_as_
    with open(os.path.join(data_result_dir, f"Rule_{file_No}.json"), "w") as f:
        json.dump(results, f, default=convert_np)

    with open(os.path.join(testing_result_dir, f"Rule_testing_{file_No}.json"), "w") as f:
        json.dump(testing_monitor, f, default=convert_np)

    if best_result:
        with open(args.save_path, "w") as f:
            json.dump(best_result, f, default=convert_np)

    print(f"\n{LINE}\nTraining Complete\n{LINE}")
    print(f"Best Reward: {best_reward}")
    print(f"Best result saved to: {args.save_path}")
    print(f"Training logs saved to: {data_result_dir} and {testing_result_dir}")

if __name__ == "__main__":
    main()