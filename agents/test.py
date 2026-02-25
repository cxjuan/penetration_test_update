# software_baseline_agent.py
from os import access
import numpy as np
import collections
from nasim.envs.host_vector import HostVector

class SoftwareBasedAgent:
    def __init__(self, env, action_map, known_exploits, max_exploit_retries):
        self.env = env
        self.action_map = action_map
        self.known_exploits = known_exploits
        self.exploited_hosts = set()
        self.scanned = set()
        self.rooted_hosts = set()
        self.known_access = {}
        self.failed_exploit_attempts = collections.Counter()
        self.max_exploit_retries = max_exploit_retries or max(20, len(self.env.network.hosts))

    def update_internal_state(self, obs):
        host_states = self.parse_state_vector(obs, self.env)
        for host_id, host in host_states.items():
            self.known_access[host_id] = host.get("access", 0)

            if host["access"] >= 2:
                self.rooted_hosts.add(host_id)

            # Backward-compatible scan detection
            if host.get("scanned_os") or any(k in host for k in ("linux", "windows")):
                self.scanned.add((host_id, "os_scan"))
            if host.get("scanned_services") or any(k in host for k in ("http", "ssh", "ftp")):
                self.scanned.add((host_id, "service_scan"))
            if host.get("scanned_processes") or any(k in host for k in ("proc_0", "proc_1", "tomcat", "daclsvc")):
                self.scanned.add((host_id, "process_scan"))
            if host.get("scanned_subnet"):
                self.scanned.add((host_id, "subnet_scan"))

    def reset(self):
        self.exploited_hosts.clear()
        self.scanned.clear()
        self.rooted_hosts.clear()
        self.known_access.clear()
        self.failed_exploit_attempts.clear()

    def _expected_access(self, service):
        for k in self.env.action_space.actions:
            if k.name.startswith("e_") and k.service == service:
                return k.access
        return 0  # fallback

    allowed_procs = {"proc_0", "proc_1", "tomcat", "daclsvc"}

    def parse_state_vector(self, state, env):
        host_states = {}
        num_hosts = len(env.network.hosts)
        state = np.array(state)

        try:
            state_matrix = state.reshape((num_hosts + 1, -1))
        except ValueError as e:
            raise ValueError(f"Cannot reshape obs of length {len(state)} to ({num_hosts + 1}, -1)") from e

        host_tensor = state_matrix[:-1]
        host_ids = list(env.network.hosts.keys())

        known_services = {"http", "ssh", "ftp"}
        known_processes = {"tomcat", "daclsvc"}
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
                if key in known_services or key in known_processes or key in known_os:
                    host_state[key] = value

            host_states[host_id] = host_state

        return host_states

    def select_action(self, state):
        host_states = self.parse_state_vector(state, self.env)

        # for debug
        # for host_id, host in host_states.items():
        #     if host["reachable"] and host["discovered"] and host["access"] >= 0:
        #         print(f"Exploit check on {host_id}: reachable={host['reachable']}, discovered={host['discovered']}, compromised={host['compromised']}, access={host['access']}")
        #         for service in self.known_exploits:
        #             already = (host_id, service) in self.exploited_hosts
        #             print(f" - Has service {service}, exploited? {already}")

        # Step 1: Exploit
        for host_id, host in host_states.items():
            if not host["reachable"] or not host["discovered"]:
                continue
            for service in self.known_exploits:
                if not (host.get(f"srv_{service}") or host.get(service)):
                    continue
                if (host_id, service) in self.exploited_hosts:
                    continue
                if self.failed_exploit_attempts[(host_id, service)] >= self.max_exploit_retries:
                    continue

                return self.action_map['exploit'](host_id, service)

        # Step 2: Privilege Escalation
        for i in range(self.env.action_space.n):
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

        # Step 3: Subnet Scan from privileged hosts
        for i in range(self.env.action_space.n):
            action = self.env.action_space.get_action(i)
            if action.name == "subnet_scan":
                key = (action.target, action.name)
                host = host_states.get(action.target)
                if host and host["access"] >= action.req_access and key not in self.scanned:
                    self.scanned.add(key)
                    return i

        # Step 4: OS/Process Scan
        for i in range(self.env.action_space.n):
            action = self.env.action_space.get_action(i)
            if action.name in {"os_scan", "process_scan"}:
                key = (action.target, action.name)
                host = host_states.get(action.target)
                if host and host["reachable"] and key not in self.scanned:
                    self.scanned.add(key)
                    return i

        # Step 5: Fallback Service Scan
        for i in range(self.env.action_space.n):
            action = self.env.action_space.get_action(i)
            if action.name == "service_scan":
                key = (action.target, action.name)
                host = host_states.get(action.target)
                if host and host["reachable"] and key not in self.scanned:
                    self.scanned.add(key)
                    return i

        # Step 6: Global fallback using scan_fn
        scan_idx = self.action_map['scan'](self.scanned)
        if scan_idx != self.action_map['noop']():
            return scan_idx

        return self.action_map['noop']()