import random
import numpy as np
import heapq

class CleaningAgent:
    def __init__(self, unique_id, start_position, end_position, initial_compliant_prob):
        self.unique_id = unique_id
        self.start_position = start_position
        self.end_position = end_position
        self.comp = False
        self.littering = False
        self.cleaning = False
        self.compliant_prob = initial_compliant_prob
        self.initial_compliant_prob = initial_compliant_prob
        self.trash_count = 1
        self.path = []
        self.reward = 0
        self.current_position = start_position
        self.litter_count = 0
        self.litter_history = []
        self.trip_id = 0
        self.observation = {}
        self.sanctioned = 0
        self.action = None  # To store the chosen action
        self.steps_since_start = 0
        self.was_sanctioned = False
        self.has_littered_this_trip = False
        self.steps_since_start = 0
        self.trip_durations = []  # To store durations of each trip
        self.first_bin_use_time = None # To store the timestep when the agent first uses the bin
        self.was_sanctioned = False
        self.is_dead = False

    def sigmoid(self, x, k, x0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def observe(self, env):
        # Collect observations of current state and previous actions
        w = env.observation_radius  # Observation radius
        agent_positions = env.get_agents_within_radius(self.current_position, w,exclude_agent_id=self.unique_id)

        # Get previous actions for agents within radius
        previous_actions_all = env.agent_actions_history.get(env.step_id - 1, {})
        previous_actions = {agent_id: action for agent_id, action in previous_actions_all.items() if agent_id in agent_positions}

        self.observation = {
            'current_state': {
                'agent_positions': agent_positions,
                'trash_locations': dict(env.trash)
            },
            'previous_actions': previous_actions
        }
        self.observation['sanctioned_agents'] = {}
        for other_agent_id in agent_positions:
            other_agent = env.get_agent_by_id(other_agent_id)
            self.observation['sanctioned_agents'][other_agent_id] = other_agent.was_sanctioned



    def choose_action(self, env):
        self.action = {}
   # If the agent was sanctioned, pick up trash and plan a path to trash bin
        if self.was_sanctioned:
            if env.get_trash_count(self.current_position) > 0:
                self.action['action'] = 'pick_up_trash'
                self.trash_count += 1
                self.was_sanctioned = False  # Reset the flag
                # Plan path to nearest trash bin
                self.path, _ = self.choose_path_via_bin(env)
                return  # Only perform one action per timestep
            else:
                # No trash at current position, agent moves towards nearest trash
                trash_positions = [pos for pos, count in env.trash.items() if count > 0]
                if trash_positions:
                    nearest_trash = min(trash_positions, key=lambda pos: self.heuristic(pos, self.current_position, env))
                    self.path = self.find_path(env, nearest_trash)
                    if self.path:
                        next_pos = self.path.pop(0)
                        self.action['move'] = next_pos
                    else:
                        # Cannot find path to trash, mark as dead
                        self.is_dead = True
                    return
                else:
                    # No trash to pick up, reset flag
                    self.was_sanctioned = False


        # Then, decide whether to litter or dispose
        if self.trash_count > 0:
            if self.current_position in env.trash_bins:
                self.action['action'] = 'dispose'
                self.trash_count = 0
                self.used_bin_this_trip = True  # Record that the agent used the bin
                return  # Only perform one action per timestep
            elif not self.comp and (self.current_position not in env.house_positions + env.office_position):
                self.action['action'] = 'litter'
                self.trash_count = 0
                self.has_littered_this_trip = True
                return  # Only perform one action per timestep
            else:
                # Cannot litter at home or office
                self.action['action'] = None  # No action
        
        replan_needed = False

        if self.path:
            next_pos = self.path[0]
            if not env.is_passable(next_pos):
                replan_needed = True
        else:
            replan_needed = True

        if replan_needed:
            # Re-plan the path
            self.path, _ = self.choose_path(env)
            if not self.path:
            # Cannot find a path to destination, mark agent as dead
                self.is_dead = True
                self.action['move'] = None
                return  # Agent will be removed in the environment

        if self.path:
            next_pos = self.path.pop(0)
            self.action['move'] = next_pos
        else:
            self.action['move'] = self.current_position  # Stay in place
        

        # Finally, decide whether to sanction
        if (env.method == "Decentralised" or (env.method == "Hybrid" and random.random() < env.b)):
            sanction_targets = []
            # if self.comp and self.current_position not in env.house_positions + env.office_position:
            if self.comp:
                for other_agent_id, other_action in self.observation['previous_actions'].items():
                    if other_action.get('action') == 'litter':
                        sanction_targets.append(other_agent_id)
            if sanction_targets:
                self.action['action'] = 'sanction'
                self.action['target_agent_ids'] = sanction_targets
                return  # Only perform one action per timestep





    def update_internal_state(self, env):
        # Reset compliance at start position
        if self.current_position == self.start_position:
            self.comp = random.random() < self.compliant_prob
            self.trash_count += 1  
            self.steps_since_start = 0 
            self.has_littered_this_trip = False


        if self.current_position == self.end_position:
            # if env.method == 'Centralised-end' and self.has_littered_this_trip:
            #     self.sanctioned += 1
            #     env.total_sanctions += 1

            # if (env.method == 'Hybrid' and random.random() >= env.b) and self.has_littered_this_trip:
            if env.method == 'Hybrid' and self.has_littered_this_trip:
                self.sanctioned += 1
                env.total_sanctions += 1
            
            self.compliant_prob = self.sigmoid(self.sanctioned, k=0.5, x0=0)
            self.trip_durations.append(self.steps_since_start)
            self.steps_since_start = 0  # Reset for the next trip
            self.used_bin_this_trip = False
            self.has_littered_this_trip = False 
            env.save_trip_steps(self.unique_id, self.trip_id)
            self.start_position = self.current_position
            self.end_position = self.choose_next_destination(env)
            self.trip_id += 1
            self.path = []
        else:
            self.steps_since_start += 1

        # if env.method == "Decentralised" or (env.method == "Hybrid" and random.random() < env.b):
        if env.method == "Decentralised" or env.method == "Hybrid":
        #     num_events = sum(
        #         1 for other_agent_id, other_action in self.observation['previous_actions'].items()
        #         if (
        #             other_action.get('action') == 'litter' and
        #             not self.observation['sanctioned_agents'].get(other_agent_id, False)
        #         )
        #     )

            # # Update self.sanctioned based on observed events
            # if num_events > 0:
            #     self.sanctioned -=  num_events # Only decrease once per timestep

            # Count events where the agent was sanctioned
            num_events2 = sum(
                1 for other_action in self.observation['previous_actions'].values()
                if (
                    other_action.get('action') == 'sanction' and
                    self.unique_id in other_action.get('target_agent_ids', [])
                )
            )

            # Update self.sanctioned based on sanctions received
            if num_events2 > 0:
                self.sanctioned += num_events2
            # self.was_sanctioned_this_trip = True


    def choose_next_destination(self, env):
        choices = [pos for pos in env.house_positions + env.office_position + env.park_positions if pos != self.start_position and pos != self.end_position]
        if not choices:
            return self.current_position  
        next_destination = random.choice(choices)
        return next_destination
    
    def find_nearest_trash_bin(self, env):
        return min(env.trash_bins, key=lambda bin: self.heuristic(bin, self.current_position, env))
    
    def choose_path_via_bin(self, env):
        via_bin = self.find_nearest_trash_bin(env)
        path_to_bin = self.find_path(env, via_bin)
        if not path_to_bin:
            return [], "blocked"
        return path_to_bin, "via trash bin"


    def choose_path(self, env):
        if self.comp:
            # Compliant agent plans path via the nearest trash bin
            via_bin = self.find_nearest_trash_bin(env)
            path_to_bin = self.find_path(env, via_bin)
            if not path_to_bin:
                # No path to bin; attempt direct path to destination
                direct_path = self.find_path(env, self.end_position)
                return direct_path, "direct"
            path_from_bin = self.find_path(env, self.end_position)
            if not path_from_bin:
                # No path from bin to destination
                return [], "blocked"
            path_from_bin = path_from_bin[1:]  # Remove starting position to avoid duplication
            return path_to_bin + path_from_bin, "via trash bin"
        else:
            # Non-compliant agent chooses the direct path
            direct_path = self.find_path(env, self.end_position)
            return direct_path, "direct"

    def find_path(self, env, goal):
        frontier = []
        heapq.heappush(frontier, (0, self.current_position))
        came_from = {self.current_position: None}
        cost_so_far = {self.current_position: 0}
        visited = set()
        

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                return self.reconstruct_path(came_from, goal)
            visited.add(current)
            for next in env.get_neighborhood(current):
                if not env.is_passable(next):
                    continue  # Skip impassable cells
                movement_cost = 1  
                new_cost = cost_so_far[current] + movement_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    # priority = new_cost + self.heuristic(goal, next, env)
                    priority = new_cost + self.heuristic(goal, next, env) + random.uniform(0, 0.1)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        # If we reach here, no path was found
        return []


    def reconstruct_path(self, came_from, goal):
        current = goal
        path = []
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def heuristic(self, a, b, env):
        # Direct path heuristic
        direct_distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
        return direct_distance
    