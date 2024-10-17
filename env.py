import random
import pygame
from collections import defaultdict
import pandas as pd

from agent import CleaningAgent

class GridWorld:
    def __init__(self, width, height, num_agents, method, b, initial_compliant_probs, house_positions, office_position, park_positions, trash_bins,num_run,t,observation_radius):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.agents = []
        self.trash = defaultdict(int)
        self.round_id = 0
        self.step_id = 0
        self.b = b
        self.cell_size = 30
        self.method = method
        self.num_run = num_run
        self.house_positions = house_positions
        self.office_position = office_position
        self.park_positions = park_positions
        self.trash_bins = trash_bins
        self.agent_actions_history = {}
        self.littering_agents = set()
        self.t=t
        self.observation_radius = observation_radius
        self.compliance_over_time = []
        self.cleanliness_over_time = []
        self.total_sanctions_over_time = []
        self.total_sanctions = 0
        self.dead_agents_count = 0

        for i in range(num_agents):
            start_position = self.house_positions[i % len(self.house_positions)]
            end_position = random.choice([pos for pos in self.office_position + self.park_positions + self.house_positions if pos != start_position])
            agent = CleaningAgent(i, start_position, end_position, initial_compliant_probs[i])
            self.agents.append(agent)

        self.clean_squares_record = []
        self.trip_steps = []
        # self.load_images()

    def get_agents_within_radius(self, position, radius, exclude_agent_id=None):
        agents_in_radius = {}
        x0, y0 = position
        for agent in self.agents:
            if agent.unique_id == exclude_agent_id:
                continue  # Skip the agent itself
            x1, y1 = agent.current_position
            distance = abs(x1 - x0) + abs(y1 - y0)  # Manhattan distance
            if distance <= radius:
                agents_in_radius[agent.unique_id] = agent.current_position
        return agents_in_radius


    def get_agent_by_id(self, agent_id):
        for agent in self.agents:
            if agent.unique_id == agent_id:
                return agent
        return None
    
  
    def get_average_compliance_over_time(self):
        return self.compliance_over_time

    def run_env(self, steps):
        pygame.init()
        cell_size = 30
        screen = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
        clock = pygame.time.Clock()

        agents_data = {agent.unique_id: {'steps': [], 'comp_probs': [], 'sanctions': []} for agent in self.agents}

        time_since_last_check = 0

        for step in range(steps):
            self.step_id = step

            # Step 1: Each agent observes the state
            for agent in self.agents:
                agent.observe(self)

            # Step 2: Each agent decides on an action
            for agent in self.agents:
                agent.choose_action(self)

            # Step 3: Collect all actions
            actions = {agent.unique_id: agent.action for agent in self.agents}

            # Record the actions
            self.agent_actions_history[step] = actions

            # Step 4: Apply all actions 
            self.apply_actions(actions)

            time_since_last_check += 1
            if self.method == 'Centralised-ts' and time_since_last_check == self.t:
                for agent_id in self.littering_agents:
                    agent = self.get_agent_by_id(agent_id)
                    agent.sanctioned += 1
                    # agent.compliant_prob = agent.sigmoid(agent.sanctioned, k=0.5, x0=0)
                    agent.has_littered_this_trip = False
                time_since_last_check = 0
                self.littering_agents.clear()

            if self.method == 'Centralised-end' or (self.method == 'Hybrid' and random.random() >= self.b) :
                for agent in self.agents:
                    if agent.has_littered_this_trip and agent.current_position == agent.end_position:
                        agent.sanctioned += 1
                        self.total_sanctions += 1
                        agent.has_littered_this_trip = False  # Reset flag

            # Step 6: Update internal state of agents
            for agent in self.agents:
                agent.update_internal_state(self)
                agents_data[agent.unique_id]['steps'].append(step)
                agents_data[agent.unique_id]['comp_probs'].append(agent.compliant_prob)
                agents_data[agent.unique_id]['sanctions'].append(agent.sanctioned)
            
            avg_compliance = sum(agent.compliant_prob for agent in self.agents) / len(self.agents)
            self.compliance_over_time.append(avg_compliance)
            self.clean_squares_record.append(self.count_clean_squares())
            # self.save_results()

            percentage_clean = self.compute_percentage_clean_cells()
            self.cleanliness_over_time.append(percentage_clean)
            self.total_sanctions_over_time.append(self.total_sanctions)

#             # Event handling for Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Rendering
            background = pygame.Surface(screen.get_size())
            background = background.convert()
            background.fill((255, 255, 255))
            screen.blit(background, (0, 0))

            for x in range(self.width):
                for y in range(self.height):
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, (200, 200, 200), rect, 1)

                    if (x, y) in self.house_positions:
                        screen.blit(self.house_img, rect)
                    elif (x, y) in self.office_position:
                        screen.blit(self.office_img, rect)
                    elif (x, y) in self.park_positions:
                        screen.blit(self.park_img, rect)
                    elif (x, y) in self.trash_bins:
                        screen.blit(self.trash_bin_img, rect)

                    trash_count = self.get_trash_count((x, y))
                    if trash_count > 0:
                        font = pygame.font.Font(None, 36)
                        text = font.render(str(trash_count), True, (0, 0, 0))
                        screen.blit(text, (x * cell_size, y * cell_size))

            for agent in self.agents:
                x, y = agent.current_position
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if agent.littering:
                    screen.blit(self.agent_littering_img, rect)
                # elif agent.punished:
                #     screen.blit(self.agent_punished_img, rect)
                elif agent.comp:
                    screen.blit(self.agent_bin_img, rect)
                else:
                    screen.blit(self.agent_normal_img, rect)

            pygame.display.flip()
            clock.tick(1000)


        all_agent_records = []
        for agent_id, data in agents_data.items():
            for step, comp_prob, sanction in zip(data['steps'], data['comp_probs'], data['sanctions']):
                all_agent_records.append({
                    'agent_id': agent_id,
                    'step': step,
                    'compliance_probability': comp_prob,
                    'sanction_level': sanction
                })

        agents_df = pd.DataFrame(all_agent_records)
       


        pygame.quit()

    def apply_actions(self, actions):
        # dead_agents = [agent for agent in self.agents if getattr(agent, 'is_dead', False)]
        # for agent in dead_agents:
        #     self.agents.remove(agent)
        #     self.num_agents -= 1
        #     self.dead_agents_count += 1  # Initialize this in __init__
        #     # Optionally, log or record the agent's death
        #     print(f"Agent {agent.unique_id} has been removed from the simulation.")

    # Also remove their actions from the actions dictionary
        # for agent in dead_agents:
        #     actions.pop(agent.unique_id, None)

        for agent in self.agents:
            agent.was_sanctioned = False
            # Update agent positions
        for agent_id, action in actions.items():
            move_pos = action.get('move')

            if move_pos is None:
                move_pos = self.agents[agent_id].current_position  # Agent stays in place
            elif not self.is_passable(move_pos):
                # Cannot move into impassable cell, stay in place
                move_pos = self.agents[agent_id].current_position
            # Update agent position
            self.agents[agent_id].current_position = move_pos

        # First, collect proposed moves
        proposed_moves = {}
        destination_counts = defaultdict(list)
        for agent_id, action in actions.items():
            move_pos = action.get('move')
            if move_pos is None:
                move_pos = self.agents[agent_id].current_position  # Agent stays in place
            elif not self.is_passable(move_pos):
                # Cannot move into impassable cell, stay in place
                move_pos = self.agents[agent_id].current_position
            proposed_moves[agent_id] = move_pos
            destination_counts[move_pos].append(agent_id)

        # Resolve conflicts
        final_positions = {}
        for agent_id, move_pos in proposed_moves.items():
            if len(destination_counts[move_pos]) > 1:
                # Conflict: agents stay in place
                final_positions[agent_id] = self.agents[agent_id].current_position
            else:
                final_positions[agent_id] = move_pos

        # Update agent positions
        for agent_id, new_pos in final_positions.items():
            self.agents[agent_id].current_position = new_pos

        # Process actions at positions
        for agent_id, action in actions.items():
            action_type = action.get('action')
            if action_type == 'litter':
                position = self.agents[agent_id].current_position
                self.add_trash(position)
                if self.method == 'Centralised-ts':
                    self.littering_agents.add(agent_id)
                
                if position in self.trash_bins:
                    # Agent is littering in a bin
                    self.agents[agent_id].used_bin_this_trip = True

                    # Record the timestep if not already recorded
                    if self.agents[agent_id].first_bin_use_time is None:
                        self.agents[agent_id].first_bin_use_time = self.step_id  # Or 'step' variable
                else:
                    self.agents[agent_id].used_bin_this_trip = False
            
            elif action_type == 'dispose':
            # Agent disposes of trash properly at a trash bin
                pass  # No action needed

            elif action_type == 'pick_up_trash':
                position = self.agents[agent_id].current_position
                self.remove_trash(position)

            elif action_type == 'sanction':
                target_agent_ids = action.get('target_agent_ids', [])
                for target_agent_id in target_agent_ids:
                    if target_agent_id is not None:
                        self.agents[target_agent_id].was_sanctioned = True
                        self.total_sanctions += 1



    def load_images(self):
        self.house_img = pygame.image.load("icons/house.png")
        self.house_img = pygame.transform.scale(self.house_img, (self.cell_size, self.cell_size))
        self.office_img = pygame.image.load("icons/office.png")
        self.office_img = pygame.transform.scale(self.office_img, (self.cell_size, self.cell_size))
        self.park_img = pygame.image.load("icons/004-park.png")
        self.park_img = pygame.transform.scale(self.park_img, (self.cell_size, self.cell_size))
        self.trash_bin_img = pygame.image.load("icons/bin.png")
        self.trash_bin_img = pygame.transform.scale(self.trash_bin_img, (self.cell_size, self.cell_size))
        self.agent_littering_img = pygame.image.load("icons/redman.png")
        self.agent_littering_img = pygame.transform.scale(self.agent_littering_img, (self.cell_size, self.cell_size))
        self.agent_punished_img = pygame.image.load("icons/Sanction List.png")
        self.agent_punished_img = pygame.transform.scale(self.agent_punished_img, (self.cell_size, self.cell_size))
        self.agent_bin_img = pygame.image.load("icons/blueman.png")
        self.agent_bin_img = pygame.transform.scale(self.agent_bin_img, (self.cell_size, self.cell_size))
        self.agent_normal_img = pygame.image.load("icons/blackman.png")
        self.agent_normal_img = pygame.transform.scale(self.agent_normal_img, (self.cell_size, self.cell_size))

    def is_passable(self, position):
        trash_count = self.get_trash_count(position)
        if position in self.trash_bins:
            return True
        else:
            return trash_count <= 3


    def get_trash_count(self, position):
        return self.trash[position]

    def add_trash(self, position):
        self.trash[position] += 1

    def remove_trash(self, position):
        if self.trash[position] > 0:
            self.trash[position] -= 1

    def get_neighborhood(self, position):
        x, y = position
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        return [pos for pos in neighbors if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height]

    def get_all_agent_positions(self):
        return {agent.unique_id: agent.current_position for agent in self.agents}

    def count_clean_squares(self):
        clean_count = 0
        for x in range(self.width):
            for y in range(self.height):
                position = (x, y)
                if self.get_trash_count(position) == 0:
                    clean_count += 1
        return clean_count

    def save_trip_steps(self, agent_id, trip_id):
        self.trip_steps.append([self.method, self.step_id, agent_id, trip_id])
    

    def compute_percentage_clean_cells(self):
        total_cells = self.width * self.height
        clean_cells = self.count_clean_squares()
        percentage_clean = (clean_cells / total_cells) * 100
        return percentage_clean