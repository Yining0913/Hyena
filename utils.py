import random

def initial_map(num_agents, width, height, seed=44):
    random.seed(seed)
    all_positions = [(x, y) for x in range(width) for y in range(height)]
    random.shuffle(all_positions)

    occupied_positions = set()

    # Determine the number of houses, offices, and parks based on the number of agents
    num_houses = max(5, num_agents // 10)
    num_offices = max(1, num_agents // 50)
    num_parks = max(2, num_agents // 20)

    # House positions
    house_positions = [all_positions.pop() for _ in range(num_houses)]
    occupied_positions.update(house_positions)

    # Office positions
    office_positions = [all_positions.pop() for _ in range(num_offices)]
    occupied_positions.update(office_positions)

    # Park positions
    park_positions = [all_positions.pop() for _ in range(num_parks)]
    occupied_positions.update(park_positions)

    # Trash bins
    trash_bins = []
    for location in house_positions + park_positions + office_positions:
        x, y = location
        possible_bin_locations = [
            (x-1, y), (x+1, y), (x, y-1), (x, y+1)
        ]
        possible_bin_locations = [
            pos for pos in possible_bin_locations
            if pos not in occupied_positions and 0 <= pos[0] < width and 0 <= pos[1] < height
        ]
        if possible_bin_locations:
            selected_bin = possible_bin_locations.pop(random.randint(0, len(possible_bin_locations) - 1))
            trash_bins.append(selected_bin)
            occupied_positions.add(selected_bin)

    return house_positions, office_positions, park_positions, trash_bins

# def initial_map(num_agents, width, height, seed=44):
#     random.seed(seed)
#     all_positions = [(x, y) for x in range(width) for y in range(height)]
#     random.shuffle(all_positions)

#     # Define the inner square region for the office
#     inner_start = (width - 4) // 2
#     inner_end = inner_start + 4

#     # House positions in the outer ring (excluding the inner 40x40 square)
#     house_candidates = [pos for pos in all_positions if pos[0] < inner_start or pos[0] >= inner_end or pos[1] < inner_start or pos[1] >= inner_end]
#     house_positions = [house_candidates.pop() for _ in range(5)]
#     occupied_positions = set(house_positions)

#     # Office positions in the inner 40x40 square
#     office_candidates = [pos for pos in all_positions if inner_start <= pos[0] < inner_end and inner_start <= pos[1] < inner_end and pos not in occupied_positions]
#     office_positions = [office_candidates.pop() for _ in range(1)]
#     occupied_positions.update(office_positions)

#     # Park positions in any unoccupied area
#     park_candidates = [pos for pos in all_positions if pos not in occupied_positions]
#     park_positions = [park_candidates.pop() for _ in range(2)]
#     occupied_positions.update(park_positions)

#     # Trash bins
#     trash_bins = []
#     for location in house_positions + park_positions + office_positions:
#         x, y = location
#         possible_bin_locations = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
#         possible_bin_locations = [pos for pos in possible_bin_locations if pos not in occupied_positions and 0 <= pos[0] < width and 0 <= pos[1] < height]
#         if possible_bin_locations:
#             selected_bin = possible_bin_locations.pop(random.randint(0, len(possible_bin_locations) - 1))
#             trash_bins.append(selected_bin)
#             occupied_positions.add(selected_bin)

#     return house_positions, office_positions, park_positions, trash_bins