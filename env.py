import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import random
from collections import defaultdict, deque

import torch
from models import GCN_QNetwork
from utils import *
from evader_potential_field import *


class NodeInfo:
    def __init__ (self, x, y, times_visited, last_visit_time, visited_by, is_landmark = True):
        self.x = x
        self.y = y
        self.times_visited = times_visited
        self.last_visit_time = last_visit_time
        self.visited_by = visited_by
        self.neighbors = set()
        self.is_landmark = is_landmark
        self.evader_nearby_time = -1


def a_star_grid(grid, start, goal):
    """
    A* on the grid from 'start' to 'goal'.
    grid: 2D numpy where:
        - 1 = obstacle (not passable)
        - 0 = free
        - 2 = discouraged (soft penalty)
    start, goal: (x, y)

    Returns: list of (x, y) path, including start and goal
             or None if no path
    """
    w, h = grid.shape[1], grid.shape[0]

    def neighbors(x, y):
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny_ < h and grid[ny_, nx_] != 1:
                yield nx_, ny_

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

    open_set = [(heuristic(start, goal), 0, start, None)]
    visited = set()
    came_from = {}

    while open_set:
        _, cost, current, parent = open_set.pop(0)
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent

        if current == goal:
            # Reconstruct path
            path = []
            c = current
            while c is not None:
                path.append(c)
                c = came_from[c]
            return path[::-1]

        for nxt in neighbors(*current):
            if nxt in visited:
                continue
            nx, ny = nxt
            step_cost = 1
            if grid[ny, nx] == 2:
                step_cost += 10  # apply penalty
            new_cost = cost + step_cost
            f = new_cost + heuristic(nxt, goal)
            open_set.append((f, new_cost, nxt, current))

        open_set.sort(key=lambda x: x[0])  # sort by f

    return None


def dijkstra_in_topo_graph(graph: nx.Graph, start_node, goal_node):
    """
    A simple shortest-path in the topological graph from start_node to goal_node.
    Returns a list of nodes [start_node, ..., goal_node] if path exists, else None.
    """
    try:
        path = nx.shortest_path(graph, source=start_node, target=goal_node, weight=None)
        return path
    except nx.NetworkXNoPath:
        return None


def load_grid_from_image(image_path, width=80, height=50):
    img = Image.open(image_path).convert("L")
    img = img.resize((width, height))
    grid = np.array(img)
    grid = (grid < 128).astype(int)  # 1 = obstacle, 0 = free
    return grid


def generate_landmarks(grid, num_landmarks=45, min_distance=10, save_path='landmarks.txt'):
    height, width = grid.shape
    landmarks = {}
    while len(landmarks) < num_landmarks:
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        if grid[y, x] == 0:
            if all(np.linalg.norm(np.array([x, y]) - np.array([lx, ly])) >= min_distance
                   for (lx, ly) in landmarks):
                label = str(len(landmarks))
                landmarks[(x, y)] = label

    # Save landmarks to file
    with open(save_path, 'w') as f:
        for (x, y), label in landmarks.items():
            f.write(f"{label},{x},{y}\n")

    return landmarks

def load_landmarks(filepath):
    landmarks = {}
    reverse_landmarks = {}
    with open(filepath, 'r') as f:
        for line in f:
            label, x, y = line.strip().split(',')
            landmarks[(int(x), int(y))] = label
            reverse_landmarks[label] = (int(x), int(y))
    return landmarks, reverse_landmarks


class Evader:
    def __init__(self, env, start=(75, 45), goal=(10,45), fov=5, evader_strat='a_star'):
        self.env = env
        self.x, self.y = start
        self.goal = goal
        self.fov = fov
        self.path = a_star_grid(self.env.grid, (self.x, self.y), self.goal)[1:]
        self.prev_visible_pursuers = set()
        self.evader_strat = evader_strat
        self.prev_pos = None

        if self.evader_strat == 'potential_fields':
            grid = -1 * (self.env.grid - 1)
            
            # self.static_pot_field = compute_static_potential_field(grid, [(goal[1], goal[0])])
            # np.save('resources/static_pot_field.npy', self.static_pot_field)

            self.static_pot_field = np.load('resources/static_pot_field.npy')


    def can_see_pursuer(self):
        visible_pursuers = []
        for pursuer in self.env.pursuers:
            dx = pursuer.x - self.x
            dy = pursuer.y - self.y
            dist = np.sqrt(dx*dx + dy*dy)

            if dist <= self.fov:
                if line_of_sight(self.env.grid, self.x, self.y, pursuer.x, pursuer.y):
                    visible_pursuers.append(pursuer)

        return visible_pursuers

    def move(self):
        # If we've arrived, do nothing
        if (self.x, self.y) == self.goal:
            return True
        
        if self.evader_strat == 'a_star':
            visible_pursuers = self.can_see_pursuer()
            if visible_pursuers:
                grid = self.env.grid.copy()
                for p in visible_pursuers:
                    # if p in self.prev_visible_pursuers:
                    #     continue
                    avoid_radius = 5
                    Y, X = np.ogrid[:grid.shape[0], :grid.shape[1]]
                    mask = (X - p.x)**2 + (Y - p.y)**2 <= avoid_radius**2
                    grid[(mask) & (grid == 0)] = 2

                self.path = a_star_grid(grid, (self.x, self.y), self.goal)
                if self.path:
                    self.path.pop(0)
                
                self.prev_visible_pursuers = visible_pursuers
            elif not self.path:
                self.path = a_star_grid(self.env.grid, (self.x, self.y), self.goal)
                if self.path:
                    self.path.pop(0) 
    
            if self.path:
                # Move one step
                self.prev_pos = (self.x, self.y)
                self.x, self.y = self.path.pop(0)
        
        elif self.evader_strat == 'potential_fields':
            pursuer_pos = [(p.y, p.x) for p in self.env.pursuers]
            grid = -1 * (self.env.grid - 1)
            last_pos = (self.x, self.y)
            if not self.prev_pos:
                self.prev_pos = last_pos
            self.y, self.x = evader_next_cell(grid, 
                                              self.static_pot_field,
                                              (self.prev_pos[1], self.prev_pos[0]),   
                                              (self.y, self.x),
                                              pursuer_pos,
                                              [(self.goal[1], self.goal[0])])
            self.prev_pos = last_pos
            

        return False


class Pursuer:
    """
    Each pursuer has:
      - A topological graph (nodes = (x, y), data = NodeInfo)
      - Partial arrival logic
      - We do a two-layer approach:
         1) Dijkstra in the graph from current topo node -> final subgoal node
         2) For each edge in that topological route, run A* in the grid, collect partial paths
         3) The final path in the grid is traveled one cell per step
    """
    def __init__(self, env, pursuer_id, start=(10,45), fov=3):
        self.env = env
        self.id = pursuer_id
        self.x, self.y = start
        self.fov = fov
        self.last_pos = None

        # Tracking for episode
        self.prev_obs = None
        self.prev_action = None
        self.done = False

        self.graph = nx.Graph()
        self.visited_nodes = set()
        # Create initial node
        node_data = NodeInfo(x=self.x, y=self.y,
                             times_visited=1,
                             last_visit_time=0.0,
                             visited_by=self.id)
        self.env.nodes[(self.x, self.y)] = node_data
        self.graph.add_node((self.x, self.y))

        self.visited_nodes.add((self.x, self.y))

        # For partial arrival
        self.is_moving = False
        self.current_path = []
        self.node_route = []
        self.path_index = 0
        self.node_path_index = 0
        self.target_coord = None
        self.prev_node_in_path = (self.x, self.y)
        self.next_node_in_path = None

        self.found_evader = False

        # For intrinsic reward
        self.seen_grid = np.zeros(self.env.grid.shape, dtype=int)
        self.new_nodes_visited = 0
        self.prev_area_seen = 0
        self.total_visit_time = 0
        self.total_num_visits = 0
    
    def update_seen_grid(self):
        Y, X = np.ogrid[:self.seen_grid.shape[0], :self.seen_grid.shape[1]]
        mask = (X - self.x)**2 + (Y - self.y)**2 <= self.fov**2
        self.seen_grid[mask] = 1
    
    def add_neighbors(self, coord, neighbors):
        nodes = self.env.nodes
        for neighbor in neighbors:
            if not neighbor:
                continue
            self.graph.add_edge(coord, neighbor)
            nodes[coord].neighbors.add(neighbor)
            if not nodes[neighbor].is_landmark:
                continue
            for p in self.env.pursuers:
                if coord in p.visited_nodes:
                    p.graph.add_edge(coord, neighbor)

    def node_visit(self, coord):
        node = self.env.nodes[coord]
        # Intrinsic Reward 
        self.new_nodes_visited += 1
        self.total_visit_time += self.env.step_count - node.last_visit_time
        self.total_num_visits += node.times_visited

        self.visited_nodes.add(coord)
        node.times_visited += 1
        node.last_visit_time = self.env.step_count
        node.visited_by = self.id
        for neighbor in node.neighbors:
            if self.env.nodes[neighbor].is_landmark:
                self.graph.add_edge(coord, neighbor)
    
    def generate_new_node(self):
        best_coord = None
        best_unseen_count = -1

        x, y = (self.x, self.y)
        r = self.fov
        h, w = self.seen_grid.shape

        # Candidate nodes exactly FOV radius away (circle perimeter)
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = x + dx, y + dy

                if not line_of_sight(self.env.grid, self.x, self.y, nx, ny):
                    continue 

                # Check exact radius, bounds, and free space in environment
                if (dx**2 + dy**2 >= (0.9 * r)**2 and
                    dx**2 + dy**2 <= r**2 and
                    0 <= nx < w and 0 <= ny < h and
                    self.env.grid[ny, nx] == 0):

                    # Count unseen cells within FOV of candidate node
                    Y, X = np.ogrid[:h, :w]
                    mask = (X - nx)**2 + (Y - ny)**2 <= r**2
                    unseen_count = np.sum((self.seen_grid == 0) & mask)

                    # Keep the candidate with the most unseen cells
                    if unseen_count > best_unseen_count:
                        best_unseen_count = unseen_count
                        best_coord = (nx, ny)

        new_node = NodeInfo(best_coord[0], best_coord[1], 
                            times_visited=0,
                            last_visit_time=0,
                            visited_by=-1,
                            is_landmark=False)
        
        new_node.neighbors.add((self.x, self.y))
        self.env.nodes[best_coord] = new_node
        self.graph.add_edge(best_coord, (self.x, self.y))


    def set_subgoal(self, final_node):
        """
        1) We find a path of topological nodes from our current topo node to 'final_node' using Dijkstra.
        2) Then for each pair of consecutive nodes, run A* in the grid to get the partial path.
        3) Concatenate those partial paths into self.current_path.
        4) We'll travel along that path one cell at a time in move_one_step().
        """

        start_node = (self.x, self.y)

        # 1) Dijkstra in topological graph
        node_route = dijkstra_in_topo_graph(self.graph, start_node, final_node)
        if not node_route:
            raise Exception('Graph not connected or missing node')
        
        if len(node_route) < 2:
            self.node_visit(start_node)
            self.is_moving = False
            self.current_path = []
            self.path_index = 0
            self.target_coord = None
            return

        # 2) Build a big cell path by running a_star_grid on each edge in node_route
        big_path = [node_route[0]]  # store coords
        for i in range(len(node_route)-1):
            src_node = node_route[i]
            dst_node = node_route[i+1]
            partial = a_star_grid(self.env.grid, src_node, dst_node)
            if not partial or len(partial) < 2:
                # can't get from src_node to dst_node in the grid
                big_path = []
                break

            big_path.pop()  # remove last to avoid duplication
            big_path += partial

        if len(big_path) < 2:
            self.is_moving = False
            self.current_path = []
            self.path_index = 0
            self.target_coord = None
            return

        # Now we have a big cell-by-cell route
        self.current_path = big_path
        self.node_route = node_route
        self.path_index = 0
        self.node_path_index = 0
        self.target_coord = final_node
        self.prev_node_in_path = start_node
        self.next_node_in_path = node_route[1]
        self.is_moving = True

    def move_one_step(self):
        """
        If is_moving, move one cell along current_path.
        Return True if we see the evader in that step.
        """
        if not self.is_moving:
            return False

        if self.path_index >= len(self.current_path) - 1:
            # Done traveling
            self.is_moving = False
            return False

        self.path_index += 1
        new_pos = self.current_path[self.path_index]
        self.x, self.y = new_pos
        self.update_seen_grid()

        if (self.x, self.y) == self.next_node_in_path:
            self.node_path_index += 1
            self.prev_node_in_path = self.next_node_in_path
            if self.node_path_index + 1 < len(self.node_route):
                self.next_node_in_path = self.node_route[self.node_path_index + 1]

            self.node_visit((self.x, self.y))

        if self.check_evader_in_fov():
            self.found_evader = True
            return True

        visible_landmarks = self.find_visible_landmarks()
        for landmark in visible_landmarks:
            if not self.graph.has_node(landmark):
                self.graph.add_node(landmark)
            self.add_neighbors(landmark, [self.prev_node_in_path, self.next_node_in_path])
        
        return False


    def move_towards_evader(self):
        """
        Move one cell towards the evader:
         - If there's any obstacle in the FOV square ahead, take one A*-step.
         - Otherwise take a greedy alignment step, avoiding loops.
        """
        grid = self.env.grid
        rows, cols = self.env.height, self.env.width

        if self.check_evader_in_fov():
            self.found_evader = True
            return True

        # 1) Compute ordinal direction toward evader
        dy = self.env.evader.y - self.y       # row difference
        dx = self.env.evader.x - self.x       # col difference
        dr_goal = int(np.sign(dy))
        dc_goal = int(np.sign(dx))

        # 2) Check obstacles in the FOV square ahead
        obstacle_ahead = False
        # f = self.fov
        f = 1
        for dy_off in range(-f, f+1):
            for dx_off in range(-f, f+1):
                # Only consider offsets whose dot with goal > 0 (forward half)
                if dy_off*dr_goal + dx_off*dc_goal <= 0:
                    continue
                r = self.y + dy_off
                c = self.x + dx_off
                if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0:
                    obstacle_ahead = True
                    break
            if obstacle_ahead:
                break

        # 3) If obstacle ahead, fallback to a single A* step
        if obstacle_ahead:
            path = a_star_grid(grid, (self.x, self.y),
                         (self.env.evader.x, self.env.evader.y))
            if path and len(path) > 1:
                next_c, next_r = path[1]
                self.last_pos = (self.x, self.y)
                self.y, self.x = next_r, next_c
                return

        # 4) Greedy alignment step
        best = None
        best_score = -float('inf')
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = self.y + dr
                nc = self.x + dc
                # Must be in-bounds and free (grid==0 is free)
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr, nc] != 0:
                    continue
                # Alignment score = dot((dr,dc),(dr_goal,dc_goal))
                score = dr_goal*dr + dc_goal*dc
                # Discourage stepping back
                if self.last_pos == (nc, nr):
                    score -= 0.5
                if score > best_score:
                    best_score = score
                    best = (nc, nr)

        if best:
            self.last_pos = (self.x, self.y)
            self.x, self.y = best



    def check_evader_in_fov(self):
        ex, ey = self.env.evader.x, self.env.evader.y
        dx = ex - self.x
        dy = ey - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist <= self.fov:
            if line_of_sight(self.env.grid, self.x, self.y, ex, ey):
                return True
        return False

    def chase_evader(self):
        ex, ey = self.env.evader.x, self.env.evader.y
        x, y = self.x, self.y
        dx = ex - self.x
        dy = ey - self.y
        dist = np.sqrt(dx*dx + dy*dy)

        neighbors = [(x+dx, y+dy) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]]
        valid_neighbors = [(nx, ny) for nx, ny in neighbors 
                        if 0 <= nx < self.env.width 
                        and 0 <= ny < self.env.height 
                        and self.env.grid[ny, nx] == 0
                        and (nx, ny) != (self.env.evader.x, self.env.evader.y)]

        if not valid_neighbors:
            return  # No valid moves available

        # Select neighbor closest to the evader
        next_move = min(valid_neighbors, key=lambda coord: (coord[0]-ex)**2 + (coord[1]-ey)**2)

        pursuer_positions = [(p.x, p.y) for p in self.env.pursuers]

        # Move pursuer
        if dist > 4 and tuple(next_move) not in pursuer_positions:
            self.x, self.y = next_move

        visible_landmarks = self.find_visible_landmarks()
        for landmark in visible_landmarks:
            self.env.nodes[landmark].evader_nearby_time = self.env.step_count


    def find_visible_landmarks(self):
        visible_landmarks = []
        for (lx, ly), label in self.env.landmarks.items():
            dx = lx - self.x
            dy = ly - self.y
            dist = np.sqrt(dx*dx + dy*dy)
            if dist <= self.fov:
                if line_of_sight(self.env.grid, self.x, self.y, lx, ly):
                    visible_landmarks.append((lx, ly))

        return visible_landmarks



class PursuitEvasionEnv:
    """
    PARTIAL ARRIVAL + 2-LAYER PATH PLANNING
      - Each environment step:
         a) We read actions (dict of pursuer_id -> topological node index).
         b) For pursuers not moving, we set_subgoal(...) which:
             - uses Dijkstra in the topological graph
             - then uses A* in the grid for each edge in that route
         c) We move each pursuer ONE cell along its current path
         d) If any sees evader, done
         e) Then evader moves one cell
         f) Check if evader reached goal
    """
    def __init__(self, image_path='resources/MRS_Map.png', num_pursuers=3,
                 pursuer_starts=[(40,45),(2,10),(75,5)],
                 evader_start=(75,45), evader_goal=(10,49), random_start=False, evader_strat="a_star", pursuer_strat="rl_agent"):
        # Grid
        self.grid = load_grid_from_image(image_path)
        self.height, self.width = self.grid.shape

        # Landmarks
        # self.landmarks = generate_landmarks(self.grid, num_landmarks=45, min_distance=7)
        self.landmarks, self.reverse_landmarks = load_landmarks('resources/landmarks.txt')
        self.nodes = dict()
        for landmark in self.landmarks:
            self.nodes[landmark] = NodeInfo(landmark[0], landmark[1],
                                            times_visited=0,
                                            last_visit_time=0,
                                            visited_by=-1)
            
        self.evader_starts = [10, 7, 12, 6, 43, 0, 32, 8, 35, 1, 25, 18, 16, 20]
        if random_start:
            rand_eidx = np.random.randint(0, len(self.evader_starts))
            evader_start = self.reverse_landmarks[str(self.evader_starts[rand_eidx])]

            # rand_pidxs = np.random.choice(45, size=3, replace=False)
            # pursuer_starts = []
            # for idx in rand_pidxs:
            #     pursuer_starts.append(self.reverse_landmarks[str(idx)])

        # Create pursuers
        self.pursuer_strat = pursuer_strat
        if pursuer_strat == "naive_patrol":
            # Hardcoded Patrol spots 
            self.patrol_landmarks = [
                [39, 33, 25, 31, 35, 8, 1, 3],
                [44, 37, 9, 36, 42, 11, 21, 17],
                [41, 19, 22, 10, 32, 26, 38]
            ]
            self.patrol_counter = [0, 0, 0]

        self.pursuers = []
        for i in range(num_pursuers):
            p = Pursuer(env=self, pursuer_id=i, start=pursuer_starts[i], fov=10)

            if pursuer_strat == "naive_patrol": 
                for idx, landmark in enumerate(self.patrol_landmarks[i]):
                    loc_landmark = self.reverse_landmarks[str(landmark)]
                    p.graph.add_node(loc_landmark)
                    if idx == 0:
                        p.graph.add_edge((p.x, p.y), loc_landmark)
                    else:
                        prev_landmark = self.reverse_landmarks[str(self.patrol_landmarks[i][idx - 1])]
                        p.graph.add_edge(prev_landmark, loc_landmark)
                
                    if idx == len(self.patrol_landmarks[i]) - 1:
                        next_landmark = self.reverse_landmarks[str(self.patrol_landmarks[i][0])]
                        p.graph.add_edge(loc_landmark, next_landmark)
                    
            self.pursuers.append(p)

        # Evader
        self.evader_step_accum = 0
        self.evader_strat = evader_strat
        self.evader = Evader(self, start=evader_start, goal=evader_goal, evader_strat=evader_strat, fov=20)

        # RL bookkeeping
        self.curr_obs = None
        self.step_count = 0
        self.max_steps = 1600
        self.done = False

        self.catch_reward = 50.0

    def reset(self, random_start=True, evader_strat='a_star', pursuer_strat='rl_agent'):
        self.__init__(random_start=random_start, evader_strat=evader_strat, pursuer_strat=pursuer_strat)
        return self._get_observation()

    def display(self):
        plt.ion()  # Turn on interactive mode (run once)

        num_pursuers = len(self.pursuers)
        total_cols = num_pursuers
        total_rows = 3  # main env, graphs, seen grids

        if not hasattr(self, 'fig'):
            self.fig, self.axs = plt.subplots(
                total_rows, total_cols,
                figsize=(5 * total_cols, 5 * total_rows)
            )

            # --- Main Environment Plot (spans all columns) ---
            self.main_ax = plt.subplot2grid((total_rows, total_cols), (0, 0), colspan=1)
            self.main_img = self.main_ax.imshow(~self.grid, cmap="gray", origin="upper")
            for (x,y), label in self.landmarks.items():
                self.main_ax.text(x, y, label, color="red", fontsize=12, ha='center', va='center')

            self.pursuer_scat = self.main_ax.scatter(
                [p.x for p in self.pursuers], [p.y for p in self.pursuers],
                color="blue", s=100
            )
            self.evader_scat = self.main_ax.scatter(
                self.evader.x, self.evader.y, color="green", s=100
            )
            self.main_ax.set_xticks([])
            self.main_ax.set_yticks([])
            self.main_ax.set_title("Main Environment")

            # --- Pursuer Graphs ---
            self.graph_axes = []
            for i, p in enumerate(self.pursuers):
                ax = self.axs[1, i]
                ax.set_title(f"Pursuer {i} Graph")
                self.graph_axes.append(ax)

            # --- Pursuer Seen Grids ---
            self.seen_axes = []
            self.seen_imgs = []
            for i, p in enumerate(self.pursuers):
                ax = self.axs[2, i]
                ax.set_title(f"Pursuer {i} Seen Grid")
                img = ax.imshow(p.seen_grid, cmap="gray", origin="upper", vmin=0, vmax=1)
                self.seen_axes.append(ax)
                self.seen_imgs.append(img)

            if self.evader_strat == 'potential_fields':
                # --- Combined Potential Field ---
                self.pot_ax = plt.subplot2grid((total_rows, total_cols), (0, 1), colspan=1)
                # compute initial dynamic potential
                pursuer_positions = [(int(p.y), int(p.x)) for p in self.pursuers]
                dyn = self.evader.static_pot_field + avoid_pursuers_potential_field(
                    self.evader.static_pot_field.shape,
                    pursuer_positions,
                    repulsive_scale=70.0,
                    influence_radius=0.2
                )
                self.pot_img = self.pot_ax.imshow(dyn, cmap="hot", origin="upper")
                self.pot_ax.set_title("Combined Potential Field")
                self.pot_ax.set_xticks([])
                self.pot_ax.set_yticks([])

            plt.tight_layout()
        else:
            # --- Update main environment positions ---
            self.pursuer_scat.set_offsets([[p.x, p.y] for p in self.pursuers])
            self.evader_scat.set_offsets([self.evader.x, self.evader.y])

            # --- Update pursuer graphs ---
            for i, p in enumerate(self.pursuers):
                ax = self.graph_axes[i]
                ax.clear()
                ax.set_title(f"Pursuer {i} Graph")
                nodes = np.array(list(p.graph.nodes))
                if len(nodes) > 0:
                    ax.scatter(nodes[:,0], nodes[:,1], color="orange", s=50, zorder=2)
                    edges = p.graph.edges()
                    for (n1, n2) in edges:
                        x_vals = [n1[0], n2[0]]
                        y_vals = [n1[1], n2[1]]
                        ax.plot(x_vals, y_vals, color='black', linewidth=0.8, zorder=1)
                ax.set_xlim(0, self.width)
                ax.set_ylim(self.height, 0)
                ax.set_xticks([])
                ax.set_yticks([])

            # --- Update pursuer seen grids ---
            for i, p in enumerate(self.pursuers):
                self.seen_imgs[i].set_data(p.seen_grid)

            if self.evader_strat == 'potential_fields':
                # --- Update combined potential field ---
                pursuer_positions = [(int(p.y), int(p.x)) for p in self.pursuers]
                dyn = self.evader.static_pot_field + avoid_pursuers_potential_field(
                    self.evader.static_pot_field.shape,
                    pursuer_positions,
                    repulsive_scale=70.0,
                    influence_radius=0.2
                )
                self.pot_img.set_data(dyn)

        plt.draw()
        plt.pause(0.01)


    def _get_observation(self):
        """
        For each pursuer i:
          - build adjacency
          - build features [x,y,times_visited,last_visit_time,visited_by,uncertainty,degree]
          - note if is_moving, current_pos, etc.
        """
        obs = {}
        for i, p in enumerate(self.pursuers):
            nodes = list(p.graph.nodes())
            node_to_idx = { n: idx for idx, n in enumerate(nodes) }
            adj = nx.to_numpy_array(p.graph, nodelist=nodes)

            features = []
            for n in nodes:
                node_data = self.nodes[n]
                unvisited_by_pursuer = 0 if n in p.visited_nodes else 1
                visited_by_someone = 1 if node_data.visited_by != -1 else 0
                evader_nearby_time = 10000 if node_data.evader_nearby_time == -1 else self.step_count - node_data.evader_nearby_time
                deg = p.graph.degree(n)
                features.append([
                    node_data.x,
                    node_data.y,
                    ((p.x - node_data.x)**2 + (p.y - node_data.y)**2)**0.5,
                    node_data.times_visited,
                    self.step_count - node_data.last_visit_time,
                    evader_nearby_time,
                    float(unvisited_by_pursuer),
                    float(visited_by_someone),
                    float(deg), 
                    self.evader.goal[0], 
                    self.evader.goal[1]
                ])

            obs[i] = {
                "nodes": nodes,
                "adj": adj,
                "features": np.array(features, dtype=np.float32),
                "current_pos": (p.x, p.y),
                "is_moving": p.is_moving,
                "target_coord": p.target_coord
            }

        self.curr_obs = obs
        return obs

    def step(self, actions, evader_speed=1):
        self.step_count += 1
        transitions = []

        all_done = True
        for p in self.pursuers:              
            if ((not p.is_moving and p.prev_obs) or p.found_evader) and not p.done:
                reward = 0

                ex, ey = self.evader.x, self.evader.y
                dist_to_evader = ((p.x - ex)**2 + (p.y - ey)**2)**0.5

                total_seen = np.sum(p.seen_grid)
                total_grid_area = self.height * self.width
                new_area_seen = total_seen - p.prev_area_seen
                p.prev_area_seen = total_seen

                # Avoid division by zero
                if p.new_nodes_visited == 0:
                    avg_visit_time = 0
                    avg_num_visits = 1  # neutral penalty
                else:
                    avg_visit_time = p.total_visit_time / p.new_nodes_visited
                    avg_num_visits = p.total_num_visits / p.new_nodes_visited

                # Reset counters
                p.total_visit_time = 0
                p.total_num_visits = 0
                p.new_nodes_visited = 0

                # Calculate terms
                penalty_visits = (avg_num_visits - 1) / np.sqrt(avg_num_visits + 1e-5) if avg_num_visits != 0 else 0
                reward_area = new_area_seen / total_grid_area
                reward_staleness = np.tanh(avg_visit_time / 200.0)  # grows smoothly, saturates ~1

                # Weighted combination
                intrinsic_reward = (
                    40 * reward_area          # prioritize new area
                    - 0.5 * penalty_visits     # discourage over-visiting
                    + 0.1 * reward_staleness   # encourage stale revisits modestly
                )

                reward += intrinsic_reward

                # Penalize longer times
                # reward -= 0.25

                # Bonus for getting closer to evader
                max_dist = self.width + self.height
                # reward += 1.0 * ((max_dist - dist_to_evader)/max_dist)

                if p.found_evader:
                    reward += self.catch_reward
                    p.done = True
                
                transitions.append((p.id, p.prev_obs, p.prev_action, reward, p.done))

            if not p.is_moving:
                nodes = list(p.graph.nodes())
                node_idx = actions[p.id]
                if node_idx < len(nodes):
                    final_node = nodes[node_idx]
                    p.set_subgoal(final_node)
                    p.prev_obs = self.curr_obs
                    p.prev_action = node_idx

                if all(node in p.visited_nodes for node in nodes):
                    p.generate_new_node()
            
            if p.found_evader:
                p.chase_evader()
            elif self.pursuer_strat == "naive_patrol" and any([pursuer.found_evader for pursuer in self.pursuers]):
                p.move_towards_evader()
            else:
                p.move_one_step()
        
            all_done = all_done and p.done
        
        self.evader_step_accum += evader_speed
        if not all_done and self.evader_step_accum >= 1:
            self.evader_step_accum -= 1.0
            self.evader.move()
            if (self.evader.x, self.evader.y) == self.evader.goal:
                all_done = True

        if self.step_count >= self.max_steps:
            all_done = True

        obs = self._get_observation()

        return obs, transitions, all_done


if __name__ == "__main__":
    model_file = './checkpoints/policy_ep200.pt'
    policy_net = GCN_QNetwork(in_features=11, hidden_dim=64)
    state_dict = torch.load(model_file)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    env = PursuitEvasionEnv(random_start=True, evader_strat="potential_fields")
    obs = env.reset(random_start=True, evader_strat="potential_fields")
    for step in range(1600):
        actions = dict()
        num_unvisited_nodes = dict()
        for p in env.pursuers:
            num_unvisited_nodes[p.id] = len(list(p.graph.nodes())) - len(p.visited_nodes)
            if not p.is_moving:
                # num_options = len(list(p.graph.nodes()))
                # rand_choice = np.random.randint(0, num_options)
                # rand_actions[p.id] = rand_choice
                x, edge_index = obs_to_tensors(obs, p.id)
                with torch.no_grad():
                    q_vals = policy_net(x, edge_index)
                    actions[p.id] = int(q_vals.argmax().item())
        
        obs, transitions, done = env.step(actions, evader_speed=1)
 
        if done:
            print('Episode Done!')

        for transition in transitions:
            print(f'Step {step} || Pursuer {transition[0]} Reward: {transition[3]}')
        env.display()
