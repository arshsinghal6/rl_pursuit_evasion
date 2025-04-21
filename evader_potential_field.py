import numpy as np
import heapq
import math

# ---------- A* on a binary grid ----------
def astar(grid: np.ndarray, start: tuple[int,int], goal: tuple[int,int]):
    """
    A* on a binary grid: 1 = free, 0 = obstacle.
    Returns the list of (r,c) from start to goal inclusive, or None if no path.
    """
    rows, cols = grid.shape
    def heuristic(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nbr = (current[0]+dr, current[1]+dc)
            r, c = nbr
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 1:
                tentative = cost + 1
                if tentative < g_score.get(nbr, float('inf')):
                    g_score[nbr] = tentative
                    came_from[nbr] = current
                    f = tentative + heuristic(nbr, goal)
                    heapq.heappush(open_set, (f, tentative, nbr))
    return None


# ---------- Compute Static Potential Field ----------
def potential_field(
    grid: np.ndarray,
    goals: list[tuple[int,int]],
    repulsive_scale: float = 0.75,
    attractive_scale: float = 1.0,
    influence_radius: float = 0.2,
    scale: int = 20
):
    """
    Computes the static potential components:
      - A* attractive to each goal
      - wall repulsion to open cells
      - wall raising at wall cells
    Returns (astar_attr, wall_rep, wall_raise).
    """
    rows, cols = grid.shape
    depth = len(goals)
    astar_attr = np.zeros((rows, cols, depth))
    wall_rep   = np.zeros((rows, cols))
    wall_raise = np.zeros((rows, cols))

    inf_r = int(influence_radius * 500 / scale)

    # Attractive: for free cells
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                for k, goal in enumerate(goals):
                    path = astar(grid, (r, c), goal)
                    dist = len(path) if path else rows*cols
                    astar_attr[r, c, k] = attractive_scale * dist

    # Repulsive + Raising
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:  # wall
                min_dist = float('inf')
                closest_pot = 0.0
                for i in range(max(0, r-inf_r), min(rows, r+inf_r+1)):
                    for j in range(max(0, c-inf_r), min(cols, c+inf_r+1)):
                        if grid[i, j] == 1:
                            d = math.hypot(r - i, c - j)
                            if 0 < d <= inf_r:
                                wall_rep[i, j] += repulsive_scale * (1 + math.cos(math.pi * d / inf_r)) / 2
                            if d < min_dist:
                                min_dist = d
                                closest_pot = np.min(astar_attr[i, j, :]) + 93
                wall_raise[r, c] = closest_pot

    return astar_attr, wall_rep, wall_raise


def compute_static_potential_field(
    grid: np.ndarray,
    goals: list[tuple[int,int]],
    **pf_kwargs
) -> np.ndarray:
    """
    Returns the combined static field:
      static = min_k(astar_attr[...,k]) + wall_rep + wall_raise
    """
    astar_attr, wall_rep, wall_raise = potential_field(grid, goals, **pf_kwargs)
    static = np.min(astar_attr, axis=2) + wall_rep + wall_raise
    return static


# ---------- Pursuer Repulsion Field ----------
def avoid_pursuers_potential_field(
    grid_shape: tuple[int,int],
    pursuers: list[tuple[int,int]],
    repulsive_scale: float = 70.0,
    influence_radius: float = 0.2,
):
    """
    Repulsive field from pursuers only.
    """
    rows, cols = grid_shape
    pot = np.zeros((rows, cols), dtype=float)
    R = int(influence_radius * rows)  # adjust units as needed
    for (pr, pc) in pursuers:
        for r in range(max(0, pr-R), min(rows, pr+R+1)):
            for c in range(max(0, pc-R), min(cols, pc+R+1)):
                d = math.hypot(r-pr, c-pc)
                if 0 < d <= R:
                    pot[r, c] += repulsive_scale * (1 + math.cos(math.pi * d / R)) / 2
    return pot


# ---------- Helper: pick lowest neighbor ----------
def get_next_position(dynamic_field: np.ndarray, evader_pos: tuple[int,int]):
    rows, cols = dynamic_field.shape
    r0, c0 = evader_pos
    best = evader_pos
    best_val = dynamic_field[r0, c0]
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        r, c = r0+dr, c0+dc
        if 0 <= r < rows and 0 <= c < cols:
            val = dynamic_field[r, c]
            if val < best_val:
                best_val, best = val, (r, c)
    return best


# ---------- Combined Evader Logic with Local Minima Avoidance ----------
def evader_next_cell(
    grid: np.ndarray,
    static_field: np.ndarray,
    evader_prev_pos: tuple[int, int],
    evader_pos: tuple[int,int],
    pursuers: list[tuple[int,int]],
    goals: list[tuple[int,int]],
    repulsive_scale: float = 70.0,
    influence_radius: float = 0.2,
    avoid_local_minima: bool = True,
    search_radius: int = 3
) -> tuple[int,int]:
    """
    Returns next (r,c) for the evader using dynamic potential + optional escape.
    """
    # 1) Static next
    static_next = get_next_position(static_field, evader_pos)
    # 2) Dynamic next
    pursuer_pot = avoid_pursuers_potential_field(
        static_field.shape, pursuers,
        repulsive_scale, influence_radius
    )
    dynamic_field = static_field + pursuer_pot
    dynamic_next = get_next_position(dynamic_field, evader_pos)

    # 3) Local minima avoidance
    if avoid_local_minima:
        dr = abs(static_next[0] - dynamic_next[0])
        dc = abs(static_next[1] - dynamic_next[1])
        wrong_dir = (dr > 1) or (dc > 1)
        if wrong_dir:
            # build modified grid marking pursuer neighborhoods as walls
            mod_grid = grid.copy()
            rows, cols = grid.shape
            for (pr, pc) in pursuers:
                rmin = max(0, pr-search_radius)
                rmax = min(rows, pr+search_radius+1)
                cmin = max(0, pc-search_radius)
                cmax = min(cols, pc+search_radius+1)
                for r in range(rmin, rmax):
                    for c in range(cmin, cmax):
                        if math.hypot(r-pr, c-pc) <= search_radius:
                            mod_grid[r, c] = 0
                
                mod_grid[evader_prev_pos[0], evader_prev_pos[1]] = 0
            # try A* to each goal
            best_path = None
            best_len = None
            for g in goals:
                path = astar(mod_grid, evader_pos, g)
                if path:
                    L = len(path)
                    if best_len is None or L < best_len:
                        best_len = L
                        best_path = path
            if best_path and len(best_path) > 1:
                return best_path[1]
    
    # Not moving
    if (dynamic_next[0], dynamic_next[1]) == evader_pos or (dynamic_next[0], dynamic_next[1]) == evader_prev_pos:
        # Build a strict obstacle map around pursuers (their cell + 4‐neighbors)
        mod_grid = grid.copy()
        rows, cols = grid.shape
        # for pr, pc in pursuers:
        #     # no_go = [(0,0),(1,0),(-1,0),(0,1),(0,-1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        #     no_go = [(0, 0)]
        #     for dr, dc in no_go:
        #         r, c = pr+dr, pc+dc
        #         if 0 <= r < rows and 0 <= c < cols:
        #             mod_grid[r, c] = 0
            
        #     # Don't go back to prev position
        #     mod_grid[evader_prev_pos[0], evader_prev_pos[1]] = 0

        for (pr, pc) in pursuers:
            rmin = max(0, pr-search_radius)
            rmax = min(rows, pr+search_radius+1)
            cmin = max(0, pc-search_radius)
            cmax = min(cols, pc+search_radius+1)
            for r in range(rmin, rmax):
                for c in range(cmin, cmax):
                    if math.hypot(r-pr, c-pc) <= search_radius:
                        mod_grid[r, c] = 0
            
            mod_grid[evader_prev_pos[0], evader_prev_pos[1]] = 0

        # Try plain A* on this modified grid to each goal
        best_path = None
        best_len = float('inf')
        for g in goals:
            path = astar(mod_grid, evader_pos, g)
            if path:
                L = len(path)
                if L < best_len:
                    best_len = L
                    best_path = path

        # If we found a detour, take one step along it
        if best_path and len(best_path) > 1:
            return best_path[1]
        
    # 4) otherwise go dynamic
    return dynamic_next
