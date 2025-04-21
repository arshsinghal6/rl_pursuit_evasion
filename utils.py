import torch
from torch_geometric.utils import dense_to_sparse

def obs_to_tensors(obs, pid, device='cpu'):
    """
    obs[pid] contains:
      - 'features': np.array shape (N, F)
      - 'adj':      np.array shape (N, N)
    Returns (x, edge_index) or (None, None) if no nodes.
    """
    info = obs.get(pid, None)
    if info is None: return None, None

    feats = info["features"]
    if feats.shape[0] == 0:
        return None, None

    adj = info["adj"]
    # to torch
    x = torch.tensor(feats, dtype=torch.float32, device=device)
    edge_index, _ = dense_to_sparse(torch.tensor(adj, dtype=torch.float32, device=device))
    return x, edge_index

def bresenham_line(x0, y0, x1, y1):
    """
    Returns a list of all (x, y) grid coordinates on the straight line
    from (x0, y0) to (x1, y1), inclusive, using Bresenham's algorithm.
    """
    points = []
    
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return points

def line_of_sight(grid, x0, y0, x1, y1):
    """
    Returns True if there's no obstacle in the direct line (via Bresenham)
    from (x0,y0) to (x1,y1). Otherwise False.
    grid: 2D numpy array, 1=obstacle, 0=free
    """
    path_cells = bresenham_line(x0, y0, x1, y1)
    for (cx, cy) in path_cells:
        if grid[cy, cx] == 1:
            # obstacle in the line => no line of sight
            return False
    return True

