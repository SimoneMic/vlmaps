import time
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt
import torch



class Point(NamedTuple):
    """
    a point in 2D with x, y coords
    """

    x: float
    y: float
    z: float

class Pixel(NamedTuple):
    """
    a pixel integer x, y coords
    """

    x: int
    y: int  
    z: int

def sampled_voxel_traversal(entry_pos, pc_grid, map):
    ray_source = torch.tensor(list(entry_pos), device='cuda')        # torch.Size([3])
    # Generate a random 10x10x10 boolean tensor using PyTorch
    #tensor = (torch.rand((10, 10, 10)) > 0.95).cuda()

    # Sample a random 3D point within the range [0, 10]
    #point = torch.rand(3).cuda() * 10

    # Define a random direction for the line
    #direction = torch.randn(3).cuda()
    direction = (pc_grid - ray_source).float()
    direction /= torch.norm(direction)  # Normalize the direction

    # Create parametric equations for the line TODO parameterize
    t = torch.linspace(0, 120, 500).cuda()  # Parameter t for the line equation: starts from 0 (camera) and goes up to 6m => 120 voxels, for 240 steps
    t = t.unsqueeze(0).unsqueeze(0)
    lines_points = ray_source.unsqueeze(1) + t * direction.unsqueeze(2)
    lines_points = lines_points.permute(1, 0, 2).reshape(3, -1)

    # Determine which voxels are crossed by the line
    #voxel_coords = torch.stack(torch.meshgrid(torch.arange(10), torch.arange(10), torch.arange(10)), dim=-1).reshape(-1, 3).float().cuda()

    # Calculate the min and max bounds for each voxel
    voxel_min = map 
    voxel_max = map + 1

    # Check if line points are within any voxel's boundary
    #within_bounds = ((lines_points.unsqueeze(2) >= voxel_min.T.unsqueeze(1)) & 
    #                 (lines_points.unsqueeze(2) < voxel_max.T.unsqueeze(1))).all(dim=0)
    batch_size = 2**12
    crossed_voxels = torch.zeros(len(map), dtype=torch.bool, device="cuda")
    for i in range(0, len(lines_points), batch_size):
        batch_points = lines_points[:, i:i + batch_size]
        within_bounds = ((batch_points.unsqueeze(2) >= voxel_min.T.unsqueeze(1)) & 
                         (batch_points.unsqueeze(2) < voxel_max.T.unsqueeze(1))).all(dim=0)

        crossed_voxels |= within_bounds.any(dim=0)

    return crossed_voxels

def raycast_map_torch(entry_pos, pc_grid, map, batch_size = 2**13, distance_threshold = 0.9):
    entry_pos = torch.tensor(list(entry_pos), device='cuda')        # torch.Size([3])
    #map = torch.tensor(list(map), device='cuda')                    # torch.size([1000000, 3])
    #occupied_map = torch.tensor(list(occupied_map), device='cuda')  # torch.size([1000, 1000, 50])

    # express the direction components in x, y, and z (the sign is the direction) for each ray
    directive_parameters = pc_grid - entry_pos   # torch.Size([9834, 3])    variable size N x 3
    # Line in space parametric equation 
    # (x,y,z) = (x1, y1, z1) + directive_parameters * t

    # For each point in the map, we look at the ones that are distant < 1 from each ray
    # Also, we consider the ones only in between the camera and the PC (pointcloud)
    #ray_mask_constrained = torch.zeros(map.shape[0], dtype=torch.bool, device="cuda")

    directive_parameters = directive_parameters.to(torch.float)
    #time_loop = time.time()
    #for ray, point in zip(directive_parameters, pc_grid):   # ray.shape = point.shape = torch.Size([3]) 
    #    # I find for each ray all the voxels that are close enpugh to the line
    #    # First, I need to find the plane that is parallel to that ray: for each point of the map
    #    d = - torch.sum(map * ray, 1)     #(ray[0] * point[0], ray[1] * point[1], ray[2] * point[2])  PLANE OFFSET
    #    # Find the position of the proection of "point" to ("ray" passing by the camera)
    #    # First find the parametric value of the perpendicular line to the ray passing by the map point
    #    t = - (d + torch.sum(point * ray)) / torch.sum(torch.square(ray))
    #    # And the projection for each map point to that ray
    #    projection_on_ray = point + (ray.to(torch.float).unsqueeze(-1) @ t.unsqueeze(0)).T
    #    # Find the distance
    #    dist = torch.norm(map - projection_on_ray, p=2, dim=1)
    #    #ray_mask = ((torch.abs((map - entry_pos)/ray) < (threshold)) & ((map > entry_pos) & (map < point))) # torch.Size([1974, 3])
    #    ray_mask = ((torch.abs(dist) < threshold) & (((map > entry_pos) & (map < point)).all(dim=1)))
    #    ray_mask_constrained = ray_mask_constrained | ray_mask
    #points_to_remove = map[ray_mask_constrained]
    ##occupied_map[ray_mask_constrained] = -1
    #print (f"Loop: {time.time() - time_loop}")
    dd = - (directive_parameters.to(torch.float16) @ map.T.to(torch.float16))
    tt = - (dd + (directive_parameters * pc_grid).sum(dim=1, keepdims=True)) / torch.sum(torch.square(directive_parameters), dim=1, keepdims=True)
    projections = pc_grid.unsqueeze(1) + torch.bmm(directive_parameters.unsqueeze(-1), tt.unsqueeze(1)).permute([0, 2, 1])
    distances = (map.unsqueeze(0) - projections).norm(dim=-1, p=2)

    # Note: all the voxels of the map have positive coordinates
    # We have to check if the camera point > or < the final point for selecting all the points in between
    crossed_voxels = torch.zeros(len(map), dtype=torch.bool, device="cuda")
    for i in range(0, len(distances), batch_size):
        distance_cond = torch.abs(distances[i:i + batch_size, :]) < distance_threshold
        bigger_than_camera = (map - entry_pos).unsqueeze(0).repeat(pc_grid.shape[0], 1, 1)[i:i + batch_size, :] * directive_parameters.unsqueeze(1).repeat([1, map.shape[0], 1])[i:i + batch_size, :] > 0.0
        smaller_than_pointcloud = (map.unsqueeze(0).repeat([pc_grid.shape[0], 1, 1])[i:i + batch_size, :] - pc_grid.unsqueeze(1).repeat([1, map.shape[0], 1])[i:i + batch_size, :]) * directive_parameters.unsqueeze(1).repeat([1, map.shape[0], 1])[i:i + batch_size, :] < 0.0
        mask = (distance_cond & (bigger_than_camera & smaller_than_pointcloud).all(dim=-1)).sum(dim=0).to(torch.bool)
        #mask = (
        # (torch.abs(distances) < threshold) &
        # (((map - entry_pos).unsqueeze(0).repeat(pc_grid.shape[0], 1, 1)[i:i + batch_size, :] * directive_parameters.unsqueeze(1).repeat([1, map.shape[0], 1]) > 0.0) &
        #  ((map.unsqueeze(0).repeat([pc_grid.shape[0], 1, 1]) - pc_grid.unsqueeze(1).repeat([1, map.shape[0], 1])) * directive_parameters.unsqueeze(1).repeat([1, map.shape[0], 1]) < 0.0)
        # ).all(dim=-1)
        #).sum(dim=0).to(torch.bool)
        crossed_voxels |= mask

    points_to_remove_matrix = map[mask]

    # TODO free GPU
    return points_to_remove_matrix

def raycast_map_torch_efficient(entry_pos, pc_grid, map, batch_size = 2**13, distance_threshold = 0.5, offset = 3.0):
    entry_pos = torch.tensor(entry_pos, device='cuda', dtype=torch.float32)
    directive_parameters = pc_grid - entry_pos
    directive_parameters = directive_parameters.to(torch.float32)

    # Initialize crossed voxels mask
    crossed_voxels = torch.zeros(map.shape[0], dtype=torch.bool, device="cuda")
    map = map.to(torch.float32)

    # Process in batches to reduce memory usage
    for pc_start in range(0, pc_grid.shape[0], batch_size):

        pc_end = min(pc_start + batch_size, pc_grid.shape[0])
        pc_batch = pc_grid[pc_start:pc_end]
        directive_batch = directive_parameters[pc_start:pc_end]

        # Calculate the lines equations
        dd = -(directive_batch @ map.T)
        tt = -(dd + (directive_batch * pc_batch).sum(dim=1, keepdims=True)) / torch.sum(directive_batch**2, dim=1, keepdims=True)

        # Calculate projections of each point of the camera pointcloud on each line
        projections = pc_batch.unsqueeze(1) + torch.bmm(directive_batch.unsqueeze(-1), tt.unsqueeze(1)).permute(0, 2, 1)

        # Calculate distances of each projection
        distances = (map.unsqueeze(0) - projections).norm(dim=-1, p=2)

        # Process map in smaller chunks to manage memory usage
        for map_start in range(0, map.shape[0], batch_size):
            map_end = min(map_start + batch_size, map.shape[0])
            distances_chunk = distances[:, map_start:map_end]

            # Distance condition of the points on the camera pointcloud
            distance_cond = distances_chunk < distance_threshold

            # Conditions for crossing
            bigger_than_camera = ((map[map_start:map_end] - entry_pos).unsqueeze(0) * directive_batch.unsqueeze(1)).sum(dim=-1) > 0.0
            smaller_than_pointcloud = ((map[map_start:map_end].unsqueeze(0) - (pc_batch - offset).unsqueeze(1)) * directive_batch.unsqueeze(1)).sum(dim=-1) < 0.0

            # Combined mask
            mask_chunk = (distance_cond & bigger_than_camera & smaller_than_pointcloud).any(dim=0)
            crossed_voxels[map_start:map_end] |= mask_chunk

    # Select the points to remove
    points_to_remove_matrix = map[crossed_voxels]

    return points_to_remove_matrix

#class Pixel:
#    def __init__(self, x, y, z):
#        self.x = x
#        self.y = y
#        self.z = z
#
#    def __repr__(self):
#        return f"Pixel(x={self.x}, y={self.y}, z={self.z})"

def traverse_pixels_torch(entry_pos, exit_pos):
    """
    Traverse through a 3D grid of voxels in parallel using PyTorch tensors.
    
    :param entry_pos: Tensor of shape (1, 3) with the entry position
    :param exit_pos: Tensor of shape (N, 3) with exit positions for N rays
    :return: List of lists of Pixels for each ray
    """
    #assert entry_pos.shape == (1, 3)
    #assert exit_pos.shape[1] == 3

    N = exit_pos.shape[0]

    # Broadcast entry position to match the shape of exit positions
    entry_pos = entry_pos.expand(N, 3)

    # Calculate direction vector and distances
    direction = exit_pos - entry_pos
    norm = torch.norm(direction, dim=1, keepdim=False)
    #direction = direction / norm

    dx, dy, dz = torch.abs(direction[:, 0]), torch.abs(direction[:, 1]), torch.abs(direction[:, 2])

    # Check where dx, dy or dz are == 0
    mask_dx0 = dx == 0
    mask_dy0 = dy == 0
    mask_dz0 = dz == 0

    # Init steps    (general case with dx!=dy!=dz!=0)
    stepX = torch.sign(direction[:, 0])
    stepY = torch.sign(direction[:, 1])
    stepZ = torch.sign(direction[:, 2])

    # Based on the masks set to 0 the relative steps
    stepX[mask_dx0] = 0
    stepY[mask_dy0] = 0
    stepZ[mask_dz0] = 0

    # Init tDelta   (general case with dx!=dy!=dz!=0 or when a single one is dx)
    tDeltaX = torch.where(dx == 0, torch.tensor(float('inf'), device='cuda'), norm/dx)
    tDeltaY = torch.where(dy == 0, torch.tensor(float('inf'), device='cuda'), norm/dy)
    tDeltaZ = torch.where(dz == 0, torch.tensor(float('inf'), device='cuda'), norm/dz)

    # init tmax     (general case with dx!=dy!=dz!=0)
    tmaxX = torch.where(direction[:, 0] > 0,
                        (torch.ceil(entry_pos[:, 0]) - entry_pos[:, 0]) * tDeltaX,
                        (entry_pos[:, 0] - torch.floor(entry_pos[:, 0])) * tDeltaX)

    tmaxY = torch.where(direction[:, 1] > 0,
                        (torch.ceil(entry_pos[:, 1]) - entry_pos[:, 1]) * tDeltaY,
                        (entry_pos[:, 1] - torch.floor(entry_pos[:, 1])) * tDeltaY)

    tmaxZ = torch.where(direction[:, 2] > 0,
                        (torch.ceil(entry_pos[:, 2]) - entry_pos[:, 2]) * tDeltaZ,
                        (entry_pos[:, 2] - torch.floor(entry_pos[:, 2])) * tDeltaZ)
    
    tmaxX[mask_dx0] = float('inf')
    tmaxY[mask_dy0] = float('inf')
    tmaxZ[mask_dz0] = float('inf')

    # How many iterations I have to do for each line
    iterations = torch.zeros(N, dtype=torch.int, device="cuda")
    iterations += dx.int() + dy.int() + dz.int()    #we have truncation here, is it correrct?

    # Initialize current positions
    current_pos = entry_pos.clone().int()

    # Initialize lists to store traversed pixels
    pixel_traversal = torch.empty([0,3], device="cuda")    #  torch.zeros((iterations.sum(), 3))

    # Initialize mask to keep track of rays that haven't reached their exit positions
    active_mask = torch.ones(N, dtype=torch.bool, device='cuda')
    for _ in range(torch.max(iterations)) :
        mask_x = (tmaxX < tmaxY) & (tmaxX < tmaxZ) & active_mask
        mask_y = (tmaxY <= tmaxX) & (tmaxY < tmaxZ) & active_mask
        mask_z = (tmaxZ <= tmaxX) & (tmaxZ <= tmaxY) & active_mask

        # TODO: do a mask for equal tmax pairs
        # mask_x_equals = ((tmaxX == tmaxY) | (tmaxX == tmaxZ)) & active_mask
        # mask_y_equals = ((tmaxY == tmaxX) | (tmaxY == tmaxZ)) & active_mask
        # mask_z_equals = ((tmaxZ == tmaxX) | (tmaxZ == tmaxY)) & active_mask

        tmaxX += mask_x.int() * tDeltaX
        tmaxY += mask_y.int() * tDeltaY
        tmaxZ += mask_z.int() * tDeltaZ

        current_pos[:, 0] += (mask_x).int() * stepX.int()
        current_pos[:, 1] += (mask_y).int() * stepY.int()
        current_pos[:, 2] += (mask_z).int() * stepZ.int()

        # Collect active current positions
        #active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        #active_positions = current_pos[active_indices]
        
        pixel_traversal = torch.cat((pixel_traversal, current_pos[torch.nonzero(active_mask, as_tuple=True)[0]]), dim=0)

        # Update the active mask
        active_mask = (abs(current_pos - exit_pos.int()) > 2.0).any(dim=1)
        
    return pixel_traversal


def traverse_pixels(entry_pos, exit_pos):
    """
    Ray equation describes a point t along its trajectory:
    
    point(t) = u + tv
    
    where u is the entry position and v is the direction of the ray.
    
    We can start the traverse towards the pixel that is closest
    _in units of t_ from starting point. This is done by adding a pixel
    size in that direction also in units of t. Repeat until reaching
    the exit pixel.
    
    Voxels have unit size, ie units are in pixel size
    such that stepX and stepY are -1 or 1
    
    From the entry and exit positions one can determine v
    ie v = norm(exit_pos - entry_position)
    
    :param entry_pos: incidence position, in pixel units, or u above
    :param exit_pos: exit position, in pixel units
    :return: list of pixels
    """
    dx = abs(exit_pos.x - entry_pos.x)
    dy = abs(exit_pos.y - entry_pos.y)
    dz = abs(exit_pos.z - entry_pos.z)

    if (dx==dy==dz==0):
        print("Entry point and exit point are the same, returning")
        return

    start_pixel = Pixel(int(entry_pos.x), int(entry_pos.y), int(entry_pos.z))
    end_pixel = Pixel(int(exit_pos.x), int(exit_pos.y), int(exit_pos.z))    
    
    x, y, z = start_pixel.x, start_pixel.y, start_pixel.z
    
    n = 0
  
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    # TODO: check for case with norm == 0
    
    # set the travelling direction quadrant based on dx, dy signs
    # Expand it in 3D -> for each plane, let's look at planes:
    if dx == 0:         # Plane YZ
        stepX = 0
        tDeltaX = np.inf
        tmaxX = np.inf
        
        # Check for travel on Y, Z axes:
        if dz == 0:
            stepZ = 0
            tDeltaZ = np.inf
            tDeltaY = dy
        elif dy == 0:
            stepY = 0
            tDeltaX = dx
            tDeltaY = np.inf
        else:
            tDeltaY = dy
            tDeltaZ = dz
        
    if dy == 0:       # Plane XZ
        stepY = 0
        tDeltaY = np.inf
        tmaxY = np.inf

        # Check for travel on X, Z axes:
        if dx == 0:
            stepX = 0
            tDeltaX = np.inf
            tDeltaZ = dy
        elif dz == 0:
            stepZ = 0
            tDeltaX = dx
            tDeltaZ = np.inf
        else:
            tDeltaX = dx
            tDeltaZ = dz
    
    elif dz == 0:   # Plane XY
        stepZ = 0
        tDeltaZ = np.inf
        tmaxZ = np.inf

        # Check for travel on X, Y axes:
        if dx == 0:
            stepX = 0
            tDeltaX = np.inf
            tDeltaY = dy
        elif dy == 0:
            stepY = 0
            tDeltaX = dx
            tDeltaY = np.inf
        else:
            tDeltaX = dx
            tDeltaY = dy
    # 3D case
    else:    
        # deltas are just the inverse component of t
        tDeltaX = norm/dx
        tDeltaY = norm/dy
        tDeltaZ = norm/dz
        
    
    # X direction
    if exit_pos.x < entry_pos.x:
        stepX = -1
        tmaxX = abs((entry_pos.x - np.floor(entry_pos.x)) * tDeltaX)
        n += x - end_pixel.x
    elif exit_pos.x > entry_pos.x:
        stepX = 1
        tmaxX = abs((np.ceil(entry_pos.x) - entry_pos.x) * tDeltaX)
        n += end_pixel.x - x

    # Y direction 
    if exit_pos.y < entry_pos.y:
        stepY = -1
        tmaxY = abs((entry_pos.y - np.floor(entry_pos.y)) * tDeltaY)
        n += y - end_pixel.y
    elif exit_pos.y > entry_pos.y:
        stepY = 1
        tmaxY = abs((np.ceil(entry_pos.y) - entry_pos.y) * tDeltaY)
        n += end_pixel.y - y
    
    # Z direction 
    if exit_pos.z < entry_pos.z:
        stepZ = -1
        tmaxZ = abs((entry_pos.z - np.floor(entry_pos.z)) * tDeltaZ)
        n += z - end_pixel.z
    elif exit_pos.z > entry_pos.z:
        stepZ = 1
        tmaxZ = abs((np.ceil(entry_pos.z) - entry_pos.z) * tDeltaZ)
        n += end_pixel.z - z

    # list of pixels travelled
    line = [Pixel(x, y, z)]     #Starts from the camera pixel

    for _ in range(n):
        # 3D case
        if tmaxX < tmaxY:
            if tmaxX < tmaxZ:
                x += stepX
                tmaxX += tDeltaX
            else:
                z += stepZ
                tmaxZ += tDeltaZ
        else:
            if tmaxY < tmaxZ:
                y += stepY
                tmaxY += tDeltaY
            else:
                z += stepZ
                tmaxZ += tDeltaZ
        
        line.append(Pixel(x,y,z))
        
    return line
  
if __name__ == '__main__':
    start = Point(np.random.uniform(-7,7), np.random.uniform(-7,7), np.random.uniform(-7,7))
    end = Point(np.random.uniform(-7,7), np.random.uniform(-7,7), np.random.uniform(-7,7))
    print(f"starting from X: {start[0]} Y: {start[1]} Z: {start[2]} TO end X: {end[0]} Y: {end[1]} Z: {end[2]}")
    pixels = traverse_pixels(start, end)
    
    # plot the results
    data = np.zeros((8,8,8))
    x = [pixel.x for pixel in pixels]
    y = [pixel.y for pixel in pixels]
    z = [pixel.z for pixel in pixels]
    data[x, y, z] = 1
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")
    ax =  plt.figure().add_subplot(projection='3d')

    ax.scatter(x, y, z, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect("equal")
    ax.set_title(f"From X: {int(start[0])} Y: {int(start[1])} Z: {int(start[2])} TO X: {int(end[0])} Y: {int(end[1])} Z: {int(end[2])}")

    plt.show()