import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt


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
        if dx == 0:
            stepX = 0
            tDeltaX = np.inf
            tDeltaY = dy
        elif dy == 0:
            stepY = 0
            tDeltaX = dx
            tDeltaY = np.inf
        else:
            tDeltaY = dy
            tDeltaZ = dz
        
    elif dy == 0:       # Plane XZ
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
        ## 2D case
        #if tmaxX < tmaxY:
        #    tmaxX += tDeltaX
        #    x += stepX
        #elif tmaxX > tmaxY:
        #    tmaxY += tDeltaY
        #    y += stepY
        #else:
        #    x += stepX
        #    y += stepY
        
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