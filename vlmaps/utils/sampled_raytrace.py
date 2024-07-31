import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


# Generate a random 10x10x10 boolean tensor using PyTorch
tensor_ = (torch.rand((10, 10, 10)) > 0.95).cuda()

# Sample a random 3D point within the range [0, 10]
point_ = torch.rand(3).cuda() * 10

# Define a random direction for the line
direction_ = torch.randn(3).cuda()
direction_ /= torch.norm(direction_)  # Normalize the direction

# Create parametric equations for the line
t_ = torch.linspace(-10, 10, 100).cuda()  # Parameter t for the line equation
line_points_ = point_.unsqueeze(1) + t_ * direction_.unsqueeze(1)

# Determine which voxels are crossed by the line
voxel_coords = torch.stack(torch.meshgrid(torch.arange(10), torch.arange(10), torch.arange(10)), dim=-1).reshape(-1, 3).float().cuda()

# Calculate the min and max bounds for each voxel
voxel_min = voxel_coords
voxel_max = voxel_coords + 1

# Check if line points are within any voxel's boundary
within_bounds = ((line_points.unsqueeze(2) >= voxel_min.T.unsqueeze(1)) & 
                 (line_points.unsqueeze(2) < voxel_max.T.unsqueeze(1))).all(dim=0)

# Aggregate across all line points to find which voxels are crossed
crossed_voxels = within_bounds.any(dim=0).view(tensor_.shape)

# Convert the tensor and crossed_voxels back to numpy for plotting
tensor_np = tensor_.cpu().numpy()
crossed_voxels_np = crossed_voxels.cpu().numpy()

# Create a figure and a 3D Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Prepare the voxel positions and values
x, y, z = np.indices((tensor_np.shape[0]+1, tensor_np.shape[1]+1, tensor_np.shape[2]+1))

# Plot the voxels
ax.voxels(x, y, z, tensor_np, edgecolor='k')

# Plot the crossed voxels in a different color
ax.voxels(x, y, z, crossed_voxels_np, facecolors='red', edgecolor='k')

# Plot the line
line_x = line_points[0].cpu().numpy()
line_y = line_points[1].cpu().numpy()
line_z = line_points[2].cpu().numpy()
ax.plot(line_x, line_y, line_z, color='r')
    
# Set axis limits to ensure they represent the [0, 10] interval
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])
    
# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
