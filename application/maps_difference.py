import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
import open3d as o3d
from vlmaps.utils.visualize_utils import (
    visualize_rgb_map_3d,
    compute_point_cloud_difference
)
import threading

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dirs = "~/vlmaps"
    vlmap_raytrace = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
    vlmap_std = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
    vlmap_std.load_map("/home/ergocub/vlmaps.h5df")     
    vlmap_raytrace.load_map("/home/ergocub/vlmaps_raytrace_tensorized.h5df")
    visualize_rgb_map_3d(vlmap_std.grid_pos, vlmap_std.grid_rgb, "vlmap_std")
    visualize_rgb_map_3d(vlmap_raytrace.grid_pos, vlmap_raytrace.grid_rgb, "vlmap_raytrace")
    diff = compute_point_cloud_difference(vlmap_std.grid_pos, vlmap_raytrace.grid_pos)
    o3d.visualization.draw_geometries_with_vertex_selection([diff])
    voxel_grid_map = o3d.geometry.VoxelGrid.create_from_point_cloud(diff, voxel_size= 1.0)   #TODO parameterize
    o3d.visualization.draw_geometries([voxel_grid_map])


if __name__ == "__main__":
    main()
