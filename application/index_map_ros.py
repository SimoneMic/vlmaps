import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

class VLMapIndexer(Node):
    def __init__(self, config: DictConfig):
        super().__init__('VLMap_indexer_node')
        data_dirs = "~/vlmaps"
        self.vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
        self.vlmap.load_map("/home/ergocub/vlmaps.h5df")
        self.vlmap._init_clip()
        self.pointcloud_pub = self.create_publisher(PointCloud2, "vlmap_query", 1)

        self.query_loop()

    def query_loop(self):
        cat = input("What is your interested category in this scene?")
        while cat != "exit":
            self.query(cat)
            cat = input("What is your interested category in this scene?")

    def query(self, cat):
        seg_mask = self.vlmap.index_map(cat, with_init_cat=False)
        self.vlmap.grid_rgb[:] = [0, 0, 255]
        self.vlmap.grid_rgb[seg_mask] = [255, 0, 0]
        mask = (self.vlmap.grid_pos > 0).all(axis=1)
        color = self.vlmap.grid_rgb[mask]
        points = self.vlmap.grid_pos[mask] * 0.05  #scale it to meters
        msg = self.xyzrgb_array_to_pointcloud2(points, color)
        self.pointcloud_pub.publish(msg)

    def xyzrgb_array_to_pointcloud2(self, points, colors, stamp=None, frame_id=None, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points and a synched array of color values.
        '''

        header = Header()
        header.frame_id = "vlmap"
        header.stamp = self.get_clock().now().to_msg()

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyzrgb')]
        nbytes = 6
        xyzrgb = np.array(np.hstack([points, colors/255]), dtype=np.float32)
        msg = PointCloud2(header=header, 
                          height = 1, 
                          width= points.shape[0], 
                          fields=fields, 
                          is_dense= False, 
                          is_bigedian=False, 
                          point_step=(itemsize * nbytes), 
                          row_step = (itemsize * nbytes * points.shape[0]), 
                          data=xyzrgb.tobytes())

        return msg


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    rclpy.init()
    indexer = VLMapIndexer(config)
    rclpy.spin(indexer)


if __name__ == "__main__":
    main()
