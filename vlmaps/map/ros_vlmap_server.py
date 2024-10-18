import sys
import rclpy
from rclpy.node import Node
from pathlib import Path
from ros2_vlmaps_interfaces.srv import IndexMap, LoadMap, ShowMap
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker

import numpy as np
import cv2

from vlmaps.utils.mapping_utils import load_3d_map
from vlmaps.map.vlmap import VLMap
from omegaconf import DictConfig
import hydra
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_masked_map_3d,
)
from vlmaps.utils.conversion_utils import xyzrgb_array_to_pointcloud2


class VLMapPublisher(Node):
    def __init__(self, 
                 node_name: str,
                 config: DictConfig,    #TODO remove this stuff and use ros2 param yaml file
                 vlmap_dir="/home/ergocub",
                 vlmap_name="vlmaps",
                 data_dirs = "~/vlmaps", #TODO change and remove this useless stuff
                 init_categories = True,
                 ) -> None:    # TODO use ros2 parameters from yaml file
        super().__init__(node_name)
        
        self.init_categories = init_categories  #TODO is this necessary?
        self.config = config
        # Publishers Init
        self.map_pub = self.create_publisher(
            Marker, "/vlmap_2d_index_marker", 10
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2, "vlmap", 10)
        self.index_pointcloud_pub = self.create_publisher(PointCloud2, "vlmap_index_result", 10)
        # Service init
        self.index_map_srv = self.create_service(IndexMap, 'index_map', self.index_map_callback) 
        self.show_rgb_map_srv = self.create_service(ShowMap, 'show_vlmap', self.show_rgb_map_callback) 
        self.load_map_srv = self.create_service(LoadMap, 'load_map', self.load_map_callback) 

        ### VLMaps stuff:
        self.vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
        file_name = vlmap_name + ".h5df"
        ### Loading
        self.map_path = Path(vlmap_dir) / file_name     # TODO check formatting for safety measure
        try:
            if self.load_vlmap(self.map_path):
                print("VLMap loaded successfully")
        except:
            print("[VLMapPublisher:__init__] An exception occurred")

        self.vlmap._init_clip()
        
        print("considering categories: ")
        self.categories = mp3dcat[:]    # TODO improve parameterization
        print(self.categories)   #all categories
        print(f"{node_name} started successfully")
        

    def index_map_callback(self, request, response):
        try:
            mask = self.vlmap.index_map(request.indexing_string)

            if self.config.index_2d:
                mask_2d = pool_3d_label_to_2d(mask, self.vlmap.grid_pos, self.config.params.gs)
                rgb_2d = pool_3d_rgb_to_2d(self.vlmap.grid_rgb, self.vlmap.grid_pos, self.config.params.gs)
                self.publish_markers(mask_2d, rgb_2d)
                #visualize_masked_map_2d(rgb_2d, mask_2d)
                #heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=self.config.params.cs, decay_rate=self.config.decay_rate)
                #visualize_heatmap_2d(rgb_2d, heatmap)
                #self.publish_markers(mask_2d, rgb_2d)
            else:
                #visualize_masked_map_3d(self.vlmap.grid_pos, mask, self.vlmap.grid_rgb)
                grid_rgb = self.vlmap.grid_rgb.copy()
                grid_rgb[:] = [0, 0, 255]
                grid_rgb[mask] = [255, 0, 0]
                pos_mask = (self.vlmap.grid_pos > 0).all(axis=1)
                color = grid_rgb[pos_mask]
                points = self.vlmap.grid_pos[pos_mask] * 0.05  #scale it to meters
                msg = xyzrgb_array_to_pointcloud2(points, color, stamp=self.get_clock().now().to_msg(), frame_id="vlmap")
                self.get_logger().info(f"{type(msg)}")
                self.index_pointcloud_pub.publish(msg)
        except Exception as ex:
            print(f"Unexpected exception: {ex=}, {type(ex)=}")
            response.is_ok = False
            response.error_msg = "[index_map_callback] An exception occurred:"
            return response
        response.is_ok = True
        return response

    def show_rgb_map_callback(self, request, response):
        try:
            pos_mask = (self.vlmap.grid_pos > 0).all(axis=1)
            color = self.vlmap.grid_rgb[pos_mask]
            points = self.vlmap.grid_pos[pos_mask] * 0.05 
            msg = xyzrgb_array_to_pointcloud2(self.get_clock().now().to_msg(), points, color)
            self.pointcloud_pub.publish(msg)
        except Exception as ex:
            print(f"Unexpected exception: {ex=}, {type(ex)=}")
            response.is_ok = False
            response.error_msg = "[show_rgb_map_callback] An exception occurred:"
            return response
        response.is_ok = True
        return response

    def publish_markers(self, mask_2d, rgb_2d):
        try:
            marker = Marker()
            marker.id = 0
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.POINTS
            marker.action = marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = 0.1
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            # VLMAPS has inverted x and y axis, on how they save the map. So we need to do a 180 rotation around Z axis
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 1.0
            marker.pose.orientation.w = 0.0

            for u in range(mask_2d.shape[0]):
                for v in range(mask_2d.shape[1]):
                    if mask_2d[u,v] == True:
                        color = rgb_2d[u, v]
                        if np.mean(color) < 255:  
                            p = Point()
                            p.x = u  * 0.05  - (mask_2d.shape[0] * 0.05 / 2) # TODO parameterize + self.config.map_origin[0]
                            p.y = v * 0.05  - (mask_2d.shape[1] * 0.05 / 2)
                            p.z = 0.20
                            marker.points.append(p)

                            c = ColorRGBA()
                            c.b, c.g, c.r = color / 255
                            c.a = 0.1
                            marker.colors.append(c)

            self.map_pub.publish(marker)
        except Exception as ex:
            print(f"[publish_markers] Unexpected exception: {ex=}, {type(ex)=}")

    def load_map_callback(self, request, response):
        try:
            path = request.path + ".h5df"
            if not self.load_vlmap(path):
                response.is_ok = False
                response.error_msg="VLMap path not valid: the file doesn't exist"
                return response
            else:
                response.is_ok = True
                return response
        except:
            print("[load_map_callback] An exception occurred")
            response.is_ok = False
            response.error_msg="An exception occurred"
            return response

    def load_vlmap(self, path) -> bool:
        self.map_path = Path(path) 
        print(f"VLMap path: {self.map_path}")
        if not self.map_path.exists():
            print("VLMap path not valid: the file doesn't exist.")
            return False

        (
            self.mapped_iter_list,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
        ) = load_3d_map(self.map_path)

        self.vlmap.vlmap_load_3d_map_params(self.mapped_iter_list,
                                            self.grid_feat,
                                            self.grid_pos,
                                            self.weight,
                                            self.occupied_ids,
                                            self.grid_rgb)
        self.get_logger().info(f"Successfully loaded map: {self.map_path}")
        return True


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="map_indexing_cfg_ros2.yaml",
)
def main(config, args=None):
    rclpy.init(args=args)
    map_dir = "/home/ergocub"
    map_name="vlmaps"
    ros_node = VLMapPublisher("vlmaps_srv_node" ,config, vlmap_dir=map_dir, vlmap_name=map_name)
    rclpy.spin(ros_node)

    rclpy.shutdown()

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)