#from tf2_ros.buffer import Buffer
#from tf2_ros.transform_listener import TransformListener
#from tf2_ros import TransformException
import sys
import rclpy
from rclpy.node import Node
from pathlib import Path
from ros2_vlmaps_interfaces.srv import IndexMap, LoadMap
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
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
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)


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
            Marker, "/vlmap_marker", 10
        )
        # Service init
        self.index_map_srv = self.create_service(IndexMap, 'index_map', self.index_map_callback) 
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
            if self.init_categories:
                self.vlmap.init_categories(self.categories)
                mask = self.vlmap.index_map(request.indexing_string, with_init_cat=True)
                self.init_categories = False    # Do it once
            else:
                mask = self.vlmap.index_map(request.indexing_string, with_init_cat=False)

            if self.config.index_2d:
                mask_2d = pool_3d_label_to_2d(mask, self.vlmap.grid_pos, self.config.params.gs)
                rgb_2d = pool_3d_rgb_to_2d(self.vlmap.grid_rgb, self.vlmap.grid_pos, self.config.params.gs)
                self.publish_markers(mask_2d, rgb_2d)
                #visualize_masked_map_2d(rgb_2d, mask_2d)
                #heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=self.config.params.cs, decay_rate=self.config.decay_rate)
                #visualize_heatmap_2d(rgb_2d, heatmap)
                #self.publish_markers(mask_2d, rgb_2d)
            else:
                visualize_masked_map_3d(self.vlmap.grid_pos, mask, self.vlmap.grid_rgb)
                heatmap = get_heatmap_from_mask_3d(
                    self.vlmap.grid_pos, mask, cell_size=self.config.params.cs, decay_rate=self.config.decay_rate
                )
                visualize_heatmap_3d(self.vlmap.grid_pos, heatmap, self.vlmap.grid_rgb)
        except Exception as ex:
            print(f"Unexpected exception: {ex=}, {type(ex)=}")
            response.is_ok = False
            response.error_msg = "[index_map_callback] An exception occurred:"
            return response
        response.is_ok = True
        return response
        ### TODO use ROS2 publishing with publish_markers for 2D

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
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            for u in range(mask_2d.shape[0]):
                for v in range(mask_2d.shape[1]):
                    if mask_2d[u,v] == True:
                        color = rgb_2d[u, v]
                        if np.mean(color) < 255:  
                            p = Point()
                            p.x = v  * 0.05  - (mask_2d.shape[0] * 0.05 / 2) # TODO parameterize + self.config.map_origin[0]
                            p.y = u * 0.05  - (mask_2d.shape[1] * 0.05 / 2)
                            p.z = 0.10
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