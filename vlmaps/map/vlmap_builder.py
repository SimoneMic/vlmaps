import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
#import open3d as o3d
from cv_bridge import CvBridge

from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.utils.mapping_utils import (
    load_3d_map,
    save_3d_map,
    cvt_pose_vec2tf,
    load_depth_npy,
    depth2pc,
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
)
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet

#ROS2 stuff
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
#import sensor_msgs_py
import rclpy
import math
#from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

#def visualize_pc(pc: np.ndarray):
#    pcd = o3d.geometry.PointCloud()
#    pcd.points = o3d.utility.Vector3dVector(pc)
#    o3d.visualization.draw_geometries([pcd])

def quaternion_matrix(quaternion):  #Copied from https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf/transformations.py#L1515
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

#### ROS2 wrapper
class VLMapBuilderROS(Node):
    def __init__(
        self,
        #data_dir: Path,
        map_config: DictConfig
    ):
        super().__init__('VLMap_builder_node')
        self.map_config = map_config

        # tf buffer init
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # subscribers init with callback
        img_topic = "/cer/realsense_repeater/color_image"   # TODO initialize from config file
        depth_topic = "/cer/realsense_repeater/depth_image"
        self.img_sub = message_filters.Subscriber(self, Image, img_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], 1, slop=0.3)        
        self.tss.registerCallback(self.sensors_callback)
        ## First part of create_mobile_base_map for init stuff
        # access config info
        camera_height = self.map_config.pose_info.camera_height
        self.cs = self.map_config.cell_size
        self.gs = self.map_config.grid_size
        self.depth_sample_rate = self.map_config.depth_sample_rate

        self.map_save_dir = "/home/ergocub"
        os.makedirs(self.map_save_dir, exist_ok=True)
        self.map_save_path = self.map_save_dir + "/" + "vlmaps.h5df"

        # init lseg model
        self.lseg_model, self.lseg_transform, self.crop_size, self.base_size, self.norm_mean, self.norm_std = self._init_lseg()

        # init the map
        (
            self.vh,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
            self.mapped_iter_set,
            self.max_id,
        ) = self._init_map(camera_height, self.cs, self.gs, self.map_save_path)

        self.cv_bridge = CvBridge()

        #### Iteration counter
        self.frame_i = 0
        # load camera calib matrix in config
        self.calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))
        self.cv_map = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.height_map = -100 * np.ones((self.gs, self.gs), dtype=np.float32)

        #pbar = tqdm(zip(self.rgb_paths, self.depth_paths, self.base_poses), total=len(self.rgb_paths))

    def sensors_callback(self, img_msg, depth_msg):
        """
        build the 3D map centering at the first base frame
        """
        self.get_logger().info('sensors_callback')
        #### Convert the rgb format from ros to OpenCv
        rgb = self.cv_bridge.imgmsg_to_cv2(img_msg) # TODO check image color encoding
        #### Convert depth from ros2 to OpenCv
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")  # TODO check image color encoding
        depth = depth.astype(np.float16)
        #### TODO should I normalize the depth?
        self.get_logger().info('Ros2 to CV2 conversion')

        # get pixel-aligned LSeg features
        pix_feats = get_lseg_feat(
            self.lseg_model, rgb, ["example"], self.lseg_transform, self.device, self.crop_size, self.base_size, self.norm_mean, self.norm_std
        )
        self.get_logger().info('lseg features extracted')
        #pix_feats_intr = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])
        #### It's the same as the camera one, we are not in simulation anymore
        pix_feats_intr = self.calib_mat 
        # backproject depth point cloud
        pc = self._backproject_depth(depth, self.calib_mat, self.depth_sample_rate, min_depth=0.2, max_depth=6) 
        self.get_logger().info('backprojected depth')
        # transform the point cloud to global frame (init base frame)
        #pc_transform = tf @ self.base_transform @ self.base2cam_tf
        #pc_global = transform_pc(pc, pc_transform)  # (3, N)

        #### Transform PC to map frame - i.e. global frame
        target_frame="map"
        source_frame="head_link"
        try:
            transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time()
                    )  #rclpy.time.Time() rclpy.duration.Duration(seconds=0.1)
            #img_msg.header.stamp
        except TransformException as ex:
                self.get_logger().info(
                        f'Could not transform {source_frame} to {target_frame}: {ex}')
                return
        self.get_logger().info('Transform available')
        #### Convert tf2 transform to np array components
        transform_pose_np = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        transform_quat_np = np.array([transform.transform.rotation.x, transform.transform.rotation.y,
                                        transform.transform.rotation.z, transform.transform.rotation.w])
        #### Let's get an SE(4) matrix form
        transform_np = quaternion_matrix(transform_quat_np)
        transform_np[0:3, -1] = transform_pose_np
        #### Transform the pc using the default util
        pc_global = transform_pc(pc, transform_np)  # (3, N)
        self.get_logger().info('Evaluation of each point in the PC and projection into map with')
        # Evaluation Loop of each point in 3d
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            #self.get_logger().info(f'loop number {i}')
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p[0], p[1], p[2])
            if self._out_of_range(row, col, height, self.gs, self.vh):
                self.get_logger().info(f"out of range with p0 {p[0]} p1 {p[1]} p2 {p[2]}")
                continue
            
            px, py, pz = project_point(self.calib_mat, p_local)
            rgb_v = rgb[py, px, :]
            px, py, pz = project_point(pix_feats_intr, p_local)
            
            if height > self.height_map[row, col]:
                self.height_map[row, col] = height
                self.cv_map[row, col, :] = rgb_v

            # when the max_id exceeds the reserved size,
            # double the grid_feat, grid_pos, weight, grid_rgb lengths
            if self.max_id >= self.grid_feat.shape[0]:
                self._reserve_map_space(self.grid_feat, self.grid_pos, self.weight, self.grid_rgb)

            # apply the distance weighting according to
            # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
            radial_dist_sq = np.sum(np.square(p_local))
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

            # update map features
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                occupied_id = self.occupied_ids[row, col, height]
                if occupied_id == -1:
                    self.occupied_ids[row, col, height] = self.max_id
                    self.grid_feat[self.max_id] = feat.flatten() * alpha
                    self.grid_rgb[self.max_id] = rgb_v
                    self.weight[self.max_id] += alpha
                    self.grid_pos[self.max_id] = [row, col, height]
                    self.max_id = self.max_id + 1
                else:
                    self.grid_feat[occupied_id] = (
                        self.grid_feat[occupied_id] * self.weight[occupied_id] + feat.flatten() * alpha
                    ) / (self.weight[occupied_id] + alpha)
                    self.grid_rgb[occupied_id] = (self.grid_rgb[occupied_id] * self.weight[occupied_id] + rgb_v * alpha) / (
                        self.weight[occupied_id] + alpha
                    )
                    self.weight[occupied_id] += alpha

        self.mapped_iter_set.add(self.frame_i)
        #if self.frame_i % 100 == 99:
        #### Save Map each N callbacks TODO parameterize
        if self.frame_i % 30 == 0:
            self.get_logger().info(f"Temporarily saving {self.max_id} features at iter {self.frame_i}...")
            self._save_3d_map(self.grid_feat, self.grid_pos, self.weight, self.grid_rgb, self.occupied_ids, self.mapped_iter_set, self.max_id)
        self.frame_i += 1   # increase counter
        self.get_logger().info(f"iter {self.frame_i}")
        # TODO put it in a callback
        #self._save_3d_map(self.grid_feat, self.grid_pos, self.weight, self.grid_rgb, self.occupied_ids, self.mapped_iter_set, self.max_id)

    def _init_map(self, camera_height: float, cs: float, gs: int, map_path: Path) -> Tuple:
        """
        initialize a voxel grid of size (gs, gs, vh), vh = camera_height / cs, each voxel is of
        size cs
        """
        # init the map related variables
        vh = int(camera_height / cs)
        grid_feat = np.zeros((gs * gs, self.clip_feat_dim), dtype=np.float32)
        grid_pos = np.zeros((gs * gs, 3), dtype=np.int32)
        occupied_ids = -1 * np.ones((gs, gs, vh), dtype=np.int32)
        weight = np.zeros((gs * gs), dtype=np.float32)
        grid_rgb = np.zeros((gs * gs, 3), dtype=np.uint8)
        mapped_iter_set = set()
        mapped_iter_list = list(mapped_iter_set)
        max_id = 0

        # check if there is already saved map TODO print warn msg
        if os.path.exists(map_path):
            (
                mapped_iter_list,
                grid_feat,
                grid_pos,
                weight,
                occupied_ids,
                grid_rgb,
            ) = load_3d_map(self.map_save_path)
            mapped_iter_set = set(mapped_iter_list)
            max_id = grid_feat.shape[0]

        return vh, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id

    def _init_lseg(self):
        crop_size = 480  # 480
        base_size = 520  # 520
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"checkpoint path is : {checkpoint_path}")

        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _backproject_depth(
        self,
        depth: np.ndarray,
        calib_mat: np.ndarray,
        depth_sample_rate: int,
        min_depth: float = 0.1,
        max_depth: float = 10,
    ) -> np.ndarray:
        pc, mask = depth2pc(depth, intr_mat=calib_mat, min_depth=min_depth, max_depth=max_depth)  # (3, N)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        return pc

    def _out_of_range(self, row: int, col: int, height: int, gs: int, vh: int) -> bool:
        #return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0
        return col >= gs or row >= gs or col < 0 or row < 0 or height < 0

    def _reserve_map_space(
        self, grid_feat: np.ndarray, grid_pos: np.ndarray, weight: np.ndarray, grid_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_feat = np.concatenate(
            [
                grid_feat,
                np.zeros((grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_pos = np.concatenate(
            [
                grid_pos,
                np.zeros((grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        weight = np.concatenate([weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0)
        grid_rgb = np.concatenate(
            [
                grid_rgb,
                np.zeros((grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        return grid_feat, grid_pos, weight, grid_rgb

    def _save_3d_map(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        occupied_ids: Set,
        mapped_iter_set: Set,
        max_id: int,
    ) -> None:
        grid_feat = grid_feat[:max_id]
        grid_pos = grid_pos[:max_id]
        weight = weight[:max_id]
        grid_rgb = grid_rgb[:max_id]
        save_3d_map(self.map_save_path, grid_feat, grid_pos, weight, occupied_ids, list(mapped_iter_set), grid_rgb)
