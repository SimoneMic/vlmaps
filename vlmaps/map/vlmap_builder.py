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
import open3d as o3d
from cv_bridge import CvBridge
import time
from geometry_msgs.msg import PoseWithCovarianceStamped
import gc

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
    base_pos2grid_id_3d_torch
)
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
import vlmaps.utils.traverse_pixels as raycast

#ROS2 stuff
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters
import rclpy
import math
import copy
import torch

try:
    from open3d.cuda.pybind.geometry import PointCloud
    from open3d.cuda.pybind.utility import Vector3dVector
    from open3d.cuda.pybind.visualization import draw_geometries
except ImportError:
    from open3d.cpu.pybind.geometry import PointCloud
    from open3d.cpu.pybind.utility import Vector3dVector
    from open3d.cpu.pybind.visualization import draw_geometries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def visualize_pc(pc: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

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

class FeturedPoint:
    def __init__(self, point, embedding, rgb) -> None:
        self.point_xyz = point
        self.embedding = embedding
        self.rgb = rgb

class FeaturedPC:
    def __init__(self, featured_points) -> None:
        self.featured_points = featured_points
        self.points_xyz = np.zeros([len(self.featured_points), 3])
        self.embeddings = np.zeros([len(self.featured_points), 512])  # TODO parameterize embeddings size
        self.rgb = np.zeros([len(self.featured_points), 3])
        i = 0
        for featured_point in self.featured_points:
            self.points_xyz[i] = featured_point.point_xyz
            self.embeddings[i] = featured_point.embedding
            self.rgb[i] = featured_point.rgb
            i += 1

#### ROS2 wrapper
class VLMapBuilderROS(Node):
    def __init__(
        self,
        #data_dir: Path,
        map_config: DictConfig
    ):
        super().__init__('VLMap_builder_node')
        self.map_config = map_config
        self.amcl_cov = np.full([36], 0.2)  #init with an high covariance value for each element
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
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            10
        )
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

        ### Make more explicit the calib intrinsics:
        self.focal_lenght_x = self.calib_mat[0,0]       #fx
        self.focal_lenght_y = self.calib_mat[1,1]       #fy
        self.principal_point_x = self.calib_mat[0,2]    #cx or ppx
        self.principal_point_y = self.calib_mat[1,2]    #cy or ppy

        self.device_o3d = o3d.core.Device("CPU:0")
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'), #, 'embeddings'
            attr_dtypes=(o3d.core.float32, o3d.core.float32),
            attr_channels=((1), (1)),  #, (512)
            voxel_size=0.05,
            block_resolution=16,
            block_count=100000,
            device=self.device_o3d)
    
    def project_pc(self, rgb, points, depth_factor=1.):
        k = np.eye(3)
        # Ergocub intrinsics: Realsense D455
        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047,
                      'width': 640, 'height': 480}
        width = intrinsics["width"]
        height = intrinsics["height"]

        #k[0, :] = np.array([intrinsics['fx'], 0, intrinsics['ppx']])
        #k[1, 1:] = np.array([intrinsics['fy'], intrinsics['ppy']])
        k = self.calib_mat

        points = np.array(points) * depth_factor
        uv = k @ points.T
        uv = uv[0:2] / uv[2, :]

        uv = np.round(uv, 0).astype(int)
    
        uv[0, :] = np.clip(uv[0, :], 0, height-1)
        uv[1, :] = np.clip(uv[1, :], 0, width-1)

        rgb[uv[1, :], uv[0, :], :] = ((points - points.min(axis=0)) / (points - points.min(axis=0)).max(axis=0) * 255).astype(int)

        return rgb

    def from_depth_to_pc(self, depth, depth_factor=1., downsample_factor=10):
        #fx, fy, cx, cy = intrinsics
        start = time.time()
        fx = self.focal_lenght_x
        fy = self.focal_lenght_y
        cx = self.principal_point_x
        cy = self.principal_point_y
        
        #points = []
        
        h, w = depth.shape
        points = np.zeros([h*w , 3])
        count = 0
        for u in range(0, h):
            for v in range(0, w):
                z = depth[u, v]
                if (z > 0.2 and z<6.0): # filter depth based on Z
                    z = z / depth_factor
                    x = ((v - cx) * z) / fx
                    y = ((u - cy) * z) / fy
                    #points.append([x, y, z])   #avoid memory allocation each loop iter
                    points[count] = [x, y, z]
                    count += 1
        np.resize(points, count)
        #points = np.array(points)
        #Downsample
        points=points[(np.random.randint(0, points.shape[0], np.round(count/downsample_factor, 0).astype(int)) )]
        time_diff = time.time() - start
        self.get_logger().info(f"Time for executing from_depth_to_pc: {time_diff}")
        return points

    def project_depth_features_pc(self, depth, features_per_pixels, color_img, depth_factor=1., downsample_factor=10):
        fx = self.focal_lenght_x
        fy = self.focal_lenght_y
        cx = self.principal_point_x
        cy = self.principal_point_y

        # randomly sample pixels from depth (should be more memory efficient)
        uu, vv = np.where(depth[:,:]>=0)
        coords = np.column_stack((uu, vv))  # pixel pairs vector
        np.random.shuffle(coords)   # I have all the pixels randomly shuffled
        # Let's take only the number of pixels scaled by the downsample factor:
        # Since we have shuffled the coordinates, we take the first N items
        downsampled_coords = coords[:np.round(len(coords)/downsample_factor, 0).astype(int)]
        feature_points_ls = np.empty([len(downsampled_coords), 3], dtype=object)
        count = 0
        for item in downsampled_coords:
            # TODO optimize iteration formulation
            u = item[0]
            v = item[1]
            z = depth[u, v]
            if (z > 0.2 and z<6.0): # filter depth based on Z TODO parameterize these thresholds
                z = z / depth_factor
                x = ((v - cx) * z) / fx
                y = ((u - cy) * z) / fy
                feature_points_ls[count] = FeturedPoint([x, y, z],copy.deepcopy(features_per_pixels[0, :, u, v]), color_img[u, v, :]) # avoid memory re-allocation each loop iter
                count += 1
        feature_points_ls.resize(count, refcheck=False)
        return FeaturedPC(feature_points_ls)

    def sensors_callback(self, img_msg, depth_msg):
        """
        build the 3D map centering at the first base frame
        """
        self.get_logger().info('sensors_callback')
        loop_timer = time.time()
        #### first do a TF check between the camera and map frame
        target_frame="map"
        source_frame="depth"
        try:
            transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    depth_msg.header.stamp
                    )
        except TransformException as ex:
                self.get_logger().info(
                        f'Could not transform {source_frame} to {target_frame}: {ex}')
                return
        self.get_logger().info('Transform available')
        # Check covariance: TODO is it enough to check only two values?
        if not (abs(self.amcl_cov.max()) < 0.01) and (abs(self.amcl_cov.min()) < 0.01):
            self.get_logger().info(f'Covariance too big: skipping callback untill')
            return

        ## Convert tf2 transform to np array components
        transform_pose_np = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        transform_quat_np = np.array([transform.transform.rotation.x, transform.transform.rotation.y,
                                        transform.transform.rotation.z, transform.transform.rotation.w])
        ## Let's get an SE(4) matrix form
        transform_np = quaternion_matrix(transform_quat_np)
        transform_np[0:3, -1] = transform_pose_np
        #### Convert Inputs formats:
        ## Convert the rgb format from ros to OpenCv
        rgb = self.cv_bridge.imgmsg_to_cv2(img_msg) # TODO check image color encoding
        ## Convert depth from ros2 to OpenCv
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")  # TODO check image color encoding
        depth = depth.astype(np.float16)
        #depth_o3d = depth.astype(np.float32)
        self.get_logger().info('Ros2 to CV2 conversion')

        #### Segment image and extract features
        # get pixel-aligned LSeg features
        start = time.time()
        pix_feats = get_lseg_feat(
            self.lseg_model, rgb, ["other", "screen", "table", "closet", "chair", "shelf", "door", "wall", "ceiling", "floor", "human"], self.lseg_transform, self.device, self.crop_size, self.base_size, self.norm_mean, self.norm_std, vis=False
        )
        time_diff = time.time() - start
        self.get_logger().info(f"lseg features extracted in: {time_diff}")

        #pc = self.from_depth_to_pc(depth, depth_factor=1.)
        #self.get_logger().info('backprojected depth')
        #### Convert normal pc to open3d format 
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(pc)
        #o3d.visualization.draw_geometries_with_vertex_selection([pcd])
        #### Convert back to numpy
        #pc_global = np.asarray(pcd_global.points)

        #### Debug
        #tmp = self.project_pc(rgb, pc, depth_factor=1.)
        #cv2.imshow("projected_pc2rgb", tmp)
        #cv2.waitKey(0)

        #### Formatted PC with aligned features to pixel
        start = time.time()
        featured_pc = self.project_depth_features_pc(depth, pix_feats, rgb, downsample_factor=20)
        time_diff = time.time() - start
        self.get_logger().info(f"Time for executing project_depth_features_pc: {time_diff}")

        #projection = self.project_pc(rgb, featured_pc.points_xyz, depth_factor=1.)
        #cv2.imshow("projected_featuredPC", projection)
        #cv2.waitKey(0)

        #### Transform PC into map frame
        start = time.time()
        pcd_feat = o3d.geometry.PointCloud()
        pcd_feat.points = o3d.utility.Vector3dVector(featured_pc.points_xyz)
        pcd_global = pcd_feat.transform(transform_np)
        featured_pc.points_xyz = np.asarray(pcd_global.points)
        time_diff = time.time() - start
        self.get_logger().info(f"Time for transforming PC in map frame: {time_diff}")
        #o3d.visualization.draw_geometries_with_vertex_selection([pcd_global])


        #### Raycast TODO combine with main loop to save computational effort
        # Camera position in grid
        start_raycast = time.time()
        map_to_cam_tf = np.linalg.inv(transform_np)
        cam_pose = map_to_cam_tf[0:3,-1]
        camera_pose_grid = base_pos2grid_id_3d(self.gs, self.cs, cam_pose[0], cam_pose[1], cam_pose[2])
        start_pixel = raycast.Pixel(camera_pose_grid[0], camera_pose_grid[1], camera_pose_grid[2])
        voxels_to_clear = []
        torch_points = torch.tensor(featured_pc.points_xyz, device="cuda")
        torch_grid = base_pos2grid_id_3d_torch(self.gs, self.cs, torch_points)
        mask = ~((torch_grid < (torch.zeros_like(torch_grid)) + (torch_grid > torch.tensor([self.gs, self.gs, self.vh], device="cuda"))).any(dim=1))
        torch_grid =  torch_grid[mask]
        
        ##################
        for point in featured_pc.points_xyz:
            point_grid = base_pos2grid_id_3d(self.gs, self.cs, point[0], point[1], point[2])
             
            if self._out_of_range(point_grid[0], point_grid[1], point_grid[2], self.gs, self.vh):
                #self.get_logger().info(f"out of range with p0 {point[0]} p1 {point[1]} p2 {point[2]}")
                continue
            end_pixel = raycast.Pixel(point_grid[0], point_grid[1], point_grid[2])
            traversed_pixels = raycast.traverse_pixels(start_pixel, end_pixel)
            # TEMPORARY FOR DEBUG
            voxels_to_clear += traversed_pixels
        ##################
        pcd_clear = o3d.geometry.PointCloud()
        pcd_clear.points = o3d.utility.Vector3dVector(voxels_to_clear)
        time_diff = time.time() - start_raycast
        self.get_logger().info(f"Time for raycasting PC in map frame: {time_diff}")
        o3d.visualization.draw_geometries_with_vertex_selection([pcd_clear])

        #### Map update TODO: separate it in another thread
        start = time.time()
        for (point, feature, rgb) in zip(featured_pc.points_xyz, featured_pc.embeddings, featured_pc.rgb):
            
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, point[0], point[1], point[2])
            if self._out_of_range(row, col, height, self.gs, self.vh):
                #self.get_logger().info(f"out of range with p0 {point[0]} p1 {point[1]} p2 {point[2]}")
                continue

            # when the max_id exceeds the reserved size,
            # double the grid_feat, grid_pos, weight, grid_rgb lengths
            if self.max_id >= self.grid_feat.shape[0]:
                self._reserve_map_space(self.grid_feat, self.grid_pos, self.weight, self.grid_rgb)
            
            # apply the distance weighting according to
            # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
            radial_dist_sq = np.sum(np.square(point))  
            sigma_sq = 0.6  #TODO parameterize
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

            #feat = pix_feats[0, :, py, px]
            feat = feature
            occupied_id = self.occupied_ids[row, col, height]

            if occupied_id == -1:
                self.occupied_ids[row, col, height] = self.max_id
                self.grid_feat[self.max_id] = feat.flatten() * alpha
                self.grid_rgb[self.max_id] = rgb
                self.weight[self.max_id] += alpha
                self.grid_pos[self.max_id] = [row, col, height]
                self.max_id = self.max_id + 1
            else:
                self.grid_feat[occupied_id] = (
                    self.grid_feat[occupied_id] * self.weight[occupied_id] + feat.flatten() * alpha
                ) / (self.weight[occupied_id] + alpha)
                self.grid_rgb[occupied_id] = (self.grid_rgb[occupied_id] * self.weight[occupied_id] + rgb * alpha) / (
                    self.weight[occupied_id] + alpha
                )
                self.weight[occupied_id] += alpha
        
        time_diff = time.time() - start
        self.get_logger().info(f"Time for updating Map: {time_diff}")
        end_loop_time = time.time() - loop_timer
        self.get_logger().info(f"CALLBACK TIME: {end_loop_time}")
        # Save map each X callbacks TODO prameterize and do it in a separate thread
        if self.frame_i % 10 == 0:
            self.get_logger().info(f"Temporarily saving {self.max_id} features at iter {self.frame_i}...")
            time_save = time.time()
            self._save_3d_map(self.grid_feat, self.grid_pos, self.weight, self.grid_rgb, self.occupied_ids, self.mapped_iter_set, self.max_id)
            time_save_diff = time.time() - time_save
            self.get_logger().info(f"Time for Saving Map: {time_save_diff}")
        self.frame_i += 1   # increase counter for map saving purposes
        self.get_logger().info(f"iter {self.frame_i}")
        return

    # Let's do this in background on a separate thread, global_pc should not be already added
    def raycasting(self, global_pc, tf_map2depth, depth_img, rgb_img):
        start = time.time()
        #### Transform the global map in the depth frame:
        map_pcd = o3d.geometry.PointCloud()
        map_pcd.points = o3d.utility.Vector3dVector(self.grid_pos[:])
        map_pcd.colors = o3d.utility.Vector3dVector(self.grid_rgb / 255)
        #o3d.visualization.draw_geometries_with_vertex_selection([map_pcd])
        map_pc_local_frame = map_pcd.transform(tf_map2depth)
        #o3d.visualization.draw_geometries_with_vertex_selection([map_pc_local_frame])

        print('voxelization')
        voxel_grid_map = o3d.geometry.VoxelGrid.create_from_point_cloud(map_pcd, voxel_size= 1.0)   #TODO parameterize
        o3d.visualization.draw_geometries([voxel_grid_map])
        #### VoxelBlockGrid computation
        
        depth_as_img = o3d.t.geometry.Image(o3d.core.Tensor((depth_img).astype(np.float32)))
        #color_as_img = o3d.t.geometry.Image(o3d.core.Tensor((rgb_img).astype(np.float32)))

        extrinsic = o3d.core.Tensor(tf_map2depth.astype(np.float64))
        depth_intrinsic = o3d.core.Tensor(self.calib_mat.astype(np.float64))
        depth_max = 6.0
        depth_scale = 1.0
        # Integration
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth_as_img, depth_intrinsic, extrinsic, depth_scale,
            depth_max)    # TODO parameterize
        self.vbg.integrate(frustum_block_coords, depth_as_img, depth_intrinsic,
                          extrinsic, depth_scale,
                          depth_max)
        # Visualize
        pcd = self.vbg.extract_point_cloud()
        o3d.visualization.draw_geometries([pcd])

        time_diff = time.time() - start
        self.get_logger().info(f"Time for raycasting: {time_diff}")

    # Simply store the covariance values, will be analyze
    def amcl_callback(self, msg):
        self.amcl_cov = msg.pose.covariance

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
        base_size = 640  # 520
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
        return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0   # TODO handle negative values

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
