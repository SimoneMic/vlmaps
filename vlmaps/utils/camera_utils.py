import numpy as np
import copy
import torch

class FeturedPoint:
    def __init__(self, point, embedding, rgb) -> None:
        self.point_xyz = point
        self.embedding = embedding
        self.rgb = rgb

class FeaturedPC:
    def __init__(self, featured_points = []) -> None:
        self.featured_points = featured_points
        self.points_xyz = np.zeros([len(self.featured_points), 3])
        self.embeddings = np.zeros([len(self.featured_points), 512])  # TODO parameterize embeddings size
        self.rgb = np.zeros([len(self.featured_points), 3])
        i = 0
        if len(featured_points) > 0:
            for featured_point in self.featured_points:
                self.points_xyz[i] = featured_point.point_xyz
                self.embeddings[i] = featured_point.embedding
                self.rgb[i] = featured_point.rgb
                i += 1

def project_depth_features_pc_torch(depth, features_per_pixels, color_img, calib_matrix, min_depth = 0.2, max_depth = 6.0, depth_factor=1.0, downsampling_factor=10.0):
        """
        Creates the 3D pointcloud, in camera frame, from the depth and alignes the clip features and RGB color for each 3D point.
        Uses tensors to speed up the process. Uses GPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param features_per_pixels: matrix of shape (1, F, W , H), featured clip embeddings aligned with each RGB pixel
        :param color_img: matrix of shape (W , H), color image image from the camera
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
        :return: numpy array of shape (N, 3) of 3D points, numpy array of shape (N, F) containing aligned CLIP features to each 3D point, numpy array of shape (N, 3) of aligned RGB color for each point
        """
        fx = calib_matrix[0, 0]
        fy = calib_matrix[1, 1]
        cx = calib_matrix[0, 2]
        cy = calib_matrix[1, 2]

        if min_depth < 0.0:
              min_depth = 0.2
        if max_depth < 0.0:
              max_depth = 6.0

        #intrisics = [[fx, 0.0, cx],
        #             [0.0, fy, cy],
        #             [0.0, 0.0, 1.0 / depth_factor]]
        #intrisics = torch.tensor(list(intrisics), device='cuda').type(torch.float32)
        depth = torch.tensor(list(depth), device='cuda').type(torch.float32)
        # filter depth coords based on z distance
        uu, vv = torch.where((depth > min_depth) & (depth < max_depth))

        # Shuffle and downsample depth pixels
        coords = torch.stack((uu, vv), dim=1)  # pixel pairs vector
        coords = coords[torch.randperm(coords.size()[0])]
        coords = coords[:int(coords.size(dim=0)/downsampling_factor)]
        uu = coords[:, 0]
        vv = coords[:, 1]
        xx = (vv - cx) * depth[uu, vv] / fx
        yy = (uu - cy) * depth[uu, vv] / fy
        zz = depth[uu, vv] / depth_factor

        uu, vv = uu.cpu().numpy(), vv.cpu().numpy()
        features = copy.deepcopy(features_per_pixels[0, :, uu, vv])
        color = color_img[uu, vv, :]

        pointcloud = torch.cat((xx.unsqueeze(1), yy.unsqueeze(1), zz.unsqueeze(1)), 1)
        pointcloud= pointcloud.cpu().numpy()

        return pointcloud, features, color

def project_depth_features_pc(depth, features_per_pixels, color_img, calib_matrix, min_depth = 0.2, max_depth = 6.0, depth_factor=1., downsample_factor=10):
        """
        Creates the 3D pointcloud, in camera frame, from the depth and alignes the clip features and RGB color for each 3D point.
        Uses CPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param features_per_pixels: matrix of shape (1, F, W , H), featured clip embeddings aligned with each RGB pixel
        :param color_img: matrix of shape (W , H), color image image from the camera
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
        :return: array of shape (N, 3) of type FeaturedPC
        """
        fx = calib_matrix[0, 0]
        fy = calib_matrix[1, 1]
        cx = calib_matrix[0, 2]
        cy = calib_matrix[1, 2]

        if min_depth < 0.0:
              min_depth = 0.2
        if max_depth < 0.0:
              max_depth = 6.0

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
            if (z > min_depth and z < max_depth): # filter depth based on Z TODO parameterize these thresholds
                z = z / depth_factor
                x = ((v - cx) * z) / fx
                y = ((u - cy) * z) / fy
                feature_points_ls[count] = FeturedPoint([x, y, z],copy.deepcopy(features_per_pixels[0, :, u, v]), color_img[u, v, :]) # avoid memory re-allocation each loop iter
                count += 1
        feature_points_ls.resize(count, refcheck=False)
        return FeaturedPC(feature_points_ls)

def from_depth_to_pc(depth, calib_matrix, min_depth = 0.2, max_depth = 6.0, depth_factor=1., downsample_factor=10):
        """
        Creates the 3D pointcloud, in camera frame, from the depth.
        Uses CPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
        :return: array of shape (N, 3)
        """
        #fx, fy, cx, cy = intrinsics
        fx = calib_matrix[0, 0]
        fy = calib_matrix[1, 1]
        cx = calib_matrix[0, 2]
        cy = calib_matrix[1, 2]

        if min_depth < 0.0:
              min_depth = 0.2
        if max_depth < 0.0:
              max_depth = 6.0
                
        h, w = depth.shape
        points = np.zeros([h*w , 3])
        count = 0
        for u in range(0, h):
            for v in range(0, w):
                z = depth[u, v]
                if (z > min_depth and z < max_depth): # filter depth based on Z
                    z = z / depth_factor
                    x = ((v - cx) * z) / fx
                    y = ((u - cy) * z) / fy
                    points[count] = [x, y, z]
                    count += 1
        np.resize(points, count)

        #Downsample
        points=points[(np.random.randint(0, points.shape[0], np.round(count/downsample_factor, 0).astype(int)) )]
        return points


def project_pc(rgb, points, calib_matrix, depth_factor=1.):
        """
        Projects the pointcloud to the RGB image
        """
        k = np.eye(3)
        # Ergocub intrinsics: Realsense D455
        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047,
                      'width': 640, 'height': 480}
        width = intrinsics["width"]
        height = intrinsics["height"]

        #k[0, :] = np.array([intrinsics['fx'], 0, intrinsics['ppx']])
        #k[1, 1:] = np.array([intrinsics['fy'], intrinsics['ppy']])
        k = calib_matrix

        points = np.array(points) * depth_factor
        uv = k @ points.T
        uv = uv[0:2] / uv[2, :]

        uv = np.round(uv, 0).astype(int)
    
        uv[0, :] = np.clip(uv[0, :], 0, height-1)
        uv[1, :] = np.clip(uv[1, :], 0, width-1)

        rgb[uv[1, :], uv[0, :], :] = ((points - points.min(axis=0)) / (points - points.min(axis=0)).max(axis=0) * 255).astype(int)

        return rgb

