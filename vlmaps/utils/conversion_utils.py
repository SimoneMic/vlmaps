import numpy as np
import math
from std_msgs.msg import Header
from sensor_msgs.msg import PointField, PointCloud2

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

def xyzrgb_array_to_pointcloud2(points, colors, stamp, frame_id, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points and a synched array of color values.
        '''

        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyzrgb')]
        nbytes = 6
        xyzrgb = np.array(np.hstack([points, colors/255]), dtype=np.float32)
        #xyzrgb = np.array(points_ren, dtype=np.float32)
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