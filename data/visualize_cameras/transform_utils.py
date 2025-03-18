import open3d as o3d
import numpy as np
import json
import numpy as np
import sys
sys.path.append('nerf_pl')
from visualize_nerf.utils import repeat_interleave


import torch
import math

import numpy as np
import torch
import tqdm
import copy

# Automatic rescale & offset the poses.
def find_transforms_center_and_scale(raw_transforms):
    print("computing center of attention...")
    frames = raw_transforms['frames']
    for frame in frames:
        frame['transform_matrix'] = np.array(frame['transform_matrix'])
                
    print("frames", len(frames))

    rays_o = []
    rays_d = []
    for f in frames:
        mf = f["transform_matrix"][0:3,:]
        rays_o.append(mf[:3,3:])
        rays_d.append(mf[:3,2:3])
    rays_o = np.asarray(rays_o)
    rays_d = np.asarray(rays_d)

    # Find the point that minimizes its distances to all rays.
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    translation = min_line_dist(rays_o, rays_d)
    normalized_transforms = copy.deepcopy(raw_transforms)
    for f in normalized_transforms["frames"]:
        f["transform_matrix"][0:3,3] -= translation

    # Find the scale.
    avglen = 0.
    for f in normalized_transforms["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    nframes = len(normalized_transforms["frames"])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    scale = 1.0 / avglen # scale to "nerf sized"

    return translation, scale

def normalize_transforms(transforms, translation, scale):
	normalized_transforms = copy.deepcopy(transforms)
	for f in normalized_transforms["frames"]:
		f["transform_matrix"] = np.asarray(f["transform_matrix"])
		f["transform_matrix"][0:3,3] -= translation
		f["transform_matrix"][0:3,3] *= scale
		f["transform_matrix"] = f["transform_matrix"].tolist()
	return normalized_transforms

def arkit_get_pose(frame_pose):
    frame_pose[0:3, 1:3] *= -1
    frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]
    frame_pose[2, :] *= -1

    return frame_pose

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def poses_to_matrices(poses):
    # Create a tensor of [0, 0, 0, 1] with shape [N, 1, 4]
    last_row = torch.zeros((poses.shape[0], 1, 4), dtype=poses.dtype, device=poses.device)
    last_row[:, :, 3] = 1.0

    # Concatenate poses with the last row
    homogeneous_matrices = torch.cat([poses, last_row], dim=1)

    return homogeneous_matrices

def auto_orient_and_center_poses(
    poses,
    method = "none",
    center_method = "poses",
): 
#-> Tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:
    """Orients and centers the poses. We provide two methods for orientation: pca and up.

    pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:
    poses: The poses are centered around the origin.
    focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    translation = torch.zeros_like(mean_origin)
    
    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
        oriented_poses = poses_to_matrices(oriented_poses)

    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
        oriented_poses = poses_to_matrices(oriented_poses)
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform3by4 = transform[:3, :]
        print("poses", poses.shape)
        oriented_poses = transform3by4 @ poses
        oriented_poses = poses_to_matrices(oriented_poses)
    else:
        raise ValueError(f"Unknown value for method: {method}")
    
    return oriented_poses, transform

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W
    
def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    return bbox_3d

def convert_points_to_homopoints(points):
  """Project 3d points (3xN) to 4d homogenous points (4xN)"""
  assert len(points.shape) == 2
  assert points.shape[0] == 3
  points_4d = np.concatenate([
      points,
      np.ones((1, points.shape[1])),
  ], axis=0)
  assert points_4d.shape[1] == points.shape[1]
  assert points_4d.shape[0] == 4
  return points_4d

def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d

def projection(c_xyz, focal, c, NV):
    """Converts [x,y,z] in camera coordinates to image coordinates 
        for the given focal length focal and image center c.
    :param c_xyz: points in camera coordinates (SB*NV, NP, 3)
    :param focal: focal length (SB, 2)
    :c: image center (SB, 2)
    :output uv: pixel coordinates (SB, NV, NP, 2)
    """
    uv = -c_xyz[..., :2] / (c_xyz[..., 2:] + 1e-9)  # (SB*NV, NC, 2); NC: number of grid cells 
    # uv = c_xyz[..., :2] / (c_xyz[..., 2:] + 1e-9)  # (SB*NV, NC, 2); NC: number of grid cells 
    uv *= repeat_interleave(
                focal.unsqueeze(1), NV if focal.shape[0] > 1 else 1
            )
    uv += repeat_interleave(
                c.unsqueeze(1), NV if c.shape[0] > 1 else 1
            )
    return uv

def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d