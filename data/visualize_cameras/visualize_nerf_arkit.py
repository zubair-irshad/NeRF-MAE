import open3d as o3d
import json
import numpy as np
import json

import torch
from kornia import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
from transform_utils import *

import trimesh
from MinimumBoundingBox import MinimumBoundingBox
# from datasets.google_scanned_utils import *
# import cv2
# from PIL import Image


def compute_box_3d(scale, transform, rotation):
    scales = [i / 2 for i in scale]
    l, h, w = scales
    center = np.reshape(transform, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(np.transpose(rotation),
                        np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    bbox3d_raw = np.transpose(corners_3d)
    return bbox3d_raw



def calculate_tight_scene_obb(extents, orientations, positions, scene_obb, margin=0.05):
    # Calculate the minimum and maximum coordinates of all object OBBs
    min_coords = np.inf
    max_coords = -np.inf

    for extent, orientation, position in zip(extents, orientations, positions):
        obb = o3d.geometry.OrientedBoundingBox(center=position, R=orientation, extent=extent)
        obb_points = np.asarray(obb.get_box_points())

        min_coords = np.minimum(min_coords, obb_points.min(axis=0))
        max_coords = np.maximum(max_coords, obb_points.max(axis=0))


    # Estimate scene bounding box using camera poses
    camera_pos = []
    for c2w in all_c2w:
        xform = np.array(c2w)
        camera_pos.append(xform[:3, 3])

    camera_pos = np.array(camera_pos)
    min_pt = np.min(camera_pos, axis=0)
    max_pt = np.max(camera_pos, axis=0)

    # # Combine camera and object bounding boxes
    min_coords = np.minimum(min_pt, min_coords)
    max_coords = np.maximum(max_pt, max_coords)


    enlarging_amt = (max_coords - min_coords) * margin
    min_coords -= enlarging_amt
    max_coords += enlarging_amt

    # Calculate the new scene bounding box
    scene_min = np.array(scene_obb[:3]) - np.array(scene_obb[3:6]) / 2.0
    scene_max = np.array(scene_obb[:3]) + np.array(scene_obb[3:6]) / 2.0
    # new_scene_min = np.minimum(scene_min, min_coords)
    # new_scene_max = np.maximum(scene_max, max_coords)

    new_scene_min = np.maximum(scene_min, min_coords)
    new_scene_max = np.minimum(scene_max, max_coords)

    new_scene_center = (new_scene_min + new_scene_max) / 2.0
    new_scene_extent = new_scene_max - new_scene_min
    new_scene_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, scene_obb[6]])

    new_scene_obb = o3d.geometry.OrientedBoundingBox(
        center=new_scene_center,
        extent=new_scene_extent,
        R=new_scene_rotation
    )

    return new_scene_obb

def estimate_scene_bounding_box(all_c2w, bounding_boxes, margin=0.1):
    print('Estimating scene bounding box using cameras and object bounding boxes')
    
    (extents, orientations, positions) =  bounding_boxes
    
    # Estimate scene bounding box using camera poses
    camera_pos = []
    for c2w in all_c2w:
        xform = np.array(c2w)
        camera_pos.append(xform[:3, 3])

    camera_pos = np.array(camera_pos)
    min_pt = np.min(camera_pos, axis=0)
    max_pt = np.max(camera_pos, axis=0)

    # Estimate scene bounding box using object bounding boxes
    object_pos = []
    for extent, orientation, position in zip(extents, orientations, positions):
    # for bbox in object_bboxes:
        bbox = o3d.geometry.OrientedBoundingBox(center = position, R = orientation, extent=extent)
        min_pt_obj = bbox.get_min_bound()
        max_pt_obj = bbox.get_max_bound()
        object_pos.append(min_pt_obj)
        object_pos.append(max_pt_obj)

    object_pos = np.array(object_pos)
    min_pt_obj = np.min(object_pos, axis=0)
    max_pt_obj = np.max(object_pos, axis=0)

    # # Combine camera and object bounding boxes
    min_pt = np.minimum(min_pt, min_pt_obj)
    max_pt = np.maximum(max_pt, max_pt_obj)

    # min_pt = min_pt_obj
    # max_pt = max_pt_obj

    print("min_pt: ", min_pt)
    print("max_pt: ", max_pt)

    # min_pt = np.array([min_pt[0], min_pt[1], min_pt[2]])
    # max_pt = np.array([max_pt[0], max_pt[1], max_pt[2]])

    enlarging_amt = (max_pt - min_pt) * margin
    min_pt -= enlarging_amt
    max_pt += enlarging_amt
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_pt, max_pt)
    aabb.color = [1, 0, 0]

    return aabb

# def estimate_scene_bounding_box(all_c2w, margin=0.1):

#     print('Estimating scene bounding box using cameras')
#     camera_pos = []
#     for c2w in all_c2w:
#         xform = np.array(c2w)
#         camera_pos.append(xform[:3, 3])

#     camera_pos = np.array(camera_pos)
#     min_pt = np.min(camera_pos, axis=0)
#     max_pt = np.max(camera_pos, axis=0)

#     enlarging_amt = (max_pt - min_pt) * margin
#     min_pt -= enlarging_amt
#     max_pt += enlarging_amt
#     aabb = o3d.geometry.AxisAlignedBoundingBox(min_pt, max_pt)
#     aabb.color = [1,0,0]

#     return aabb



def load_json(js_path):
    with open(js_path, "r") as f:
        json_data = json.load(f)
    return json_data




def draw_aabb():
    # Create a PointCloud object
    # pcd = o3d.geometry.PointCloud()

    min_bound = np.array([0.535191, 0.354459, 0.506802])
    max_bound = np.array([4.22641, 1.70179, 1.5952])
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # # Define the vertices of the AABB
    # vertices = np.array([
    #     [min_coords[0], min_coords[1], min_coords[2]],
    #     [max_coords[0], min_coords[1], min_coords[2]],
    #     [min_coords[0], max_coords[1], min_coords[2]],
    #     [max_coords[0], max_coords[1], min_coords[2]],
    #     [min_coords[0], min_coords[1], max_coords[2]],
    #     [max_coords[0], min_coords[1], max_coords[2]],
    #     [min_coords[0], max_coords[1], max_coords[2]],
    #     [max_coords[0], max_coords[1], max_coords[2]]
    # ])

    # # Create an axis-aligned bounding box (AABB)
    # aabb = o3d.geometry.AxisAlignedBoundingBox()
    # aabb.set_box(vertices)
    # aabb.color = [1,0,0]
    aabb.color = [1,0,0]
    return aabb
    # # Draw the AABB
    # pcd.points = o3d.utility.Vector3dVector(vertices)
    # o3d.visualization.draw_geometries([pcd, aabb])

def get_boxes(path):

    extents = []
    translations = []
    rotations = []
    json_data = load_json(path)
    bbox_list = []
    for label_info in json_data["data"]:
        rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
        transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3).reshape(3,1)
        scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3).reshape(3,1)

        print("rotation, transform, scale", rotation.shape, transform.shape, scale.shape)

        extents.append(scale)
        translations.append(transform)
        rotations.append(np.transpose(rotation))
        box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
        bbox_list.append(box3d)
    # bbox_list = np.asarray(bbox_list)
    bounding_boxes = (extents, rotations, translations)
    return bounding_boxes, bbox_list

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


def transform_hypersim_trajectory(transform_matrices):
    avglen = 0.
    for f in transform_matrices:
        avglen += np.linalg.norm(f[0:3,3])
    avglen /= len(transform_matrices)
    # print("avg camera distance from origin", avglen)

    for f in transform_matrices:
        f[0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    # offsets = np.array([x[:3, 3] for x in transform_matrices])
    # mean_offset = offsets.mean(axis=0)
    # dists = np.linalg.norm(offsets - mean_offset, axis=1)
    # mean_dist = dists.mean()

    # for i in range(len(transform_matrices)):
    #     transform_matrices[i][:3, 3] -= mean_offset
    #     transform_matrices[i][:3, 3] /= (mean_dist / 2)

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in transform_matrices:
        mf = f[0:3,:]
        for g in transform_matrices:
            mg = g[0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.01:
                totp += p*w
                totw += w

    totp /= totw
    # print("looking at", totp) # the cameras are looking at totp
    for f in transform_matrices:
        f[0:3,3] -= totp

    xform = np.eye(4)
    xform[[0, 1, 2], [0, 1, 2]] = 4.0 / avglen
    xform[:3, 3] = -totp
    # print("transformation performed\n", xform)

    return xform, transform_matrices


def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def recenter_poses(poses: np.ndarray):
  """Recenter poses around the origin."""

#   scale = 0.15
#   poses[:, :3, 3] *= scale
  cam2world = average_pose(poses)
  print("cam2world", cam2world.shape)
  transform = np.linalg.inv(pad_poses(cam2world))
  print("transform", transform.shape)
  print("poses 0", poses[0,:,:])
  print("po", poses[0,:3,:4])
  poses = transform @ pad_poses(poses)

  return poses, transform


def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
    print("directions", directions.shape)
    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_camera_frustum(img_size, focal, C2W, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / focal) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / focal) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    # C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def get_rays_mvs(H, W, focal, c2w):
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
    ys, xs = ys.reshape(-1), xs.reshape(-1)

    dirs = torch.stack([(xs-W/2)/focal, (ys-H/2)/focal, torch.ones_like(xs)], -1) # use 1 instead of -1
    rays_d = dirs @ c2w[:3,:3].t() # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)
    print("rays_o", rays_o.shape, rays_d.shape)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def visualize_cameras(all_c2w, sphere_radius, focal, camera_size=0.1, geometry_file=None, geometry_type='mesh', folder=None, bounding_boxes= None, transform = None, scale_factor = None, boxes_list = None, scene_obb = None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    # things_to_draw = []


    # room_bbox = colored_camera_dicts['train'][1]['room_bbox'] 
    # room_bbox = np.array(([-1,-1,-1], [1,1,1]))

    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=room_bbox[0], max_bound=room_bbox[1])
    # line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

    # line_set.paint_uniform_color([1.0, 0.0, 0.0])  # Set color to red
    # things_to_draw.append(line_set)


    # #SAVE new poses only for custom scene

    # # Modify the poses in the loaded JSON data
    # new_poses = []
    # for i in range(len(all_c2w)):
    #     pose = all_c2w[i].tolist()
    #     new_poses.append({
    #         "file_path": poses_dict["frames"][i]["file_path"],
    #         "transform_matrix": pose
    #     })

    # # Create a new dictionary with the modified poses
    # new_transforms_dict = poses_dict.copy()
    # new_transforms_dict["aabb_scale"] = 1.0
    # new_transforms_dict["scale"] = 1.0
    # new_transforms_dict["frames"] = new_poses

    # # Save the modified JSON data to "train_transforms_new.json"
    # with open(os.path.join(folder, "transforms.json"), "w") as file:
    #     json.dump(new_transforms_dict, file, indent=2)



    idx = 0
    fig = pv.figure()

    # for type, camera_dict in new_transforms_dict.items():
    #     print("type", type)
    #     if type =='test':
    #         type = 'val'
    # color, poses_dict = camera_dict
    # poses_dict = new_transforms_dict
    frustums = []
    focal = focal
    # all_c2w = []
    for i in range(len(all_c2w)):
        # C2W = np.array(poses_dict[i].reshape((4, 4)))
        # C2W = np.array(poses_dict['frames'][i]['transform_matrix']).reshape((4, 4))
        C2W = all_c2w[i]
        # all_c2w.append(C2W)
        img_size = (256, 192)
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.2, color=[0,1,0]))
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.2, strict_check=False)
        # refnerf_poses_dict[type] = np.array(all_c2w).tolist()

    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    (extents, orientations, positions) =  bounding_boxes



    # for bbox3d_raw in boxes_list:

    #     bbox_lines = [
    #         [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
    #         [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
    #         [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    #     ]
    #     # Create Open3D PointCloud from the bounding box vertices
    #     bbox_vertices = o3d.utility.Vector3dVector(bbox3d_raw)
    #     bbox_pc = o3d.geometry.PointCloud()
    #     bbox_pc.points = bbox_vertices

    #     # Create Open3D LineSet from the bounding box edges
    #     bbox_lines = o3d.utility.Vector2iVector(bbox_lines)
    #     bbox_ls = o3d.geometry.LineSet(points=bbox_pc.points, lines=bbox_lines)
    #     things_to_draw.append(bbox_ls)


    # aabb = draw_aabb()
    # things_to_draw.append(aabb)
    
    ##DRAW AABB==================================\n

    # scene_obb = None
    if scene_obb is not None:

        obb = calculate_tight_scene_obb(extents, orientations, positions, scene_obb)
        #obb = o3d.geometry.OrientedBoundingBox(center=scene_obb[:3], extent=scene_obb[3:6], R=o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, scene_obb[6]]))
        things_to_draw.append(obb)
    else:
        aabb = estimate_scene_bounding_box(np.copy(all_c2w), bounding_boxes)
        things_to_draw.append(aabb)
    
    for extent, orientation, position in zip(extents, orientations, positions):
        bbox = o3d.geometry.OrientedBoundingBox(center = position, R = orientation, extent=extent)
        bbox.color = [0,1,0]
        things_to_draw.append(bbox)


    # refnerf_poses_dict = json.dumps(refnerf_poses_dict, cls=NumpyEncoder)

    # with open('data.json', 'w') as f:
    #     json.dump(data, f)
    # import pickle
    # with open('refnerf_poses.p', 'wb') as fp:
    #     pickle.dump(refnerf_poses_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # for type, camera_dict in colored_camera_dicts.items():
    #     print("type")
    #     color, poses_dict = camera_dict
    #     idx += 1
    #     cnt = 0
    #     frustums = []
    #     focal = 0.5*800/np.tan(0.5*poses_dict['camera_angle_x'])
    #     all_c2w = []
    #     print("len(len(camera_dict['frames']))", len(poses_dict['frames']))
    #     for i in range(len(poses_dict['frames'])):
    #         C2W = np.array(poses_dict['frames'][i]['transform_matrix']).reshape((4, 4))
    #         all_c2w.append(C2W)
    #         img_size = (800, 800)
    #         frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=1.0, color=color))
    #         cnt += 1
    #     for C2W in all_c2w:
    #         fig.plot_transform(A2B=C2W, s=0.2, strict_check=False)
    
        

    # directions = get_ray_directions(800, 800, focal) # (h, w, 3)
    # print("directions", directions.shape)
    # c2w = convert_pose(all_c2w[0])
    # c2w = torch.FloatTensor(c2w)[:3, :4]
    # rays_o, rays_d = get_rays(directions, c2w)

    # rays_o, rays_d = get_rays_mvs(800, 800, focal, c2w)

    # rays_o = rays_o.numpy()
    # rays_d = rays_d.numpy()
    # for j in range(2500):
    #     start = rays_o[j,:]
    #     end = rays_o[j,:] + rays_d[j,:]*2
    #     line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #     fig.plot(line, c=(1.0, 0.5, 0.0))

    #     start = rays_o[j,:] + rays_d[j,:]*2
    #     end = rays_o[j,:] + rays_d[j,:]*6
    #     line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #     fig.plot(line, c=(0.0, 1.0, 0.0))

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)

            # mesh = trimesh.load(geometry_file)
            # mesh.remove_unreferenced_vertices()

            # mesh.process()

            # mesh.show()

            # # print("mesh", mesh)
            # # transform = mesh.principal_inertia_transform
            # # mesh.apply_transform(transform)

            # mesh.apply_obb()

            # geometry = mesh.as_open3d

            # geometry = geometry.sample_points_uniformly(number_of_points=100000)

            # geometry = geometry.translate(-transform)
            # geometry.scale(scale_factor, center=geometry.get_center())

            #statistical outlier removal
            # cl, ind = geometry.remove_statistical_outlier(nb_neighbors=500,
            #                                                     std_ratio=0.2)
            # geometry = geometry.select_by_index(ind)
            # geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    # mesh_box = geometry.get_oriented_bounding_box()

    # mesh_box.color = [0,1,0]

    # things_to_draw.append(mesh_box)

    # max_bounds = geometry.get_max_bound()
    # min_bounds = geometry.get_min_bound()

    # print("max_bounds, min_bounds", max_bounds, min_bounds)

    # o3d.visualization.draw_geometries(things_to_draw)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    fig.show()

from plyfile import PlyData, PlyElement

def find_minimum_bounding_box(vertices):
    '''
    Find the minimum bounding box of a set of points after projected onto xy plane.
    '''
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    proj = vertices[:, :2]
    box = MinimumBoundingBox(proj)

    center = np.array(box.rectangle_center)
    size = np.array((box.length_parallel, box.length_orthogonal))
    angle = box.unit_vector_angle

    center = np.concatenate([center, [(min_z + max_z) / 2]])
    size = np.concatenate([size, [max_z - min_z]])
    obb = np.concatenate([center, size, [angle]])
    return obb


def load_ply(file_path):
    plydata = PlyData.read(file_path)
    num_verts = plydata['vertex'].count
    vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    vertices[:,0] = plydata['vertex'].data['x']
    vertices[:,1] = plydata['vertex'].data['y']
    vertices[:,2] = plydata['vertex'].data['z']

    vertex_positions = np.array(vertices)
    obb = find_minimum_bounding_box(vertex_positions)
    return obb

if __name__ == '__main__':
    import os

    base_dir = './'
    # folder = 'Downloads/FRONT3D_render/'
    # folder_name = '3dfront_2002_02'

    # folder = "ARKitScenes/data/raw/Training/40776204/train"
    #train_cam_dict = json.load(open(os.path.join(folder, folder_name, 'train/transforms.json')))

    base_path = 'ARKitScenes/data/raw/Training'
    transform_path = 'ARKitScenes/data/ngp_data'
    # folder_name = '40776204'
    folder_name = '40753679'
    # folder_name = '47333462'
    # folder_name = '40777073'

    folder = os.path.join(transform_path, folder_name)
    mesh_folder = os.path.join(base_path, folder_name)
    # folder = "ARKitScenes/data/raw/Training/40753679"

    


    mesh_file_path = os.path.join(mesh_folder, folder_name+'_3dod_mesh.ply')
    box_file_path = os.path.join(mesh_folder, folder_name+'_3dod_annotation.json')

    print("box_file_path", box_file_path)

    bounding_boxes, boxes_list = get_boxes(box_file_path)
    # folder = 'instant-ngp'
    train_cam_dict = json.load(open(os.path.join(folder, 'train','transforms.json')))
    


    #i-ngp transforms
    # translation, scale = find_transforms_center_and_scale(train_cam_dict)
    # train_cam_dict = normalize_transforms(train_cam_dict, translation, scale)


    frustums = []
    focal = 0.5*256/np.tan(0.5*train_cam_dict['camera_angle_x'])
    all_c2w = []
    print("len(len(camera_dict['frames']))", len(train_cam_dict['frames']))
    all_c2w = []
    
    for i in range(len(train_cam_dict['frames'])):
        C2W = np.array(train_cam_dict['frames'][i]['transform_matrix']).reshape((4, 4))
        #only for arkit
        # C2W = arkit_get_pose(C2W)
        all_c2w.append(C2W)
    all_c2w = np.array(all_c2w)



    #nerfstudio transforms
    # all_c2w = torch.from_numpy(all_c2w.astype(np.float32))
    # all_c2w, transform = auto_orient_and_center_poses(all_c2w, center_method='poses')
    # print("transform", transform.shape)
    # scale_factor = 1.0
    # scale_factor /= float(torch.max(torch.abs(all_c2w[:, :3, 3])))
    # all_c2w[:, :3, 3] *= scale_factor
    # all_c2w = all_c2w.numpy()

    #i-ngp transforms
	# translation, scale = find_transforms_center_and_scale(all_c2w)
	# normalized_transforms = normalize_transforms(all_c2w, translation, scale)

    # select = [0,5,10,15,20]
    # all_c2w = all_c2w[select]
    # Modify the poses in the loaded JSON data
    # new_poses = []
    # for i in range(len(all_c2w)):
    #     pose = all_c2w[i].tolist()
    #     new_poses.append({
    #         "file_path": train_cam_dict["frames"][i]["file_path"],
    #         "transform_matrix": pose
    #     })


    # # Create a new dictionary with the modified poses
    # new_transforms_dict = train_cam_dict.copy()
    # new_transforms_dict["aabb_scale"] = 8.0
    # new_transforms_dict["scale"] = 1.0
    # new_transforms_dict["frames"] = new_poses


    # # Save the modified JSON data to "train_transforms_new.json"
    # with open(os.path.join(folder, "train", "transforms.json"), "w") as file:
    #     json.dump(new_transforms_dict, file, indent=2)

    # xform, all_c2w = transform_hypersim_trajectory(all_c2w)


    sphere_radius = 1.
    # train_cam_dict = json.load(open(''))
    #test_cam_dict = json.load(open('mvsnerf/data/nerf_synthetic/nerf_synthetic/hotdog/transforms_test.json'))
    # path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
    camera_size = 0.1
    colored_camera_dicts = {'train': ([0, 1, 0], train_cam_dict)}
    # geometry_file = os.path.join(base_dir, 'mesh_norm.ply')
    # geometry_type = 'mesh'

    obb = load_ply(mesh_file_path)
    translation = None
    scale = None
    print("translation, scale", translation, scale)

    all_c2w = all_c2w[:2]

    visualize_cameras(all_c2w, sphere_radius, focal,
                      camera_size=camera_size, geometry_file=mesh_file_path, geometry_type='mesh', folder=folder, bounding_boxes=bounding_boxes, transform=translation, scale_factor=scale, boxes_list=boxes_list, scene_obb = obb)