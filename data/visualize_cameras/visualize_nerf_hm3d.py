import open3d as o3d
import json
import numpy as np
import json

import torch
from kornia import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv

# import sys
# sys.path.append('/home/zubairirshad/NeRF_MAE_internal')
from transform_utils import *
import pandas as pd
import h5py


# from datasets.google_scanned_utils import *
# import cv2
# from PIL import Image

excluded_labels_nyu40 = [
    'wall',
    'floor',
    'door',
    'window',
    'counter',
    'shelves',
    'curtain',
    'books',
    'refrigerator',
    'television',
    'paper',
    'towel',
    'box',
    'whiteboard',
    'person',
    'night stand',
    'sink',
    'bag',
    'bookshelf',
    'otherprop',
    'pillow',
    # 'lamp'
]

hm3d_to_mp3d_path = "/home/zubairirshad/NeRF_MAE_internal/data/hm3d/matterport_category_mappings.tsv"
df = pd.read_csv(hm3d_to_mp3d_path, sep="    ", header=0, engine="python")
hm3d_to_nyu40 = {row["category"]: row["nyu40id"] for _, row in df.iterrows()}
nyu40_id_label = {1: 'wall', 8: 'door', 22: 'ceiling', 2: 'floor', 11: 'picture', 9: 'window', 5: 'chair', 0: 'void', 18: 'pillow', 40: 'otherprop', 35: 'lamp', 3: 'cabinet', 16: 'curtain', 7: 'table', 19: 'mirror', 27: 'towel', 34: 'sink', 15: 'shelves', 6: 'sofa', 4: 'bed', 32: 'night stand', 33: 'toilet', 38: 'otherstructure', 25: 'television', 14: 'desk', 29: 'box', 39: 'otherfurniture', 12: 'counter', 21: 'clothes', 36: 'bathtub', 23: 'books', 17: 'dresser', 24: 'refridgerator', 10: 'bookshelf', 28: 'shower curtain', 13: 'blinds', 20: 'floor mat', 37: 'bag', 30: 'whiteboard', 26: 'paper', 31: 'person'}


from viz_utils import *

def get_bounding_boxes(mesh_path, transform):
    extents_file = h5py.File(os.path.join(mesh_path, 'metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5'), 'r')
    orientation_file = h5py.File(os.path.join(mesh_path, 'metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5'), 'r')
    pos_file = h5py.File(os.path.join(mesh_path, 'metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5'), 'r')

    extents_data = extents_file['dataset']
    orientation_data = orientation_file['dataset']
    pos_data = pos_file['dataset']

    extents = np.array(extents_data)
    orientation = np.array(orientation_data)
    pos = np.array(pos_data)

    ext_inf_filter = np.isinf(extents)
    ori_inf_filter = np.isinf(orientation)
    
    extents = extents[~ext_inf_filter].reshape(-1, 3)
    orientation = orientation[~ori_inf_filter].reshape(-1, 3, 3)
    pos = pos[~ext_inf_filter].reshape(-1, 3)

    assert extents.shape[0] == orientation.shape[0] == pos.shape[0]

    # extents *= transform[[0, 1, 2], [0, 1, 2]]
    # pos = np.matmul(pos, transform[:3, :3].T) + transform[:3, 3]

    return extents, orientation, pos

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

def get_hypersim_intrinsics(metadata_path, scene_name):
    metadata_df = pd.read_csv(metadata_path)
    params = metadata_df[metadata_df['scene_name'] == scene_name]
    fov_x = params['camera_physical_fov'].values[0]
    if np.isnan(fov_x):
        fov_x = 1.0471975803375244      # default

    height = params['settings_output_img_height'].values[0]
    width = params['settings_output_img_width'].values[0]
    focal_length = width / (2 * np.tan(fov_x / 2))
    fov_y = 2 * np.arctan(height / (2 * focal_length))

    return height, width, fov_y, fov_x, focal_length


def load_hypersim_trajectory(camera_path):
    orientation_filename = os.path.join(camera_path, 'camera_keyframe_orientations.hdf5')
    pos_filename = os.path.join(camera_path, 'camera_keyframe_positions.hdf5')

    orientation_file = h5py.File(orientation_filename, 'r')
    pos_file = h5py.File(pos_filename, 'r')

    orientation_data = orientation_file['dataset']
    pos_data = pos_file['dataset']

    transform_matrices = []
    for i in range(orientation_data.shape[0]):
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = orientation_data[i]
        affine_matrix[:3, 3] = pos_data[i]
        transform_matrices.append(affine_matrix)

    return transform_matrices


def create_json_from_hypersim(scene_path, metadata_path):
    scene_name = os.path.basename(scene_path)
    # os.makedirs(output_path, exist_ok=True)

    camera_metadata = pd.read_csv(os.path.join(scene_path, '_detail', 'metadata_cameras.csv'))
    camera_names = camera_metadata['camera_name'].tolist()
    camera_names = [c for c in camera_names 
                    if os.path.exists(os.path.join(scene_path, '_detail', c)) 
                    and os.path.exists(os.path.join(scene_path, 'images', 'scene_{}_final_preview'.format(c)))]

    camera_paths = []
    img_paths = []
    for camera in camera_names:
        camera_path = os.path.join(scene_path, '_detail', camera)
        img_path = os.path.join(scene_path, 'images', 'scene_{}_final_preview'.format(camera))
        if os.path.exists(camera_path) and os.path.exists(img_path):
            camera_paths.append(camera_path)
            img_paths.append(img_path)

    transform_matrices = []
    num_xforms = []
    # img_names = []

    for i in range(len(camera_names)):
        # imgs = collect_hypersim_images(img_paths[i], camera_names[i] + '.', 
        #                                 output_path=os.path.join(output_path, 'images'))

        xforms = load_hypersim_trajectory(camera_paths[i])

        num_xforms.append(len(transform_matrices))
        transform_matrices += xforms
        # img_names += imgs

    transform_matrices = np.array(transform_matrices)
    print("scene_name", scene_name)
    height, width, fov_y, fov_x, focal_length = get_hypersim_intrinsics(metadata_path, scene_name)

    return transform_matrices, focal_length, height, width, fov_y, fov_x



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

def visualize_cameras(colored_camera_dicts, sphere_radius, focal, bounding_boxes, camera_size=0.1, geometry_file=None, geometry_type='mesh', folder=None, viz_boxes=False):
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

    # mesh_box.translate((bbox.get_max_bound() + bbox.get_min_bound()))
    # mesh_box.scale(1.0, center=(0, 0, 0))
    # things_to_draw.append(line_set)

    pose_dict ={}
    for type, camera_dict in colored_camera_dicts.items():
        print("type")
        color, poses_dict = camera_dict
        cnt = 0
        frustums = []
        # focal = 0.5*640/np.tan(0.5*poses_dict['camera_angle_x'])
        focal = focal
        all_c2w = []
        # print("len(len(camera_dict['frames']))", len(poses_dict['frames']))
        all_c2w = []
        # for i in range(len(poses_dict['frames'])):
        for i in range(len(poses_dict)):
            C2W = np.array(poses_dict[i])

            #only for arkit
            # C2W = arkit_get_pose(C2W)
            all_c2w.append(C2W)
        all_c2w = np.array(all_c2w)


        print("all_c2w", all_c2w.shape)
        all_c2w = torch.from_numpy(all_c2w.astype(np.float32))

        # all_c2w, _ = auto_orient_and_center_poses(all_c2w, center_method='poses')
        # scale_factor = 1.0
        # scale_factor /= float(torch.max(torch.abs(all_c2w[:, :3, 3])))
        # all_c2w[:, :3, 3] *= scale_factor
        # all_c2w = all_c2w.numpy()



        # all_c2w, _ = recenter_poses(all_c2w)
        # pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
        # all_c2w[:, :3, 3] *= pose_scale_factor
        pose_dict[type] = [color, all_c2w]


    #SAVE new poses only for custom scene

    # Modify the poses in the loaded JSON data
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
    
    refnerf_poses_dict = {}
    # for type, camera_dict in new_transforms_dict.items():
    #     print("type", type)
    #     if type =='test':
    #         type = 'val'
    # color, poses_dict = camera_dict
    # poses_dict = new_transforms_dict
    
    frustums = []
    focal = focal
    all_c2w = []
    for i in range(len(poses_dict)):
        # C2W = np.array(poses_dict[i].reshape((4, 4)))
        C2W = np.array(poses_dict[i])
        all_c2w.append(C2W)
        img_size = (512, 512)
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.2, color=color))
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.2, strict_check=False)
        # refnerf_poses_dict[type] = np.array(all_c2w).tolist()

    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    unitcube = unit_cube()
    for k in range(len(unitcube)):
        fig.add_geometry(unitcube[k]) 
        
    #draw bounding boxes

    if viz_boxes:
        (extents, orientations, positions) =  bounding_boxes
        
        for extent, orientation, position in zip(extents, orientations, positions):
            bbox = o3d.geometry.OrientedBoundingBox(center = position, R = orientation, extent=extent)
            things_to_draw.append(bbox)

        aabb = calculate_scene_obb_from_objects(bounding_boxes)
        aabb.color = (1, 0, 0)
        things_to_draw.append(aabb)
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

    # o3d.visualization.draw_geometries(things_to_draw)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    fig.show()


def transform_poses_pca(poses: np.ndarray):
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  # Just make sure it's it in the [-1, 1]^3 cube
  scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  poses_recentered[:, :3, 3] *= scale_factor
  transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  return poses_recentered, transform


def transform_bounding_boxes(bounding_boxes, transform_matrix):
    extents, orientations, positions = bounding_boxes
    new_positions = []
    new_extents = []
    new_orientations = []
    for extent, orientation, position in zip(extents, orientations, positions):
        bbox = o3d.geometry.OrientedBoundingBox(center = position, R = orientation, extent=extent)
        # bbox_transformed = bbox.transform(transform_matrix)

        bbox.rotate(transform_matrix[:3, :3], center=(0, 0, 0))
        bbox.translate(transform_matrix[:3, 3])

        new_positions.append(bbox.center)
        new_extents.append(bbox.extent)
        new_orientations.append(bbox.R)

    bounding_boxes = (new_extents, new_orientations, new_positions)

    return bounding_boxes


def calculate_scene_obb_from_objects(bounding_boxes):
    # Calculate the minimum and maximum coordinates of all object OBBs

    extents, orientations, positions = bounding_boxes
    min_coords = np.inf
    max_coords = -np.inf

    for extent, orientation, position in zip(extents, orientations, positions):
        obb = o3d.geometry.OrientedBoundingBox(center=position, R=orientation, extent=extent)
        obb_points = np.asarray(obb.get_box_points())

        min_coords = np.minimum(min_coords, obb_points.min(axis=0))
        max_coords = np.maximum(max_coords, obb_points.max(axis=0))

    print("min_coords", min_coords)
    print("max_coords", max_coords)
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coords, max_bound=max_coords)
    return aabb

def get_boxes(box_file_path, filter_by_size=True, min_size=6.5, grid_res=160):
    extents = []
    rotations = []
    translations = []
    # Load the JSON data from the bounding box file
    with open(box_file_path, 'r') as bbox_file:
        bbox_data = json.load(bbox_file)
        
    bounding_boxes = []
    # excluded_classes = ['mirror', 'door', 'pillow', 'fan', 'light', 'closet']
    # excluded_classes = ['bed', 'pillow']
    for bbox_info in bbox_data:
        class_name = bbox_info['class_name']
        # print("class_name", class_name)
        # if class_name not in excluded_classes:
        #     continue

        class_name_nyu40  = hm3d_to_nyu40.get(class_name, None)
        nyu40_label = nyu40_id_label.get(int(class_name_nyu40), None)
        
        if nyu40_label in excluded_labels_nyu40:
            continue
        else:
            print("nyu40_label", nyu40_label)

        bbox = bbox_info['bbox']

        min_pt = bbox[0]
        max_pt = bbox[1]

        bbox_min = np.min(min_pt, axis=0)
        bbox_max = np.max(max_pt, axis=0)
        diag = bbox_max - bbox_min
        extent = np.array(bbox[1])-np.array(bbox[0])
        extent = extent / diag * grid_res
        print("extent", extent)
        if filter_by_size and (extent < min_size).any():
            continue

        

        
        #Let's do xzy instead of xyz
        min_pt[1], min_pt[2] = min_pt[2], min_pt[1]
        max_pt[1], max_pt[2] = max_pt[2], max_pt[1]
        bbox[0] = min_pt
        bbox[1] = max_pt
        # obj_aabb = obj_dict['aabb']

        extents.append(np.array(bbox[1])-np.array(bbox[0]))
        rotations.append(np.eye(3))
        translations.append((np.array(bbox[0])+np.array(bbox[1]))/2.0)

    return extents, rotations, translations


if __name__ == '__main__':
    import os
    import glob
    # base_path = '/home/zubairirshad/Downloads/ai_002_005'

    instance_name = '00891-cvZr5TUy5C5_6'
    # pose_folder = os.path.join('/home/zubairirshad/Downloads/masked_rdp_2', instance_name)

    pose_folder = os.path.join('/home/zubairirshad/Downloads/hm3d_raw', instance_name)
    # pose_folder = '/home/zubairirshad/Downloads/masked_rdp_2/00891-cvZr5TUy5C5_6'

    all_poses = []
    pose_path = os.path.join(pose_folder, 'train', 'transforms.json')
    pose_list = glob.glob(os.path.join(pose_folder, "*.json"))


    # with open(pose_path, 'r') as f:
    #     json_dict = json.load(f)

    # Sort the list by the numeric part of the filenames

    # with open(pose_path, 'r') as f:
    #     json_dict = json.load(f)

    # for i in range(len(json_dict['frames'])):
    #     pose = np.array(json_dict['frames'][i]['transform_matrix']).reshape((4, 4))
    #     all_poses.append(pose)


    pose_list = sorted(pose_list, key=lambda x: int(os.path.basename(x).split('.')[0]))

    print(os.path.basename(pose_list[0]).split('.')[0])
    print("pose_list", pose_list[6])

    # print("pose_list", pose_list)
    for pos_file in pose_list:
        pos = np.array(json.load(open(pos_file))["pose"]).astype(np.float32)
        all_poses.append(pos)

    focal = 256.0 / np.tan(np.deg2rad(90.0) / 2)

    # all_poses = all_poses[::10]

    all_poses = np.array(all_poses)

    print("all_poses", all_poses.shape)
    # all_poses = np.expand_dims(all_poses[6, :, :], axis=0)


    # print("all_poses", all_poses.shape)
    # all_poses, transform_matrix = transform_poses_pca(all_poses)
    # pose_shape = all_poses.shape[0]
    # all_poses_homogeneous = np.zeros((pose_shape, 4, 4))
    # all_poses_homogeneous[:, :3, :4] = all_poses
    # all_poses_homogeneous[:, 3, 3] = 1.0
    # all_poses = all_poses_homogeneous




    # print("all_poses", all_poses.shape)
    # scene_path = base_path
    # metadata_path = os.path.join(scene_path, 'metadata_camera_parameters.csv')

    # transform_matrices, focal_length, height, width, fov_y, fov_x =  create_json_from_hypersim(scene_path, metadata_path)
    # #xform, transform_matrices = transform_hypersim_trajectory(transform_matrices)
    # xform = None
    # mesh_path = os.path.join(scene_path, '_detail', 'mesh')
    # extents, orientation, pos = get_bounding_boxes(mesh_path, xform)
    # bounding_boxes = (extents, orientation, pos)
    

    box_file_path = os.path.join('/home/zubairirshad/Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis', instance_name + '.json')
    # bbox_file_path = "/home/zubairirshad/Downloads/objects_bboxes_per_room/new_single_room_bboxes_replace_nofilterdetected_all_concepts_replace_revised_axis/00891-cvZr5TUy5C5_6.json"


    extents, rotations, translations = get_boxes(box_file_path)
        # obj_bbox_ngp = {
        #     "extents": (np.array(bbox[1])-np.array(bbox[0])).tolist(),
        #     "orientation": np.eye(3).tolist(),
        #     "position": ((np.array(bbox[0])+np.array(bbox[1]))/2.0).tolist(),
        # }
        # bounding_boxes.append(obj_bbox_ngp)
        # bounding_boxes.append([bbox[0], bbox[1]])

    print("total objects", len(extents))
    bounding_boxes = (extents, rotations, translations)

    # print("transform_matrix", transform_matrix.shape)
    # bounding_boxes = transform_bounding_boxes(bounding_boxes, transform_matrix)

    sphere_radius = 1.
    camera_size = 0.1
    colored_camera_dicts = {'train': ([0, 1, 0], all_poses)}
    # geometry_file = os.path.join(base_dir, 'mesh_norm.ply')
    # geometry_type = 'mesh'

    visualize_cameras(colored_camera_dicts, sphere_radius, focal,bounding_boxes=bounding_boxes,
                      camera_size=camera_size, geometry_file=None, geometry_type=None, viz_boxes=True)