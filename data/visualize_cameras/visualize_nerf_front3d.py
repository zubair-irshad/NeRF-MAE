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
# from datasets.google_scanned_utils import *
# import cv2
# from PIL import Image


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

def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh', folder=None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    # things_to_draw = [sphere, coord_frame]

    things_to_draw = []


    # room_bbox = colored_camera_dicts['train'][1]['room_bbox'] 
    room_bbox = np.array(([-1,-1,-1], [1,1,1]))
    
    # np.array([
    #     [-3.2983999252319336, -3.3817999362945557, -0.00021800286776851863],
    #     [0.11159999668598175, 2.8482000827789307, 3.0458500385284424]
    # ])

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=room_bbox[0], max_bound=room_bbox[1])

    # mesh_box = o3d.geometry.TriangleMesh.create_box(width=bbox.get_max_bound()[0]-bbox.get_min_bound()[0],
    #                                                 height=bbox.get_max_bound()[1]-bbox.get_min_bound()[1],
    #                                                 depth=bbox.get_max_bound()[2]-bbox.get_min_bound()[2])
    # # mesh_box.paint_uniform_color([1.0, 0.0, 0.0])  # Set color to red


    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

    line_set.paint_uniform_color([1.0, 0.0, 0.0])  # Set color to red

    # mesh_box.translate((bbox.get_max_bound() + bbox.get_min_bound()))
    # mesh_box.scale(1.0, center=(0, 0, 0))

    things_to_draw.append(line_set)

    pose_dict ={}
    for type, camera_dict in colored_camera_dicts.items():
        print("type")
        color, poses_dict = camera_dict
        cnt = 0
        frustums = []
        focal = 0.5*640/np.tan(0.5*poses_dict['camera_angle_x'])
        all_c2w = []
        print("len(len(camera_dict['frames']))", len(poses_dict['frames']))
        all_c2w = []
        for i in range(len(poses_dict['frames'])):
            C2W = np.array(poses_dict['frames'][i]['transform_matrix']).reshape((4, 4))

            #only for arkit
            C2W = arkit_get_pose(C2W)
            all_c2w.append(C2W)
        all_c2w = np.array(all_c2w)


        print("all_c2w", all_c2w.shape)

        all_c2w = torch.from_numpy(all_c2w.astype(np.float32))
        all_c2w, _ = auto_orient_and_center_poses(all_c2w, center_method='poses')
        scale_factor = 1.0
        scale_factor /= float(torch.max(torch.abs(all_c2w[:, :3, 3])))
        all_c2w[:, :3, 3] *= scale_factor
        all_c2w = all_c2w.numpy()



        # all_c2w, _ = recenter_poses(all_c2w)
        # pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
        # all_c2w[:, :3, 3] *= pose_scale_factor
        pose_dict[type] = [color, all_c2w]


    #SAVE new poses only for custom scene

    # Modify the poses in the loaded JSON data
    new_poses = []
    for i in range(len(all_c2w)):
        pose = all_c2w[i].tolist()
        new_poses.append({
            "file_path": poses_dict["frames"][i]["file_path"],
            "transform_matrix": pose
        })

    # Create a new dictionary with the modified poses
    new_transforms_dict = poses_dict.copy()
    new_transforms_dict["aabb_scale"] = 1.0
    new_transforms_dict["scale"] = 1.0
    new_transforms_dict["frames"] = new_poses

    # Save the modified JSON data to "train_transforms_new.json"
    with open(os.path.join(folder, "transforms.json"), "w") as file:
        json.dump(new_transforms_dict, file, indent=2)



    idx = 0
    fig = pv.figure()
    
    refnerf_poses_dict = {}
    # for type, camera_dict in new_transforms_dict.items():
    #     print("type", type)
    #     if type =='test':
    #         type = 'val'
    # color, poses_dict = camera_dict
    poses_dict = new_transforms_dict
    frustums = []
    focal = focal
    all_c2w = []
    for i in range(len(poses_dict['frames'])):
        # C2W = np.array(poses_dict[i].reshape((4, 4)))
        C2W = np.array(poses_dict['frames'][i]['transform_matrix']).reshape((4, 4))
        all_c2w.append(C2W)
        img_size = (640, 480)
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.05, color=color))
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.05, strict_check=False)
        # refnerf_poses_dict[type] = np.array(all_c2w).tolist()

    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)
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
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    # o3d.visualization.draw_geometries(things_to_draw)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    fig.show()


if __name__ == '__main__':
    import os

    base_dir = './'
    folder = '/home/zubairirshad/Downloads/FRONT3D_render/'
    folder_name = '3dfront_2002_02'

    # folder = "/home/zubairirshad/ARKitScenes/data/raw/Training/40776204/train"
    #train_cam_dict = json.load(open(os.path.join(folder, folder_name, 'train/transforms.json')))

    # folder = "/home/zubairirshad/Downloads/record3d_2/EXR_RGBD/train"
    # folder = '/home/zubairirshad/instant-ngp'
    train_cam_dict = json.load(open(os.path.join(folder, folder_name,'train/transforms.json')))

    sphere_radius = 1.
    # train_cam_dict = json.load(open(''))
    #test_cam_dict = json.load(open('/home/zubairirshad/mvsnerf/data/nerf_synthetic/nerf_synthetic/hotdog/transforms_test.json'))
    # path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
    camera_size = 0.1

    #subsample every 8 frames
    train_cam_dict['frames'] = train_cam_dict['frames'][::8]

    # train_cam_dict = train_cam_dict[::8]
    # train_cam_dict = train_cam_dict[]

    colored_camera_dicts = {'train': ([0, 1, 0], train_cam_dict)}
    # geometry_file = os.path.join(base_dir, 'mesh_norm.ply')
    # geometry_type = 'mesh'

    visualize_cameras(colored_camera_dicts, sphere_radius, 
                      camera_size=camera_size, geometry_file=None, geometry_type=None, folder=folder)