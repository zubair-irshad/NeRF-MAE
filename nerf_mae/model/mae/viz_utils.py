import colorsys
import numpy as np

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt
import random

def generate_point_cloud_mask(point_cloud):
    point_cloud = point_cloud.numpy()  # Convert tensor to NumPy array
    mask = np.logical_and.reduce((point_cloud[:, 0] >= 0, point_cloud[:, 0] <= 1,
                                  point_cloud[:, 1] >= 0, point_cloud[:, 1] <= 0.8,
                                  point_cloud[:, 2] >= 0, point_cloud[:, 2] <= 1))
    return mask


# def custom_draw_geometry_with_key_callback(pcd):
#     def change_background_to_black(vis):
#         opt = vis.get_render_option()
#         opt.background_color = np.asarray([0, 0, 0])
#         return False

#     def load_render_option(vis):
#         vis.get_render_option().load_from_json("../../TestData/renderoption.json")
#         return False

#     def capture_depth(vis):
#         depth = vis.capture_depth_float_buffer()
#         plt.imshow(np.asarray(depth))
#         plt.show()
#         return False

#     def capture_image(vis):
#         image = vis.capture_screen_float_buffer()
#         plt.imshow(np.asarray(image))
#         plt.show()
#         return False

#     def rotate_view(vis):
#         ctr = vis.get_view_control()
#         ctr.rotate(1.0, 0.0)
#         return False

#     key_to_callback = {}
#     key_to_callback[ord("K")] = change_background_to_black
#     key_to_callback[ord("R")] = load_render_option
#     key_to_callback[ord(",")] = capture_depth
#     key_to_callback[ord(".")] = capture_image
#     key_to_callback[ord("O")] = rotate_view

#     o3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)


# def custom_draw_geometry_with_rotation(pcd):
#     def rotate_view(vis):
#         ctr = vis.get_view_control()
#         ctr.rotate(10.0, 0.0)
#         return False

#     o3d.visualization.draw_geometries_with_animation_callback(pcd, rotate_view)


def draw_grid_colors(grid, colors=None, coordinate_system = False):
    pcds = []
    grid_pts = o3d.utility.Vector3dVector(grid)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # o3d_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
    #     pcd.points
    # ).get_box_points()

    o3d_bbox = pcd.get_axis_aligned_bounding_box().get_box_points()

    # print("o3d_bbox", np.array(o3d_bbox))

    cylinder_segments = line_set(np.array(o3d_bbox))
    for i in range(len(cylinder_segments)):
        pcds.append(cylinder_segments[i])

    if coordinate_system:
        mesh =  o3d.geometry.TriangleMesh.create_coordinate_frame()
        pcds.append(mesh)
    pcds.append(pcd)
    # custom_draw_geometry_with_rotation(pcds)
    o3d.visualization.draw_geometries(pcds)

def window_masking_3d(
    x,
    patch_size=(4, 4, 4),
    p_remove=0.50,
    mask_token=None,
    sampling_strategy="random",
):
    batch_size, height, width, depth, channels = x.shape
    mask = torch.zeros((batch_size, height, width, depth, 1)).to(x.device)
    patch_h, patch_w, patch_d = patch_size

    patch_h, patch_w, patch_d = patch_size
    # num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
    # num_masked_patches = 0

    if sampling_strategy == "grid":
        # Calculate the number of patches in each dimension
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        num_patches_d = depth // patch_d

        # Calculate the total number of patches
        num_patches = num_patches_h * num_patches_w * num_patches_d

        # Create a list of all patch indices
        all_patch_indices = [
            (h, w, d)
            for h in range(num_patches_h)
            for w in range(num_patches_w)
            for d in range(num_patches_d)
        ]

        # Iterate through the shuffled patch indices and keep only 'num_to_keep' patches
        num_masked_patches = 0
        count = 0
        for h, w, d in all_patch_indices:
            # we want to mask one out of every four patches

            if count == 0 or count == 1 or count == 2:
                h_start, h_end = h * patch_h, (h + 1) * patch_h
                w_start, w_end = w * patch_w, (w + 1) * patch_w
                d_start, d_end = d * patch_d, (d + 1) * patch_d
                mask[:, h_start:h_end, w_start:w_end, d_start:d_end, :] = 1
                num_masked_patches += 1
                count += 1
            else:
                count += 1
                if count == 4:
                    count = 0

    elif sampling_strategy == "block":
        # mask our 25% of the volume 3 times at random starting and ending locations
        # Calculate the number of patches in each dimension
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        num_patches_d = depth // patch_d

        # Calculate the total number of patches
        num_patches = num_patches_h * num_patches_w * num_patches_d

        # Create a list of all patch indices
        all_patch_indices = [
            (h, w, d)
            for h in range(num_patches_h)
            for w in range(num_patches_w)
            for d in range(num_patches_d)
        ]

        num_to_keep = num_patches // 4  # Keep one out of every four patches
        total_masked_patches = 0
        for i in range(3):
            num_masked_patches = 0
            h_start = height // patch_h
            w_start = width // patch_w
            d_start = depth // patch_d

            h_start = np.random.randint(0, h_start - 0.25 * h_start)

            for h, w, d in all_patch_indices:
                # select random strting locations for the 25% of the volume
                if h > h_start:
                    h_start_new, h_end = h * patch_h, (h + 1) * patch_h
                    w_start_new, w_end = w * patch_w, (w + 1) * patch_w
                    d_start_new, d_end = d * patch_d, (d + 1) * patch_d

                    # check if the patch is already masked, if not mask it
                    if (
                        mask[
                            :,
                            h_start_new:h_end,
                            w_start_new:w_end,
                            d_start_new:d_end,
                            :,
                        ].sum()
                        == 0
                    ):
                        mask[
                            :,
                            h_start_new:h_end,
                            w_start_new:w_end,
                            d_start_new:d_end,
                            :,
                        ] = 1
                        num_masked_patches += 1

                if num_masked_patches >= num_to_keep:
                    total_masked_patches += num_masked_patches
                    num_masked_patches = 0

                    break  # Stop masking patches once we've reached the desired number
        num_masked_patches = total_masked_patches

    elif sampling_strategy == "random":
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        num_patches_d = depth // patch_d

        # Calculate the total number of patches
        num_patches = num_patches_h * num_patches_w * num_patches_d

        num_masked_patches = 0
        for h in range(0, height - patch_h + 1, patch_h):
            for w in range(0, width - patch_w + 1, patch_w):
                for d in range(0, depth - patch_d + 1, patch_d):
                    if random.random() < p_remove:
                        num_masked_patches += 1
                        mask[
                            :, h : h + patch_h, w : w + patch_w, d : d + patch_d, :
                        ] = 1

    masked_x = x.clone()
    masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
    if mask_token is not None:
        mask_token = mask_token.to(masked_x.device)
        index_mask = (mask.bool()).squeeze(-1)
        masked_x[index_mask, :] = mask_token

    # percent_masked = 100 * num_masked_patches / num_patches
    # print(f"Total number of patches: {num_patches}")
    # print(f"Total number of masked patches: {num_masked_patches}")
    # print(f"Percentage of patches that are masked: {percent_masked:.2f}%")

    return masked_x, mask
    

# def window_masking(
#     x: torch.Tensor,
#     r: int = 4,
#     remove: bool = False,
#     mask_len_sparse: bool = False,
# ):
#     embed_dim = 3
#     mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#     """
#     The new masking method, masking the adjacent r*r number of patches together

#     Optional whether to remove the mask patch,
#     if so, the return value returns one more sparse_restore for restoring the order to x

#     Optionally, the returned mask index is sparse length or original length,
#     which corresponds to the different size choices of the decoder when restoring the image

#     x: [N, L, D]
#     r: There are r*r patches in a window
#     remove: Whether to remove the mask patch
#     mask_len_sparse: Whether the returned mask length is a sparse short length
#     """
#     mask_ratio = 0.50
#     x = rearrange(x, "B H W C -> B (H W) C")
#     print("x", x.shape)
#     B, L, D = x.shape
#     assert int(L**0.5 / r) == L**0.5 / r
#     d = int(L**0.5 // r)

#     noise = torch.rand(B, d**2, device=x.device)
#     print("noise", noise.shape)
#     sparse_shuffle = torch.argsort(noise, dim=1)
#     sparse_restore = torch.argsort(sparse_shuffle, dim=1)
#     sparse_keep = sparse_shuffle[:, : int(d**2 * (1 - mask_ratio))]

#     index_keep_part = (
#         torch.div(sparse_keep, d, rounding_mode="floor") * d * r**2
#         + sparse_keep % d * r
#     )
#     index_keep = index_keep_part
#     for i in range(r):
#         for j in range(r):
#             if i == 0 and j == 0:
#                 continue
#             index_keep = torch.cat(
#                 [index_keep, index_keep_part + int(L**0.5) * i + j], dim=1
#             )

#     index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
#     index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
#     for i in range(B):
#         index_mask[i] = np.setdiff1d(
#             index_all[i], index_keep.cpu().numpy()[i], assume_unique=True
#         )
#     index_mask = torch.tensor(index_mask, device=x.device)
#     index_shuffle = torch.cat([index_keep, index_mask], dim=1)
#     index_restore = torch.argsort(index_shuffle, dim=1)

#     if mask_len_sparse:
#         mask = torch.ones([B, d**2], device=x.device)
#         mask[:, : sparse_keep.shape[-1]] = 0
#         mask = torch.gather(mask, dim=1, index=sparse_restore)
#     else:
#         mask = torch.ones([B, L], device=x.device)
#         mask[:, : index_keep.shape[-1]] = 0
#         mask = torch.gather(mask, dim=1, index=index_restore)

#     if remove:
#         x_masked = torch.gather(
#             x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D)
#         )
#         x_masked = rearrange(
#             x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5)
#         )
#         return x_masked, mask, sparse_restore
#     else:
#         x_masked = torch.clone(x)
#         for i in range(B):
#             x_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
#         x_masked = rearrange(
#             x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5)
#         )
#         return x_masked, mask


# import random


def window_masking_3d(x, patch_size=(4, 4, 4), p_remove=0.50, mask_token=None):
    batch_size, height, width, depth, channels = x.shape
    mask = torch.ones((batch_size, height, width, depth, 1)).to(x.device)
    patch_h, patch_w, patch_d = patch_size

    patch_h, patch_w, patch_d = patch_size
    num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
    num_masked_patches = 0

    # p_keep = 1 - p_remove
    for h in range(0, height - patch_h + 1, patch_h):
        for w in range(0, width - patch_w + 1, patch_w):
            for d in range(0, depth - patch_d + 1, patch_d):
                if random.random() < p_remove:
                    num_masked_patches += 1
                    mask[:, h : h + patch_h, w : w + patch_w, d : d + patch_d, :] = 0

    masked_x = x.clone()
    masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
    if mask_token is not None:
        mask_token = mask_token.to(masked_x.device)
        index_mask = (mask.bool()).squeeze(-1)
        masked_x[index_mask, :] = mask_token

    percent_masked = 100 * num_masked_patches / num_patches
    print(f"Total number of patches: {num_patches}")
    print(f"Total number of masked patches: {num_masked_patches}")
    print(f"Percentage of patches that are masked: {percent_masked:.2f}%")

    return masked_x, mask


# def window_masking_3d(x, patch_size=(4, 4, 4), p=0.25, mask_token=None):
#     batch_size, height, width, depth, channels = x.shape
#     mask = torch.zeros((batch_size, height, width, depth, 1))
#     patch_h, patch_w, patch_d = patch_size

#     patch_h, patch_w, patch_d = patch_size
#     num_patches = (height // patch_h) * (width // patch_w) * (depth // patch_d)
#     num_masked_patches = 0

#     for h in range(0, height - patch_h + 1, patch_h):
#         for w in range(0, width - patch_w + 1, patch_w):
#             for d in range(0, depth - patch_d + 1, patch_d):
#                 if random.random() < p:
#                     num_masked_patches += 1
#                     mask[:, h : h + patch_h, w : w + patch_w, d : d + patch_d, :] = 1

#     masked_x = x.clone()
#     masked_x[mask.bool().expand(-1, -1, -1, -1, channels)] = 0
#     if mask_token is not None:
#         mask_token = mask_token.to(masked_x.device)
#         index_mask = (mask.bool()).squeeze(-1)
#         masked_x[index_mask, :] = mask_token

#     percent_masked = 100 * num_masked_patches / num_patches
#     print(f"Total number of patches: {num_patches}")
#     print(f"Total number of masked patches: {num_masked_patches}")
#     print(f"Percentage of patches that are masked: {percent_masked:.2f}%")

#     return masked_x


def construct_grid(res):
    res_x, res_y, res_z = res
    x = torch.linspace(0, res_x - 1, res_x) / max(res)
    y = torch.linspace(0, res_y - 1, res_y) / max(res)
    z = torch.linspace(0, res_z - 1, res_z) / max(res)

    # Shift by 0.5 voxel
    x += 0.5 / max(res)
    y += 0.5 / max(res)
    z += 0.5 / max(res)

    # Construct grid using broadcasting
    xx = x.view(res_x, 1, 1).repeat(1, res_y, res_z).flatten()
    yy = y.view(1, res_y, 1).repeat(res_x, 1, res_z).flatten()
    zz = z.view(1, 1, res_z).repeat(res_x, res_y, 1).flatten()
    grid = torch.stack((xx, yy, zz), dim=1)

    return grid


# def construct_grid(res):
#     res_x, res_y, res_z = res
#     x = torch.linspace(0, res_x, res_x)
#     y = torch.linspace(0, res_y, res_y)
#     z = torch.linspace(0, res_z, res_z)

#     scale = torch.tensor(res).max()
#     x /= scale
#     y /= scale
#     z /= scale

#     # Shift by 0.5 voxel
#     x += 0.5 * (1.0 / scale)
#     y += 0.5 * (1.0 / scale)
#     z += 0.5 * (1.0 / scale)

#     grid = []
#     for i in range(res_z):
#         for j in range(res_y):
#             for k in range(res_x):
#                 grid.append([x[k], y[j], z[i]])

#     return torch.tensor(grid)


# def construct_grid(res):
#     res_x, res_y, res_z = res
#     x = np.linspace(0, res_x, res_x)
#     y = np.linspace(0, res_y, res_y)
#     z = np.linspace(0, res_z, res_z)

#     scale = res.max()
#     x /= scale
#     y /= scale
#     z /= scale

#     # Shift by 0.5 voxel
#     x += 0.5 * (1.0 / scale)
#     y += 0.5 * (1.0 / scale)
#     z += 0.5 * (1.0 / scale)

#     grid = []
#     for i in range(res_x):
#         for j in range(res_y):
#             for k in range(res_z):
#                 grid.append([x[i], y[j], z[k]])

#     return np.array(grid)


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)

def density_to_alpha_scannet(density):
    # ScanNet uses dense depth priors NeRF, which uses ReLU activation
    activation = np.clip(density, a_min=0, a_max=None)
    return np.clip(1.0 - np.exp(-activation / 100.0), 0.0, 1.0)


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
                points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
                lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
                colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
                radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = (
            np.array(lines)
            if lines is not None
            else self.lines_from_ordered_points(self.points)
        )
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length
            )
            cylinder_segment = cylinder_segment.translate(translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                    center=cylinder_segment.get_center(),
                )
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors


def line_set(points_array):
    open_3d_lines = [
        [0, 1],
        [0, 3],
        [2, 0],
        [3, 5],
        [2, 5],
        [1, 6],
        [3, 6],
        [6, 4],
        [2, 7],
        [5, 4],
        [4, 7],
        [1, 7],
    ]

    colors = [[0, 0, 0.5] for i in range(len(open_3d_lines))]
    # colors = random_colors(len(open_3d_lines))
    open_3d_lines = np.array(open_3d_lines)
    line_set = LineMesh(points_array, open_3d_lines, colors=colors, radius=0.002)
    line_set = line_set.cylinder_segments
    return line_set
