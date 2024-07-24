import torch
import numpy as np


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_l = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_l)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.reshape(1, grid_size, grid_size, grid_size, embed_dim)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W, D/2)
    emb_l = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w, emb_l], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def pad_tensor(tensor, target_shape, pad_value):
    """
    Pad pytorch tensor of shpae 1, H,W,L, 3 pad it to a size of 1, target_shape[0],target_shape[1],target_shape[2],3 with zeros in pytorch and give me the resulting mask of non-padded values
    """
    mask = torch.ones(tensor.shape)
    # tensor = tensor.permute(3, 0, 1, 2)
    tensor = torch.nn.functional.pad(
        tensor,
        (
            0,
            target_shape[0] - tensor.shape[3],
            0,
            target_shape[1] - tensor.shape[2],
            0,
            target_shape[2] - tensor.shape[1],
        ),
        mode="constant",
        value=pad_value,
    )

    mask = torch.nn.functional.pad(
        mask,
        (
            0,
            target_shape[0] - mask.shape[3],
            0,
            target_shape[1] - mask.shape[2],
            0,
            target_shape[2] - mask.shape[1],
        ),
        mode="constant",
        value=0,
    )
    # mask = mask == 1
    return tensor.unsqueeze(0), mask.unsqueeze(0)
