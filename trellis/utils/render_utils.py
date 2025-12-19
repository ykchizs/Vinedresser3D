import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image
import imageio
import os

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, return_types=['color', 'normal'])
            if 'normal' not in rets: rets['normal'] = []
            if 'color' not in rets: rets['color'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        else:
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)

def Trellis_render_multiview_images(sample, yaws, pitch):
    resolution, bg_color, r, fov = 512, (0, 0, 0), 2, 40
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color})

def Seg_render_multiview_imgs_pc(points, labels=None, save_path=None, name=None):

    """Create and save 8 different views of the point cloud"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Get axis limits from points
    margin = 0.1  # Add 10% margin
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals
    
    ax.set_xlim(min_vals[0] - margin*range_vals[0], max_vals[0] + margin*range_vals[0])
    ax.set_ylim(min_vals[1] - margin*range_vals[1], max_vals[1] + margin*range_vals[1])
    ax.set_zlim(min_vals[2] - margin*range_vals[2], max_vals[2] + margin*range_vals[2])

    # Plot the points with colors based on labels
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                c=labels, cmap='rainbow', marker='.', s=1, alpha=0.7)

    # Create 8 different views (45 degrees apart)
    elevs = [20, -20, 20, -20, 20, -20, 20, -20]
    azims = [45, 45, 135, 135, 225, 225, 315, 315]
    
    for i in range(8):
        ax.view_init(elev=elevs[i], azim=azims[i])
        plt.savefig(f"{save_path}/{name}_view{i}.png")

    plt.close()
    print(f"Finish visualizing {name} from 8 views")

def Seg_render_multiview_imgs_voxel(voxels, labels=None, save_path=None, name=None):

    """Create and save 8 different views of the voxel visualization"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    # Convert coords to voxel grid
    voxel_grid = np.zeros((64, 64, 64), dtype=bool)
    voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    
    # Create color array using 8 obvious colors: red, yellow, blue, green, purple, brown, orange, black
    colors = np.zeros((64, 64, 64, 4))
    label_colors = np.array([
        [1.0, 0.0, 0.0, 1.0],      # red
        [1.0, 1.0, 0.0, 1.0],      # yellow
        [0.0, 0.0, 1.0, 1.0],      # blue
        [0.0, 1.0, 0.0, 1.0],      # green
        [0.5, 0.0, 0.5, 1.0],      # purple
        [0.6, 0.3, 0.0, 1.0],      # brown
        [1.0, 0.5, 0.0, 1.0],      # orange
        [0.0, 0.0, 0.0, 1.0],      # black
    ])
    color_array = label_colors[labels].copy()
    color_array[:, 3] = 0.7  # Set alpha value
    
    # Assign colors to voxels
    colors[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = color_array
    
    # Plot voxels with colors
    ax.voxels(voxel_grid, facecolors=colors, edgecolor=None)
    
    # Create 8 different views (45 degrees apart)
    elevs = [20, -20, 20, -20, 20, -20, 20, -20]
    azims = [45, 45, 135, 135, 225, 225, 315, 315]
    for i in range(8):
        ax.view_init(elev=elevs[i], azim=azims[i])
        plt.savefig(f"{save_path}/{name}_view{i}.png")

    plt.close()
    print(f"Finish visualizing {name} from 8 views")


def render_voxels(coords, yaw, pitch, save_path=None, n=64):

    coords = coords.cpu().numpy()
    length = n

    voxels = np.zeros((length, length, length), dtype=bool)
    voxels[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    colors = np.zeros((length, length, length, 4))
    colors[coords[:, 0], coords[:, 1], coords[:, 2]] = [1.0, 1.0, 1.0, 1]  # White with alpha=0.7

    """Create and save a single view of voxels at specified yaw and pitch"""
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    
    # Hide axes and grid
    ax.set_axis_off()
    ax.grid(False)
    
    # Plot the voxels with colors
    ax.voxels(voxels, facecolors=colors, edgecolor=None)
    
    # Set view angle based on yaw and pitch
    ax.view_init(elev=20, azim=45)
    
    # Save the image with black background
    plt.savefig(save_path, facecolor='black', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Finished rendering voxels at view yaw={yaw}, pitch={pitch}")