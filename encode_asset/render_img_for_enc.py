import os
import json
import numpy as np
from subprocess import DEVNULL, call
from .utils import sphere_hammersley_sequence
import open3d as o3d
import utils3d

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def render(file_path, name, output_dir, num_views=150):
    output_folder = os.path.join(output_dir, name)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL, stderr=DEVNULL)

def voxelize(file, name, output_dir):
    mesh = o3d.io.read_triangle_mesh(file)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(os.path.join(output_dir, f'voxels.ply'), vertices)

def renderImg_voxelize(input_file):
    install_blender()
    name = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(f"outputs/img_Enc", exist_ok=True)
    render(f"data/{input_file}", name, f"outputs/img_Enc/")
    voxelize(f"outputs/img_Enc/{name}/mesh.ply", name, f"outputs/img_Enc/{name}")

if __name__ == '__main__':

    # renderImg_voxelize("ancientFighter.glb")
    # renderImg_voxelize("BATHROOM_CLASSIC.glb")
    # renderImg_voxelize("CAR_CARRIER_TRAIN.glb")
    # renderImg_voxelize("castle.glb")
    # renderImg_voxelize("elephant.glb")
    # renderImg_voxelize("foodCartTwo.glb")
    # renderImg_voxelize("horseCart.glb")

    renderImg_voxelize("KITCHEN_FURNITURE_SET.glb")
    renderImg_voxelize("PartObjaverseTiny_Eight.glb")
    renderImg_voxelize("PartObjaverseTiny_Five.glb")
    renderImg_voxelize("PartObjaverseTiny_Seventeen.glb")
    renderImg_voxelize("RJ_Rabbit_Easter_Basket_Blue.glb")
    renderImg_voxelize("Sonny_School_Bus.glb")
    renderImg_voxelize("telephone.glb")