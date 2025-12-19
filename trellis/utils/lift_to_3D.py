
import numpy as np
import torch
import utils3d

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

def euler_to_rotation_matrix(yaw, pitch):
    """Convert yaw and pitch to a rotation matrix."""
    yaw_rad = yaw
    pitch_rad = pitch

    R_yaw = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    return np.dot(R_yaw, R_pitch)

def pixel_to_ray_direction(x, y, focal_length, image_width, image_height):
    """Convert pixel coordinates to ray direction."""
    c_x = image_width / 2
    c_y = image_height / 2
    
    ray_directions = np.concatenate([
        np.expand_dims((x - c_x) / focal_length, axis=-1),
        np.expand_dims((y - c_y) / focal_length, axis=-1),
        np.expand_dims(np.ones_like(x), axis=-1)
    ], axis=-1)
    
    return ray_directions / np.expand_dims(np.linalg.norm(ray_directions, axis=-1), axis=-1)

def ray_voxel_intersection(ray_origins, ray_directions, voxel_coords):
    """Find the nearest voxel hit by the ray."""
    # nearest_voxel = None
    # nearest_distance = float('inf')  # Initialize to infinity

    # for voxel in voxel_coords:
    #     # Define voxel boundaries
    #     min_corner = voxel
    #     max_corner = voxel + np.array([1, 1, 1])  # Assuming voxel size is 1 unit

    #     # Compute t values for the AABB intersection
    #     t_min = (min_corner - ray_origin) / ray_direction
    #     t_max = (max_corner - ray_origin) / ray_direction

    #     t1 = np.minimum(t_min, t_max)
    #     t2 = np.maximum(t_min, t_max)

    #     t_near = np.max(t1)
    #     t_far = np.min(t2)

    #     if t_near <= t_far and t_far >= 0:
    #         # Calculate the distance from the ray origin to the intersection point
    #         distance = t_near
            
    #         if distance < nearest_distance:  # Check if this voxel is closer
    #             nearest_distance = distance
    #             nearest_voxel = voxel

    # Define voxel boundaries
    min_corners = voxel_coords
    max_corners = voxel_coords + np.array([1, 1, 1])  # Assuming voxel size is 1 unit

    # Compute t values for the AABB intersection
    t_min = (min_corners[:, None, :] - ray_origins[None, :, :]) / ray_directions[None, :, :]
    t_max = (max_corners[:, None, :] - ray_origins[None, :, :]) / ray_directions[None, :, :]

    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)

    t_near = np.max(t1, axis=-1)
    t_far = np.min(t2, axis=-1)

    intersected = (t_near <= t_far) & (t_far >= 0)
    hit_voxels = np.any(intersected, axis=0)
    distances = t_near
    distances[~intersected] = np.inf
    nearest_voxel = np.argmin(distances, axis=0)
    return hit_voxels, nearest_voxel

def emit_rays(image_width, image_height, fov, r, yaw, pitch, voxel_coords, aggreated_mask):
    """Emit rays from each pixel and find intersections with voxels."""
    # rotation_matrix = euler_to_rotation_matrix(yaw, pitch)
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    # Calculate focal length from field of view (FOV)
    # For perspective projection, focal length = 1/tan(FOV/2)
    # This determines how much the scene is "zoomed in/out"
    # Smaller FOV = larger focal length = more zoomed in
    # fov_rad = fov * np.pi / 180
    # focal = 1 / np.tan(fov_rad / 2)
    # Extract rotation matrix from extrinsics
    # Extrinsics is a 4x4 matrix where the top-left 3x3 is the rotation matrix
    # rotation_matrix = extrinsics[:3, :3].cpu().numpy()

    # hit_voxels = {}

    # for y in range(image_height):
    #     for x in range(image_width):
    #         ray_direction = pixel_to_ray_direction(x, y, focal_length, image_width, image_height)
    #         transformed_ray = np.dot(rotation_matrix, ray_direction)

    #         # Assuming the camera is at the origin
    #         ray_origin = np.array([0, 0, 0])
    #         hit_voxel = ray_voxel_intersection(ray_origin, transformed_ray, voxel_coords)

    #         if hit_voxel is not None:
    #             breakpoint()
    #             hit_voxels[(x, y)] = hit_voxel

    # Create grid of pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(0, image_height, 4), np.arange(0, image_width, 4), indexing='xy')
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    in_mask = aggreated_mask[y_coords, x_coords]
    breakpoint()

    # Scale voxel coordinates from 64^3 to world space (assuming unit cube)
    voxel_scale = 1.0 / 64.0
    scaled_voxel_coords = voxel_coords * voxel_scale

    # Convert scaled voxel coordinates to homogeneous coordinates 
    voxel_coords_homo = np.concatenate([scaled_voxel_coords, np.ones((scaled_voxel_coords.shape[0], 1))], axis=1)

    # Transform voxels to camera space using extrinsics
    voxels_camera = np.dot(voxel_coords_homo, extrinsics.cpu().numpy().T)

    # Project to image space using intrinsics
    voxels_image = np.dot(voxels_camera[:, :3], intrinsics.cpu().numpy().T)

    # Perspective divide
    voxels_image = voxels_image / voxels_camera[:, 2:3]

    # Get pixel coordinates
    pixel_coords = voxels_image[:, :2].astype(np.int32)

    # Filter points that are within image bounds
    valid_points = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_width) & \
                  (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_height) & \
                  (voxels_camera[:, 2] > 0)  # Points in front of camera
    
    breakpoint()

    # Create hit_voxels array - True for voxels that project to valid image coordinates
    hit_voxels = valid_points

    # Create nearest_voxel array - for each pixel, store index of closest voxel
    # Initialize with -1 for pixels with no voxel
    nearest_voxel = np.zeros(len(x_coords), dtype=np.int32)
    
    # For valid points, find distances to camera
    distances = np.full(len(x_coords), np.inf)
    valid_distances = np.linalg.norm(voxels_camera[valid_points, :3], axis=1)
    
    # For each valid projected point, update nearest_voxel if it's closer than current
    for i, (x, y, dist) in enumerate(zip(pixel_coords[valid_points, 0], 
                                       pixel_coords[valid_points, 1],
                                       valid_distances)):
        pixel_indices = (y_coords == y) & (x_coords == x)
        if pixel_indices.any():
            if dist < distances[pixel_indices]:
                distances[pixel_indices] = dist
                nearest_voxel[pixel_indices] = np.where(valid_points)[0][i]
    

    
    # ray_directions = pixel_to_ray_direction(x_coords, y_coords, focal, image_width, image_height)
    
    # # Transform all rays at once
    # transformed_rays = np.dot(ray_directions, rotation_matrix.T)
    
    # # Create array of ray origins
    # ray_origins = np.tile(np.array([0, 0, 0]), (len(transformed_rays), 1))
    
    # # Find intersections for all rays
    # hit_voxels, nearest_voxel = ray_voxel_intersection(ray_origins, transformed_rays, voxel_coords)

    # return hit_voxels, nearest_voxel, in_mask

if __name__ == "__main__":
    coords = torch.load("outputs/full/slat/animalcar_coords.pt")
    aggreated_mask = torch.load("outputs/aggregated_mask/animalcar_view0.pt")
    emit_rays(512, 512, 40, 2, 45/360*2*np.pi, 0.45, coords[:,1:].cpu().numpy(), aggreated_mask.cpu().numpy())



# Get image dimensions
            # H, W = img.shape[:2]
            # # Create pixel coordinates for all pixels
            # y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            # y_coords = y_coords.flatten()
            # x_coords = x_coords.flatten()
            # # Convert to normalized device coordinates (-1 to 1)
            # ndc_x = (2 * x_coords.float() / W - 1)
            # ndc_y = (2 * y_coords.float() / H - 1)
            # # Calculate ray directions based on FOV and image pose
            # fov_rad = fov * np.pi / 180
            # # Calculate focal length from field of view (FOV)
            # # For perspective projection, focal length = 1/tan(FOV/2)
            # # This determines how much the scene is "zoomed in/out"
            # # Smaller FOV = larger focal length = more zoomed in
            # focal = 1 / np.tan(fov_rad / 2)
            # # Get camera rotation matrix for this view
            # rot_pitch = torch.tensor([[1, 0, 0],
            #                         [0, np.cos(pitches[i]), -np.sin(pitches[i])],
            #                         [0, np.sin(pitches[i]), np.cos(pitches[i])]])
            
            # # rot_yaw = torch.tensor([[np.cos(yaws[i]), 0, np.sin(yaws[i])],
            # #                        [0, 1, 0],
            # #                        [-np.sin(yaws[i]), 0, np.cos(yaws[i])]])
            # rot_yaw = torch.tensor([[np.cos(yaws[i]), -np.sin(yaws[i]), 0],
            #                        [np.sin(yaws[i]), np.cos(yaws[i]), 0],
            #                        [0, 0, 1]])
            
            # R = rot_yaw @ rot_pitch
            # # Calculate ray directions in camera space
            # # -ndc_y: Flip y-axis because image coordinates have origin at top-left (y increases downward)
            # # while OpenGL/camera coordinates have origin at center with y increasing upward
            # # -focal: The focal length is negative because we want rays to point forward from camera
            # # which is in the -z direction in camera space (OpenGL convention)
            # ray_dirs = torch.stack([ndc_x, -ndc_y, -focal * torch.ones_like(ndc_x)], dim=-1)
            # ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
            # # Transform ray directions to world space
            # ray_dirs = (R @ ray_dirs.to(torch.double).unsqueeze(-1)).squeeze(-1)
            # # Camera origin in world space (assuming camera at r distance)
            # cam_pos = torch.tensor([r * np.sin(yaws[i]), r * np.sin(pitches[i]), r * np.cos(yaws[i])])
            # # Get voxel coordinates
            # voxel_coords = slat.coords[:,1:].cpu()
            
            # # Initialize arrays to store results
            # hit_voxels = []  # Will store voxel indices for each ray that hits
            # hit_pixels = []  # Will store pixel coordinates for each ray that hits
            # hit_mask = []    # Will store whether hit pixel was in aggregated mask
            
            # # Only process rays from pixels in the aggregated mask
            # mask_indices = torch.where(aggreated_mask.flatten())[0]
            
            # # For each ray from masked pixels, find first voxel intersection
            # for idx in mask_indices:
            #     origin = cam_pos
            #     direction = ray_dirs[idx]
                
            #     # Calculate intersections with all voxels
            #     t = (voxel_coords - origin) / direction.unsqueeze(0)
                
            #     # Find minimum positive t value for valid intersection
            #     valid_mask = (t > 0).all(dim=1)
            #     if valid_mask.any():
            #         t_valid = t[valid_mask]
            #         # Get closest intersection
            #         min_t_idx = t_valid.max(dim=1)[0].argmin()
            #         voxel_idx = torch.where(valid_mask)[0][min_t_idx]
                    
            #         # Store results
            #         pixel_y = y_coords[idx]
            #         pixel_x = x_coords[idx]
            #         hit_voxels.append(voxel_idx)
            #         hit_pixels.append((pixel_y, pixel_x))
            #         hit_mask.append(True)  # Since we only process masked pixels
            
            # print(f"View {i}:")
            # print(f"Total rays that hit voxels: {len(hit_voxels)}")
            # print(f"Rays from masked pixels that hit: {sum(hit_mask)}")
            # print(f"Rays from unmasked pixels that hit: {len(hit_voxels) - sum(hit_mask)}")