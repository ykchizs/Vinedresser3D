
import os
from plyfile import PlyData
import torch
import numpy as np
from PIL import Image
import utils3d
import math
import torch.nn.functional as F
from torchvision import transforms
import json

from trellis.modules import sparse as sp
import trellis.models as models

os.environ['SPCONV_ALGO'] = 'native' 

def load_ply_to_numpy(filename):
    """
    Load a PLY file and extract the point cloud as a (N, 3) NumPy array.

    Parameters:
        filename (str): Path to the PLY file.

    Returns:
        numpy.ndarray: Point cloud array of shape (N, 3).
    """
    ply_data = PlyData.read(filename)

    # Extract vertex data
    vertex_data = ply_data["vertex"]
    
    # Convert to NumPy array (x, y, z)
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T

    return points

def encode_into_SLAT(name):

    num_views = 150

    indices = load_ply_to_numpy(f"outputs/img_Enc/{name}/voxels.ply")
    indices = torch.from_numpy((indices+0.5)*64).long().cuda()
    positions = (indices.to(torch.float32)/64.0 - 0.5)

    # extract feature utilis
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    patchtokens_lst = []
    uv_lst = []
    fov = math.radians(40)
    views = json.load(open(f"outputs/img_Enc/{name}/transforms.json"))["frames"]
    for i in range(num_views):

        img = Image.open(f"outputs/img_Enc/{name}/{i:03d}.png")
        img = img.resize((518, 518), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 255
        if img.shape[2] == 4:
            img = img[:, :, :3] * img[:, :, 3:]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = transform(img)

        batch_images = torch.stack([img])
        batch_images = batch_images.cuda()

        c2w = torch.tensor(views[i]['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsic = torch.inverse(c2w)
        fov = views[i]['camera_angle_x']
        intrinsic = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        batch_extrinsics = extrinsic.unsqueeze(0).cuda()
        batch_intrinsics = intrinsic.unsqueeze(0).cuda()

        features = dinov2_model(batch_images, is_training=True)
        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(1, 1024, n_patch, n_patch)
        patchtokens_lst.append(patchtokens.detach().cpu())
        uv_lst.append(uv.detach().cpu())

    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)
    feats = F.grid_sample(
        patchtokens,
        uv.unsqueeze(1),
        mode='bilinear',
        align_corners=False,
    ).squeeze(2).permute(0, 2, 1).detach().cpu().numpy()
    feats = np.mean(feats, axis=0).astype(np.float16)

    encoder = models.from_pretrained("JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16").eval().cuda()
    aggregated_features = sp.SparseTensor(
        feats = torch.from_numpy(feats).float(),
        coords = torch.cat([
            torch.zeros(feats.shape[0], 1).int(),
            indices.cpu().int(),
        ], dim=1),
    ).cuda()
    latent = encoder(aggregated_features, sample_posterior=False)
    assert torch.isfinite(latent.feats).all(), "Non-finite latent"

    torch.save(latent.feats, f"outputs/slat/{name}_feats.pt")
    torch.save(latent.coords, f"outputs/slat/{name}_coords.pt")

    print(f"finish encoding {name}")
    