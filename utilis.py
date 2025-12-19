import torch
import numpy as np
import utils3d
from trellis.modules import sparse as sp
import trellis.models as models
from sklearn.neighbors import NearestNeighbors

def find_nn_label(points, point_labels, coords):
    """Find the nearest neighbor point label for given coordinates (x,y,z)"""
    # Convert inputs to appropriate format
    query_points = coords.float()
    points = points.float()

    # Use scikit-learn's KNN for nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points.cpu().numpy())
    dists, indices = nbrs.kneighbors(query_points.cpu().numpy())
    indices = torch.from_numpy(indices)
    
    # Get labels of nearest neighbors
    nn_labels = point_labels[indices.squeeze().cpu().numpy()]
    
    return torch.Tensor(nn_labels).to(points.device).squeeze()

def pc_to_voxel(voxel_coords, points, point_labels):
    points[:,0] = (points[:,0] - points[:,0].min())/(points[:,0].max() - points[:,0].min()) * (voxel_coords[:,0].max() - voxel_coords[:,0].min()) + voxel_coords[:,0].min()
    points[:,1] = (points[:,1] - points[:,1].min())/(points[:,1].max() - points[:,1].min()) * (voxel_coords[:,1].max() - voxel_coords[:,1].min()) + voxel_coords[:,1].min()
    points[:,2] = (points[:,2] - points[:,2].min())/(points[:,2].max() - points[:,2].min()) * (voxel_coords[:,2].max() - voxel_coords[:,2].min()) + voxel_coords[:,2].min()
    label_count = torch.zeros(64,64,64,np.max(point_labels)+1).to(points.device)
    assigned_coords = torch.floor(points)
    indices = torch.stack([assigned_coords[:,0], assigned_coords[:,1], assigned_coords[:,2], torch.tensor(point_labels.squeeze(), device=assigned_coords.device)], dim=1).long()
    label_count.index_put_(
        (indices[:,0], indices[:,1], indices[:,2], indices[:,3]),
        torch.ones(len(assigned_coords), device=assigned_coords.device),
        accumulate=True
    )
    nearest_label = find_nn_label(points, point_labels, voxel_coords)
    max_values, max_ids = label_count[voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2], :].max(dim=-1)
    use_label_count = (max_values > 0).to(torch.int32)
    voxel_labels = (max_ids*use_label_count + nearest_label*(1-use_label_count)).to(torch.int32)
    return voxel_labels

def ply_to_coords(ply_path):
    position = utils3d.io.read_ply(ply_path)[0]
    coords = ((torch.tensor(position) + 0.5) * 64).int().contiguous().cuda()
    return coords

def feats_to_slat(pipeline, feats_path):
    feats = np.load(feats_path)
    feats_tensor = sp.SparseTensor(
        feats=torch.from_numpy(feats["patchtokens"]).float(),
        coords=torch.cat(
            [
                torch.zeros(feats["patchtokens"].shape[0], 1).int(),
                torch.from_numpy(feats["indices"]).int(),
            ],
            dim=1,
        ),
    ).cuda()
    feats_encoder = models.from_pretrained("JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16").eval().cuda()
    slat = feats_encoder(feats_tensor, sample_posterior=False)
    return slat