import os
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
from PIL import Image
from google import genai
import pickle
import re
import argparse
import imageio

from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.modules import sparse as sp
from VLM_LLM.Gemini_VLM import select_editing_parts, obtain_overall_prompts, select_img_to_edit, select_K, select_the_best_edited_object
from VLM_LLM.Gemini_LLM import identify_new_part, identify_ori_part, decompose_prompt
from Nano_banana import Nano_banana_edit
from PartField_segmentation import PartField_segmentation
from utilis import pc_to_voxel
from interweave_Trellis import interweave_Trellis_TI
from encode_asset.render_img_for_enc import renderImg_voxelize
from encode_asset.encode_into_SLAT import encode_into_SLAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_render_views():
    yaws = torch.Tensor([45, 45, 135, 135, 225, 225, 315, 315])
    yaws = yaws/360*2*np.pi
    pitches = torch.Tensor([0.45, -0.45, 0.45, -0.45, 0.45, -0.45, 0.45, -0.45])
    yaws = yaws.tolist()
    pitches = pitches.tolist()
    return yaws, pitches

def render_multiview_images(args, gaussian):

    prefix = f"outputs/img_multiview/{args['name']}"
    yaws, pitches = get_render_views()
    imgs = render_utils.Trellis_render_multiview_images(gaussian, yaws, pitches)['color']
    for i in range(len(imgs)):
        Image.fromarray(imgs[i]).save(f"{prefix}_{i:03d}.png")

def get_prompts(args):

    path = f"outputs/prompts/{args['name']}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            prompts = pickle.load(f)
    else:
        prompts = {}
        prompts["edit_prompt"] = args['prompt_edit']
        ret = obtain_overall_prompts(args['client'], args['prompt_edit'], args['name'])
        prompts["ori_cpl"] = ret.split("&&&")[0]
        prompts["editing_part"] = ret.split("&&&")[1]
        prompts["new_cpl"] = ret.split("&&&")[2]
        prompts["target_part"] = ret.split("&&&")[3]
        prompts["edit_type"] = (ret.split("&&&")[4]).split(".")[0]

        prompts["ori_s1_cpl"], prompts["ori_s2_cpl"] = decompose_prompt(args['client'], prompts["ori_cpl"])
        prompts["new_s1_cpl"], prompts["new_s2_cpl"] = decompose_prompt(args['client'], prompts["new_cpl"])
        prompts["ori_s1_part"] = identify_ori_part(args['client'], prompts["ori_s1_cpl"], prompts["editing_part"])
        prompts["ori_s2_part"] = identify_ori_part(args['client'], prompts["ori_s2_cpl"], prompts["editing_part"])
        prompts["new_part"] = identify_new_part(args['client'], prompts["new_cpl"], prompts["target_part"])
        prompts["new_s1_part"] = identify_new_part(args['client'], prompts["new_s1_cpl"], prompts["target_part"])
        prompts["new_s2_part"] = identify_new_part(args['client'], prompts["new_s2_cpl"], prompts["target_part"])
    if "Modification" in prompts["edit_type"]:
        prompts["edit_type"] = "Modification"
    elif "Deletion" in prompts["edit_type"]:
        prompts["edit_type"] = "Deletion"
    elif "Addition" in prompts["edit_type"]:
        prompts["edit_type"] = "Addition"
    else:
        raise ValueError(f"Invalid editing type: {prompts['edit_type']}")
    for key in prompts.keys():
        print(f"{key}: {prompts[key]}")
    with open(f"outputs/prompts/{args['name']}.pkl", "wb") as f:
        pickle.dump(prompts, f)
    return prompts

def segmentation(args, gaussian, voxel_coords):

    if not os.path.exists(f"outputs/images_seg/{args['name']}"):
        print(f"rendering seg of {args['name']}")
        os.makedirs(f"outputs/images_seg/{args['name']}")
        points = gaussian._xyz.detach()
        for K in range(3, 9):
            point_labels = np.load(f"PartField/clustering_results/cluster_out/{args['name']}_0_{K:02d}.npy").astype(np.int32)
            voxel_labels = pc_to_voxel(voxel_coords, points, point_labels).cpu().numpy()
            render_utils.Seg_render_multiview_imgs_voxel(voxel_coords.cpu().numpy(), voxel_labels, f"outputs/images_seg/{args['name']}", f"{args['name']}_{K:02d}_seg")

def grounding(args, slat, guassian, prompts):

    if args['edit_type'] == "Modification" or args['edit_type'] == "Deletion":

        length = 1

        path = f"outputs/grounding/{args['name']}.txt"
        if not os.path.exists(path):
            result = select_editing_parts(args['client'], f"outputs/images_seg/{args['name']}", args['name'], prompts["editing_part"])
            with open(path, "w") as f:
                f.write(result)
        else:
            with open(path, "r") as f:
                result = f.read()
        result = result.split("&&&")
        K = int(re.findall(r'\d+', result[0])[-1])
        selected_colors = result[1].split(",")

        points = guassian._xyz.detach()
        points_labels = np.load(f"PartField/clustering_results/cluster_out/{args['name']}_0_{K:02d}.npy").astype(np.int32)
        voxel_coords = slat.coords[:,1:]
        voxel_labels = pc_to_voxel(voxel_coords, points, points_labels)
        
        edit_parts = torch.zeros(64,64,64).to(device).to(torch.bool)
        preserved_parts = torch.zeros(64,64,64).to(device).to(torch.bool)
        colors = ["red", "yellow", "blue", "green", "purple", "brown", "orange", "black"]
        print(f"K: {K}")
        for i in range(K):
            selected = (voxel_labels == i).to(torch.bool)
            if colors[i] in selected_colors:
                print(f"select {colors[i]}")
                edit_parts[slat.coords[selected,1], slat.coords[selected,2], slat.coords[selected,3]] = True
            else:
                preserved_parts[slat.coords[selected,1], slat.coords[selected,2], slat.coords[selected,3]] = True
        
        bbox_edit = torch.zeros(64,64,64).to(device).to(torch.bool)
        bbox_preserved = torch.zeros(64,64,64).to(device).to(torch.bool)
        for i in range(K):
            selected = (voxel_labels == i).to(torch.bool)
            min_coords = torch.min(slat.coords[selected,1:], dim=0)[0]
            max_coords = torch.max(slat.coords[selected,1:], dim=0)[0]
            if colors[i] in selected_colors:
                bbox_edit[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, min_coords[2]:max_coords[2]+1] = True
            else:
                bbox_preserved[max(min_coords[0]-(length-1), 0):min(max_coords[0]+length, 64), max(min_coords[1]-(length-1), 0):min(max_coords[1]+length, 64), max(min_coords[2]-(length-1), 0):min(max_coords[2]+length, 64)] = True
        
        return edit_parts, preserved_parts, bbox_edit, bbox_preserved
    
    else:
        length = 1

        path = f"outputs/grounding/{args['name']}.txt"
        if not os.path.exists(path):
            result = select_K(args['client'], f"outputs/images_seg/{args['name']}", args['name'])
            K = result
            with open(path, "w") as f:
                f.write("Add " + str(result))
        else:
            with open(path, "r") as f:
                K = int(f.read().split(" ")[1])
        
        points = guassian._xyz.detach()
        points_labels = np.load(f"PartField/clustering_results/cluster_out/{args['name']}_0_{K:02d}.npy").astype(np.int32)
        voxel_coords = slat.coords[:,1:]
        voxel_labels = pc_to_voxel(voxel_coords, points, points_labels)

        preserved_parts = torch.zeros(64,64,64).to(device).to(torch.bool)
        bbox_preserved = torch.zeros(64,64,64).to(device).to(torch.bool)
        for i in range(K):
            selected = (voxel_labels == i).to(torch.bool)
            preserved_parts[slat.coords[selected,1], slat.coords[selected,2], slat.coords[selected,3]] = True
            min_coords = torch.min(slat.coords[selected,1:], dim=0)[0]
            max_coords = torch.max(slat.coords[selected,1:], dim=0)[0]
            bbox_preserved[max(min_coords[0]-(length-1), 0):min(max_coords[0]+length, 64), max(min_coords[1]-(length-1), 0):min(max_coords[1]+length, 64), max(min_coords[2]-(length-1), 0):min(max_coords[2]+length, 64)] = True
        
        return None, preserved_parts, None, bbox_preserved

def compute_editing_region(slat, edit_parts, preserved_parts, bbox_edit, bbox_preserved):

    mask = torch.zeros(64,64,64).to(device).to(torch.bool)

    empty_coords = torch.argwhere((preserved_parts | edit_parts) == 0)
    # Find K nearest neighbors for each empty voxel
    k = 100
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(slat.coords[:,1:].cpu().numpy())
    distances, indices = nbrs.kneighbors(empty_coords.cpu().numpy())
    indices = torch.from_numpy(indices).to(device)
    neighbor_masks = edit_parts[slat.coords[indices,1], slat.coords[indices,2], slat.coords[indices,3]]
    mask_proportions = neighbor_masks.float().mean(dim=1)
    threshold = 0.5
    mask[empty_coords[:,0], empty_coords[:,1], empty_coords[:,2]] = (mask_proportions > threshold)
    
    mask = mask | edit_parts
    mask = mask & ~preserved_parts
    mask = mask | ~bbox_preserved

    return mask

def convert_box_to_mask(box):
    mask = torch.zeros(64,64,64).to(device).to(torch.bool)
    for x in range(box[0], box[3]+1):
        for y in range(box[1], box[4]+1):
            for z in range(box[2], box[5]+1):
                mask[x,y,z] = True
    return mask

def obtain_img_new(args, gaussian, prompts):

    yaws = torch.Tensor([0, 45, 90, 135, 180, 225, 270, 315]*3)
    yaws = yaws/360*2*np.pi
    angle = 0.45
    pitches = torch.Tensor([angle]*8 + [0.0]*8 + [-angle]*8)
    yaws = yaws.tolist()
    pitches = pitches.tolist()

    imgs = render_utils.Trellis_render_multiview_images(gaussian, yaws, pitches)['color']
    for i in range(len(imgs)):
        Image.fromarray(imgs[i]).save(f"outputs/img_multiview/{args['name']}_preEdit{i:03d}.png")
    
    path = f"outputs/img_edit/ID/{args['name']}_editImgID.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            idx = int(f.read())
    else:
        idx = select_img_to_edit(args['client'], f"{args['name']}", prompts['edit_prompt'])
        print(f"selected img idx: {idx}")
        with open(path, "w") as f:
            f.write(str(idx))
    
    path = f"outputs/img_edit/img/{args['name']}_editedImg.png"
    if os.path.exists(path):
        img_new = Image.open(path)
    else:
        img_new = Nano_banana_edit(args['client'], f"outputs/img_multiview/{args['name']}_preEdit{idx:03d}.png", prompts['edit_prompt'], prompts['new_part'])
        print("obtain img_new")
        img_new = img_new.resize((518, 518), Image.Resampling.LANCZOS)
        img_new.save(path)
    return img_new

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--editing_prompt", type=str, required=True)
    inputs = parser.parse_args()

    name = inputs.input_file.split("/")[-1].split(".")[0]
    os.makedirs(f"outputs/grounding", exist_ok=True)
    os.makedirs(f"outputs/images_seg", exist_ok=True)
    os.makedirs(f"outputs/img_edit/ID", exist_ok=True)
    os.makedirs(f"outputs/img_edit/img", exist_ok=True)
    os.makedirs(f"outputs/img_multiview", exist_ok=True)
    os.makedirs(f"outputs/masks", exist_ok=True)
    os.makedirs(f"outputs/prompts", exist_ok=True)
    os.makedirs(f"outputs/slat", exist_ok=True)
    os.makedirs(f"outputs/videos", exist_ok=True)
    os.makedirs(f"PartField/data", exist_ok=True)

    args = {
        'name': name,
        'prompt_edit': inputs.editing_prompt,
        'combs': [0, 1, 2, 3, 4]
    }

    if not os.path.exists(f"outputs/img_Enc/{name}"):
        renderImg_voxelize(inputs.input_file)
    if not os.path.exists(f"outputs/slat/{name}_feats.pt"):
        encode_into_SLAT(name)

    # load models
    trellis_text = TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")
    trellis_img = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    trellis_text.cuda()
    trellis_img.cuda()
    args['client'] = genai.Client()

    slat = sp.SparseTensor(
        feats=torch.load(f"outputs/slat/{args['name']}_feats.pt"),
        coords=torch.load(f"outputs/slat/{args['name']}_coords.pt"),
    )
    gaussian = (trellis_text.decode_slat(slat, ['gaussian']))["gaussian"][0]

    gaussian.save_ply(f"PartField/data/{args['name']}.ply")
    os.system("python PartField_segmentation.py")

    render_multiview_images(args, gaussian)
    prompts = get_prompts(args)
    args['edit_type'] = prompts["edit_type"]

    segmentation(args, gaussian, slat.coords[:,1:])
    edit_parts, preserved_parts, bbox_edit, bbox_preserved = grounding(args, slat, gaussian, prompts)
    if prompts["edit_type"] == "Modification":
        mask = compute_editing_region(slat, edit_parts, preserved_parts, bbox_edit, bbox_preserved)
    elif prompts["edit_type"] == "Deletion":
        mask = edit_parts
    elif prompts["edit_type"] == "Addition":
        mask = ~preserved_parts
    else:
        raise ValueError(f"Invalid editing type: {prompts['edit_type']}")
    torch.save(mask, f"outputs/masks/{args['name']}.pt")

    if prompts["edit_type"] == "Addition" or prompts["edit_type"] == "Modification":
        gaussian = (trellis_text.decode_slat(slat, ['gaussian']))["gaussian"][0]
        img_new = obtain_img_new(args, gaussian, prompts)
    else:
        img_new = None

    combinations = [
        {"s1_pos_cond": "new_s1_cpl", "s1_neg_cond": "ori_s1_cpl", "s2_pos_cond": "new_s2_cpl", "s2_neg_cond": "ori_s2_cpl", "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "new_s1_cpl", "s1_neg_cond": "null", "s2_pos_cond": "new_s2_cpl", "s2_neg_cond": "null", "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "new_s1_part", "s1_neg_cond": "ori_s1_part", "s2_pos_cond": "new_s2_part", "s2_neg_cond": "ori_s2_part", "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "new_s1_part", "s1_neg_cond": "null", "s2_pos_cond": "new_s2_part", "s2_neg_cond": "null", "cnt": 1, "cfg_strength": 7.5},
        {"s1_pos_cond": "null", "s1_neg_cond": "null", "s2_pos_cond": "null", "s2_neg_cond": "null", "cnt": 1, "cfg_strength": 0},
    ]
    slats_edited = []
    for i in range(len(combinations)):
        if i not in args['combs']:
            continue
        args.update(combinations[i])
        slat_new = interweave_Trellis_TI(args, trellis_text, trellis_img,
            slat, mask,
            prompts,
            img_new,
            seed=1)
        slats_edited.append(slat_new)

        outputs = trellis_text.decode_slat(slat_new, ['gaussian'])
        yaws = torch.Tensor([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5])
        yaws = yaws/360*2*np.pi
        pitches = torch.Tensor([0.45]*16)
        yaws = yaws.tolist()
        pitches = pitches.tolist()
        imgs = render_utils.Trellis_render_multiview_images(outputs['gaussian'][0], yaws, pitches)['color']
        for j in range(len(imgs)):
            Image.fromarray(imgs[j]).save(f"outputs/img_multiview/{args['name']}_edited_comb{i}_{j:03d}.png")
    
    best_index = select_the_best_edited_object(args['client'], args['name'], prompts['edit_prompt'])
    print(f"best index: {best_index}")

    torch.save(slats_edited[best_index].feats, f"outputs/slat/{args['name']}_edited_feats.pt")
    torch.save(slats_edited[best_index].coords, f"outputs/slat/{args['name']}_edited_coords.pt")
    outputs = trellis_text.decode_slat(slats_edited[best_index], ['gaussian'])
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"outputs/videos/{args['name']}_edited.mp4", video, fps=30)
    print(f"finish editing {args['name']}.mp4")