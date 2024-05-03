# Command to run:
# python map_matcher_indoors.py --src_file data/demo/src.npy --ref_file data/demo/ref.npy --gt_file data/demo/gt.npy

# Change to experiment directory
model_directory = "/usr/local/src/GeoTransformer/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"

import sys

sys.path.insert(1, model_directory)

import open3d as o3d

import argparse

import torch
import numpy as np
import psutil

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data_config", required=True, help="Path to JSON file that denotes pairs"
    # )
    parser.add_argument("--src_file", required=True, help="")
    parser.add_argument("--ref_file", required=True, help="")
    return parser


def load_pose(filename):
    pose = np.vstack(
        (np.loadtxt(filename).reshape(3, 4), np.zeros((1, 4), dtype=float))
    )
    pose[3, 3] = 1.0
    return pose


def load_data(src_file, ref_file):
    src_pcd = o3d.io.read_point_cloud(src_file)
    ref_pcd = o3d.io.read_point_cloud(ref_file)

    src_points = np.asarray(src_pcd.points)
    ref_points = np.asarray(ref_pcd.points)

    print(f"src: {src_points.shape} ref: {ref_points.shape}")

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    src_pose = load_pose(src_file[:-4] + "-gtpose.txt")
    ref_pose = load_pose(ref_file[:-4] + "-gtpose.txt")
    transform = src_pose @ np.linalg.inv(ref_pose)
    data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg()

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load("/weights/geotransformer-3dmatch.pth.tar")
    model.load_state_dict(state_dict["model"])

    # prepare data
    data_dict = load_data(args.src_file, args.ref_file)
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
    )

    for key, val in data_dict.items():
        print(f"Key: {key} Val: {type(val)}")

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]

    # # visualization
    # ref_points = output_dict["ref_points"]
    # src_points = output_dict["src_points"]
    # ref_pcd = make_open3d_point_cloud(ref_points)
    # ref_pcd.estimate_normals()
    # ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    # src_pcd = make_open3d_point_cloud(src_points)
    # src_pcd.estimate_normals()
    # src_pcd.paint_uniform_color(get_color("custom_blue"))
    # draw_geometries(ref_pcd, src_pcd)
    # src_pcd = src_pcd.transform(estimated_transform)
    # draw_geometries(ref_pcd, src_pcd)

    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
