# Change to experiment directory for additional scripts
model_directory = "/usr/local/src/GeoTransformer/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"

import sys

sys.path.insert(1, model_directory)

import argparse
import json
import os
import time
import datetime

import psutil
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.registration import compute_registration_error
from geotransformer.utils.open3d import (
    make_open3d_point_cloud,
    get_color,
    draw_geometries,
)

from config import make_cfg
from model import create_model


p = psutil.Process()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config", required=True, help="Path to JSON file that denotes pairs"
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.1, help="Grid size for KPConv backend"
    )
    parser.add_argument("--visualize", action="store_true")
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

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
    }

    src_pose = load_pose(src_file[:-4] + "-gtpose.txt")
    ref_pose = load_pose(ref_file[:-4] + "-gtpose.txt")
    transform = ref_pose @ np.linalg.inv(src_pose)
    data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg()
    cfg.backbone.init_voxel_size = args.grid_size

    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch

    # Load data config
    with open(args.data_config, "r") as f:
        data_config = json.load(f)

    print(
        f" | Data root: {data_config['root']}. Num. pairs: {len(data_config['pairs'])}"
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load("/weights/geotransformer-3dmatch.pth.tar")
    model.load_state_dict(state_dict["model"])

    # JSON to store results
    output_data = {
        "data_config": args.data_config,
        "full_configuration": {
            "grid_size": cfg.backbone.init_voxel_size,
        },
        "results": [],
    }

    experiment_name_from_data = "_".join(args.data_config[:-5].split("/")[-2:])
    time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_file = os.path.join(
        "/results", experiment_name_from_data + "_" + time_string + ".json"
    )

    print(f" | Output file: '{output_file}'")
    with open(output_file, "w") as f:
        json.dump(output_data, f)

    print(f" | Start processing...")
    print()

    for pair in data_config["pairs"]:
        stats = {"pcd1": pair[0], "pcd2": pair[1]}

        # Reset GPU memory usage data statistics to eliminate any leak
        torch.cuda.reset_peak_memory_stats()

        map1_path = os.path.join(data_config["root"], pair[0])
        map2_path = os.path.join(data_config["root"], pair[1])

        print(f" | -> Pair: {pair[1]} -> {pair[0]}")

        # Load PCD into the memory
        #
        # Any additional processing beyond raw pointcloud that is needed to be performed
        # are separated and done inside the timed block.
        #
        # Ensure that the order is correct here
        data_dict = load_data(map2_path, map1_path)
        stats["pcd1_size"] = len(data_dict["ref_points"])
        stats["pcd2_size"] = len(data_dict["src_points"])

        # Show target parameters
        tf = data_dict["transform"]
        theta = Rotation.from_matrix(tf[0:3, 0:3]).as_euler("zyx")[0]

        stats["target_x"] = float(tf[0, 3])
        stats["target_y"] = float(tf[1, 3])
        stats["target_z"] = float(tf[2, 3])
        stats["target_t"] = float(theta)
        print(
            f" | Target x: {tf[0, 3]: .5f} y: {tf[1, 3]: .5f} z: {tf[2, 3]: .5f}, t: {theta: .5f}"
        )

        try:
            # prediction
            start = time.time()

            src_feats = np.ones_like(data_dict["src_points"][:, :1])
            ref_feats = np.ones_like(data_dict["ref_points"][:, :1])

            data_dict["src_feats"] = src_feats.astype(np.float32)
            data_dict["ref_feats"] = ref_feats.astype(np.float32)

            data_dict = registration_collate_fn_stack_mode(
                [data_dict],
                cfg.backbone.num_stages,
                cfg.backbone.init_voxel_size,
                cfg.backbone.init_radius,
                neighbor_limits,
            )

            data_dict = to_cuda(data_dict)
            output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)

            end = time.time()
            stats["t_total"] = end - start

            # get results
            tf_est = output_dict["estimated_transform"]
            theta_est = Rotation.from_matrix(tf_est[0:3, 0:3]).as_euler("zyx")[0]

            stats["result_x"] = float(tf_est[0, 3])
            stats["result_y"] = float(tf_est[1, 3])
            stats["result_z"] = float(tf_est[2, 3])
            stats["result_t"] = float(theta_est)

            print(
                f" | Result x: {tf_est[0, 3]: .5f} y: {tf_est[1, 3]: .5f} z: {tf_est[2, 3]: .5f}, t: {theta_est: .5f}"
            )

            # compute error
            rre_deg, rte = compute_registration_error(tf, tf_est)
            rre = float(np.deg2rad(rre_deg))
            rte = float(rte)

            stats["error_position"] = rte
            stats["error_angle"] = rre

            stats["mem_cpu"] = p.memory_info().rss
            stats["mem_gpu"] = torch.cuda.max_memory_allocated()

            print(f" | Error: {rte:.5f}m / {rre:.5f}rad. Took {stats['t_total']:.5f}s")

            if args.visualize:
                ref_points = output_dict["ref_points"]
                ref_pcd = make_open3d_point_cloud(ref_points)
                ref_pcd.estimate_normals()
                ref_pcd.paint_uniform_color(get_color("custom_yellow"))

                src_points = output_dict["src_points"]
                src_pcd = make_open3d_point_cloud(src_points)
                src_pcd = src_pcd.transform(tf_est)
                src_pcd.estimate_normals()
                src_pcd.paint_uniform_color(get_color("custom_blue"))
                draw_geometries(ref_pcd, src_pcd)

        except:
            print(f" | Cannot process")

        finally:
            stats["mem_cpu"] = p.memory_info().rss
            stats["mem_gpu"] = torch.cuda.max_memory_allocated()

            # Save results after each run
            output_data["results"].append(stats)
            with open(output_file, "w") as f:
                json.dump(output_data, f)
            print()

    print(" | Completed!")


if __name__ == "__main__":
    main()
