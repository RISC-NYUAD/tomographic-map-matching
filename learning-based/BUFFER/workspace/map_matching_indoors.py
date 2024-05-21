import sys

sys.path.append("/usr/local/src/BUFFER")

import argparse
import json
import os
import time
import psutil

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch

from models.BUFFER import buffer
from ThreeDMatch.config import make_cfg
from ThreeDMatch.dataloader import collate_fn_descriptor

import warnings

p = psutil.Process()


def load_pose(filename):
    pose = np.vstack(
        (np.loadtxt(filename).reshape(3, 4), np.zeros((1, 4), dtype=float))
    )
    pose[3, 3] = 1.0
    return pose


def compute_gt_pose(src_file, ref_file):
    src_pose = load_pose(src_file[:-4] + "-gtpose.txt")
    ref_pose = load_pose(ref_file[:-4] + "-gtpose.txt")
    transform = ref_pose @ np.linalg.inv(src_pose)
    return transform


def compute_error(tf, tf_est):
    error_position = np.linalg.norm(tf_est[:3, 3] - tf[:3, 3])
    error_rotation = np.arccos(
        np.clip(
            (np.trace(tf_est[:3, :3].T @ tf[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16
        )
    )
    return error_position, error_rotation


def downsample_and_shuffle(pcd, voxel_size, max_size=0):
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.array(downsampled.points)
    np.random.shuffle(pts)

    if max_size > 0:
        idx = np.random.choice(range(pts.shape[0]), max_size, replace=False)
        pts = pts[idx]

    return pts, downsampled


def compute_and_append_normals(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location()

    normals = np.array(pcd.normals)
    pts_with_normals = np.concatenate([pts, normals], axis=-1)
    return pts_with_normals


def process_pcd_pair(source_raw, target_raw, pose, config, neighborhood_limits):
    # Mainly imitating the process here:
    # https://github.com/The-Learning-And-Vision-Atelier-LAVA/BUFFER/blob/main/ThreeDMatch/dataset.py#L89

    # First downsampling
    source_pts, source_pcd = downsample_and_shuffle(source_raw, config.data.downsample)
    target_pts, target_pcd = downsample_and_shuffle(target_raw, config.data.downsample)

    # Keypoints
    source_kpt, source_pcd = downsample_and_shuffle(
        source_pcd, config.data.voxel_size_0, config.data.max_numPts
    )
    target_kpt, target_pcd = downsample_and_shuffle(
        target_pcd, config.data.voxel_size_0, config.data.max_numPts
    )

    # Augment points with normals
    source_kpt_normals = compute_and_append_normals(source_kpt)
    target_kpt_normals = compute_and_append_normals(target_kpt)

    data = {
        "src_fds_pts": source_pts,
        "tgt_fds_pts": target_pts,
        "relt_pose": pose,
        "src_sds_pts": source_kpt_normals,
        "tgt_sds_pts": target_kpt_normals,
        "src_id": 0,
        "tgt_id": 0,
    }

    return collate_fn_descriptor([data], config, neighborhood_limits)


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config", required=True, help="Path to JSON file that denotes pairs"
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.1, help="Grid size for KPConv backend"
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    time_string = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Load model
    config = make_cfg()
    config.stage = "test"
    model = buffer(config)

    for stage in config.train.all_stage:
        path_to_weights = f"/weights/ThreeDMatch/{stage}/best.pth"
        state_dict = torch.load(path_to_weights)
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print(f" | Loaded stage {stage} from file {path_to_weights}")

    total = sum([param.nelement() for param in model.parameters()])
    print(f" | Number of parameters: {total}")

    model = torch.nn.DataParallel(model, device_ids=[0])
    model.eval()

    # Load data config
    with open(args.data_config, "r") as f:
        data_config = json.load(f)

    print(
        f" | Data root: {data_config['root']}. Num. pairs: {len(data_config['pairs'])}"
    )

    experiment_name_from_data = "_".join(args.data_config[:-5].split("/")[-2:])
    output_file = os.path.join(
        "/results", experiment_name_from_data + "_" + time_string + ".json"
    )

    # JSON to store results
    output_data = {
        "data_config": args.data_config,
        "full_configuration": {},
        "results": [],
    }

    # Picking higher bound of neighbors. Idk what it refers to, in general
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.point.conv_radius) ** 3))
    neighborhood_limits = [hist_n] * 5

    print(f" | Start processing...")
    print()

    # Process and collate pair
    for pair in data_config["pairs"]:
        # Reset GPU memory usage data statistics to eliminate any leak
        torch.cuda.reset_peak_memory_stats()

        print(f" | -> Pair: {pair[1]} -> {pair[0]}")
        stats = {"pcd1": pair[0], "pcd2": pair[1]}

        # Load pcd from files
        map1_path = os.path.join(data_config["root"], pair[0])
        map2_path = os.path.join(data_config["root"], pair[1])
        map1_pcd = o3d.io.read_point_cloud(map1_path)
        map2_pcd = o3d.io.read_point_cloud(map2_path)

        # Ensure the order is correct after this point
        tf = compute_gt_pose(map2_path, map1_path)
        theta = Rotation.from_matrix(tf[0:3, 0:3]).as_euler("zyx")[0]

        stats["target_x"] = float(tf[0, 3])
        stats["target_y"] = float(tf[1, 3])
        stats["target_z"] = float(tf[2, 3])
        stats["target_t"] = float(theta)
        print(
            f" | Target x: {tf[0, 3]: .5f} y: {tf[1, 3]: .5f} z: {tf[2, 3]: .5f}, t: {theta: .5f}"
        )

        try:
            with torch.no_grad():
                start = time.time()
                start_data = time.time()
                data = process_pcd_pair(
                    map2_pcd, map1_pcd, tf, config, neighborhood_limits
                )
                end_data = time.time()

                start_algo = time.time()
                tf_est, source_axis, target_axis = model(data)
                end_algo = time.time()
                end = time.time()

                stats["t_data_processing"] = end_data - start_data
                stats["t_registration"] = end_algo - start_algo
                stats["t_total"] = end - start

                theta_est = Rotation.from_matrix(tf_est[0:3, 0:3]).as_euler("zyx")[0]

                stats["result_x"] = float(tf_est[0, 3])
                stats["result_y"] = float(tf_est[1, 3])
                stats["result_z"] = float(tf_est[2, 3])
                stats["result_t"] = float(theta_est)

                print(
                    f" | Result x: {tf_est[0, 3]: .5f} y: {tf_est[1, 3]: .5f} z: {tf_est[2, 3]: .5f}, t: {theta_est: .5f}"
                )

                # # compute error
                rte, rre = compute_error(tf, tf_est)
                stats["error_position"] = rte
                stats["error_angle"] = rre

                print(
                    f" | Error: {rte:.5f}m / {rre:.5f}rad. Took {stats['t_total']:.5f}s"
                )

        except Exception as e:
            print(f" | Cannot process. Reason: {e}")
            stats["exception"] = str(e)

        finally:
            stats["mem_cpu"] = p.memory_info().rss
            stats["mem_gpu"] = torch.cuda.max_memory_allocated()

            # Save results after each run
            output_data["results"].append(stats)
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            print()

            torch.cuda.empty_cache()

    print(" | Completed!")


if __name__ == "__main__":
    main()
