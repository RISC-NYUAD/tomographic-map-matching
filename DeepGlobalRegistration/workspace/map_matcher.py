import argparse
import json
import os
import sys
import time

from easydict import EasyDict
import numpy as np
import open3d as o3d
import psutil
from scipy.spatial.transform import Rotation
import torch

sys.path.append("/usr/local/src/DeepGlobalRegistration")

from core.deep_global_registration import DeepGlobalRegistration

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


def visualize(map1_pcd, map2_pcd, tf):
    # Colorize
    map1_pcd.estimate_normals()
    map1_pcd.paint_uniform_color([0.7, 0.0, 0.0])

    map2_pcd.estimate_normals()
    map2_pcd.paint_uniform_color([0.0, 0.7, 0.0])

    map2_pcd.transform(tf)

    o3d.visualization.draw_geometries([map1_pcd, map2_pcd])


def main():
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        required=True,
        help="Path to JSON file that provides model hyperparameters",
    )
    parser.add_argument(
        "--data_config", required=True, help="Path to JSON file that denotes pairs"
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    # Load model
    with open(args.model_config, "r") as f:
        model_config = EasyDict(json.load(f))

    model = DeepGlobalRegistration(model_config)

    # Override voxel size
    model.voxel_size = model_config.voxel_size

    # Deactivate ICP refinement as it is not used in any other method
    model.icp = False

    print(" | Model init. complete")

    # Load data config
    with open(args.data_config, "r") as f:
        data_config = json.load(f)

    print(
        f" | Data root: {data_config['root']}. Num. pairs: {len(data_config['pairs'])}"
    )

    algorithm_name = "dgr"
    experiment_name_from_data = "_".join(args.data_config[:-5].split("/")[-2:])
    output_file = os.path.join(
        "/results",
        algorithm_name + "_" + experiment_name_from_data + "_" + time_string + ".json",
    )

    # JSON to store results
    output_data = {
        "data_config": args.data_config,
        "full_configuration": dict(model_config),
        "results": [],
    }

    print(" | Start processing...")
    print()

    for pair in data_config["pairs"]:
        # Reset GPU memory usage between pairs
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
            # with torch.no_grad():  # DGR requires grad in global registration
            start = time.time()
            tf_est = model.register(map2_pcd, map1_pcd).copy()
            end = time.time()

            stats["t_total"] = end - start
            theta_est = Rotation.from_matrix(tf_est[0:3, 0:3]).as_euler("zyx")[0]

            stats["result_x"] = float(tf_est[0, 3])
            stats["result_y"] = float(tf_est[1, 3])
            stats["result_z"] = float(tf_est[2, 3])
            stats["result_t"] = float(theta_est)

            print(
                f" | Result x: {tf_est[0, 3]: .5f} y: {tf_est[1, 3]: .5f} z: {tf_est[2, 3]: .5f}, t: {theta_est: .5f}"
            )

            # compute error
            rte, rre = compute_error(tf, tf_est)
            stats["error_position"] = rte
            stats["error_angle"] = rre

            print(f" | Error: {rte:.5f}m / {rre:.5f}rad. Took {stats['t_total']:.5f}s")

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
