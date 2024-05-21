import argparse
import json
import os
import psutil
import sys
import time

import easydict
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch

sys.path.append("/usr/local/src/RoITr")

from lib.utils import setup_seed
from model.RIGA_v2 import RIGA_v2
from configs.utils import load_config
from dataset.common import collate_fn
from registration.benchmark_utils import ransac_pose_estimation_correspondences

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


def preprocess_pcd(pcd_raw, max_pts, knn):
    pts = np.asarray(pcd_raw.points)
    if pts.shape[0] > max_pts:
        idx = np.random.permutation(pts.shape[0])[:max_pts]
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_towards_camera_location()

    normals = np.asarray(pcd.normals)
    feats = np.ones(shape=(pts.shape[0], 1))
    return pts.astype(np.float32), normals.astype(np.float32), feats.astype(np.float32)


def process_pcd_pair(src_pcd_raw, tgt_pcd_raw, tf, max_pts=30000, knn=33):
    src_pcd, src_normals, src_feats = preprocess_pcd(src_pcd_raw, max_pts, knn)
    tgt_pcd, tgt_normals, tgt_feats = preprocess_pcd(tgt_pcd_raw, max_pts, knn)
    return collate_fn(
        [
            [
                src_pcd,
                tgt_pcd,
                src_normals,
                tgt_normals,
                src_feats,
                tgt_feats,
                tf[:3, :3].astype(np.float32),
                tf[:3, 3][:, None].astype(np.float32),
                src_pcd,
                None,
            ]
        ],
        config=None,
    )


def visualize_output(output):
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(output["src_points"].cpu().numpy())
    src_pcd.paint_uniform_color([0.7, 0.0, 0.0])

    src_corr = o3d.geometry.PointCloud()
    src_corr.points = o3d.utility.Vector3dVector(
        output["src_corr_points"].cpu().numpy()
    )
    src_corr.paint_uniform_color([0.0, 0.7, 0.7])

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(output["tgt_points"].cpu().numpy())
    tgt_pcd.paint_uniform_color([0.0, 0.7, 0.0])

    tgt_corr = o3d.geometry.PointCloud()
    tgt_corr.points = o3d.utility.Vector3dVector(
        output["tgt_corr_points"].cpu().numpy()
    )
    tgt_corr.paint_uniform_color([0.7, 0.0, 0.7])

    o3d.visualization.draw_geometries([src_pcd, tgt_pcd, src_corr, tgt_corr])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True, help="Config for RoITr model")
    parser.add_argument(
        "--data_config", required=True, help="Path to JSON file that denotes pairs"
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    time_string = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Load config
    model_config_dict = load_config(args.model_config)
    model_config_dict["local_rank"] = args.local_rank
    model_config = easydict.EasyDict(model_config_dict)
    torch.cuda.set_device(0)
    model_config.device = torch.device("cuda", 0)

    # Load model & parameters
    model = RIGA_v2(model_config).to(model_config.device)
    state = torch.load(model_config.pretrain)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    )
    model.eval()

    print(" | Model init. complete")

    # Fix seed the same way as original
    # FIXME: Results are still random. Not seeding o3d RANSAC?
    setup_seed(42)

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
        "full_configuration": model_config_dict,
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

        # print(f" | PCD Sizes: {len(map1_pcd.points)}, {len(map2_pcd.points)}")

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
                inputs = process_pcd_pair(
                    map2_pcd, map1_pcd, tf, max_pts=model_config.max_pts
                )

                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(model_config.device) for item in v]
                    elif v is None:
                        inputs[k] = v
                    else:
                        inputs[k] = v.to(model_config.device)

                rot, trans = inputs["rot"][0], inputs["trans"][0]
                src_pcd, tgt_pcd = (
                    inputs["src_points"].contiguous(),
                    inputs["tgt_points"].contiguous(),
                )
                src_normals, tgt_normals = (
                    inputs["src_normals"].contiguous(),
                    inputs["tgt_normals"].contiguous(),
                )
                src_feats, tgt_feats = (
                    inputs["src_feats"].contiguous(),
                    inputs["tgt_feats"].contiguous(),
                )
                src_raw_pcd = inputs["raw_src_pcd"].contiguous()
                end_data = time.time()

                start_algo = time.time()
                output = model.forward(
                    src_pcd,
                    tgt_pcd,
                    src_feats,
                    tgt_feats,
                    src_normals,
                    tgt_normals,
                    rot,
                    trans,
                    src_raw_pcd,
                )

                correspondences = torch.from_numpy(
                    np.arange(output["src_corr_points"].shape[0])[:, np.newaxis]
                ).expand(-1, 2)

                tf_est = ransac_pose_estimation_correspondences(
                    output["src_corr_points"],
                    output["tgt_corr_points"],
                    correspondences,
                ).copy()

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

                # compute error
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
