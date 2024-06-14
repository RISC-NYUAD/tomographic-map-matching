import open3d as o3d
import argparse
import numpy as np


def load_pcd_from_path(path, color=None):
    pcd = o3d.io.read_point_cloud(path)
    if color is not None:
        pcd.paint_uniform_color(color)
    pcd.estimate_normals()

    pose_file = path[:-4] + "-gtpose.txt"
    pose = np.vstack(
        (np.loadtxt(pose_file).reshape(3, 4), np.zeros((1, 4), dtype=float))
    )
    pose[3, 3] = 1.0
    pcd.transform(np.linalg.inv(pose))

    return {"pcd": pcd, "pose": pose}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the point cloud"
    )
    args = parser.parse_args()
    pcd = load_pcd_from_path(args.path, color=np.array([228, 26, 28]) / 255)
    o3d.visualization.draw_geometries([pcd["pcd"]])


if __name__ == "__main__":
    main()
