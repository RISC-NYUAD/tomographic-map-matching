import open3d as o3d
import numpy as np
import argparse
from scipy.spatial.transform import Rotation


def construct_tf_from_parameters(x, y, z, t):
    R = Rotation.from_euler("z", t).as_matrix()
    t = np.array([[x], [y], [z]], dtype=float)
    tf = np.vstack((np.hstack((R, t)), np.zeros((1, 4), dtype=float)))
    tf[3, 3] = 1.0
    print("tf:")
    print(tf)
    return tf


def retrieve_parameters_from_tf(tf):
    t = Rotation.from_matrix(tf[:3, :3]).as_euler("zyx")[0]
    x, y, z = tf[:3, 3]
    print(f"x: {x: .5f}, y: {y: .5f}, z: {z: .5f}, t: {t: .5f}")
    return x, y, z, t

    R = Rotation.from_euler("z", t).as_matrix()
    t = np.array([[x], [y], [z]], dtype=float)
    tf = np.vstack((np.hstack((R, t)), np.zeros((1, 4), dtype=float)))
    tf[3, 3] = 1.0
    print("tf:")
    print(tf)
    return tf


def load_pcd_from_path(path, color):
    pcd = o3d.io.read_point_cloud(path)
    pcd.paint_uniform_color(color)
    pcd.estimate_normals()

    pose_file = path[:-4] + "-gtpose.txt"
    pose = np.vstack(
        (np.loadtxt(pose_file).reshape(3, 4), np.zeros((1, 4), dtype=float))
    )
    pose[3, 3] = 1.0

    return {"pcd": pcd, "pose": pose}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map1", type=str, required=True, help="Path to first map")
    parser.add_argument("--map2", type=str, required=True, help="Path to second map")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Perform pose refinement with point to plane ICP",
    )
    args = parser.parse_args()

    # Load maps
    map1 = load_pcd_from_path(args.map1, [0.8, 0.0, 0.0])
    map2 = load_pcd_from_path(args.map2, [0.0, 0.0, 0.8])

    # Visualize 2 -> 1
    tf_init = map1["pose"] @ np.linalg.inv(map2["pose"])
    theta = Rotation.from_matrix(tf_init[0:3, 0:3]).as_euler("zyx")[0]
    print(
        f"Target x: {tf_init[0, 3]: .5f} y: {tf_init[1, 3]: .5f} z: {tf_init[2, 3]: .5f}, t: {theta: .5f}"
    )
    map2_init = o3d.geometry.PointCloud(map2["pcd"])
    map2_init.transform(tf_init)
    o3d.visualization.draw_geometries([map1["pcd"], map2_init])

    if args.refine:
        # Refine:
        threshold = 0.2
        icp_reg = o3d.pipelines.registration.registration_icp(
            map2["pcd"],
            map1["pcd"],
            threshold,
            tf_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        tf = icp_reg.transformation.copy()
        theta = Rotation.from_matrix(tf[0:3, 0:3]).as_euler("zyx")[0]

        print(
            f"Refined x: {tf[0, 3]: .5f} y: {tf[1, 3]: .5f} z: {tf[2, 3]: .5f}, t: {theta: .5f}"
        )

        map2["pcd"].transform(tf)

        o3d.visualization.draw_geometries([map1["pcd"], map2["pcd"]])


if __name__ == "__main__":
    main()
