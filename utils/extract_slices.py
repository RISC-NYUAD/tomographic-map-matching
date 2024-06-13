import open3d as o3d
import numpy as np
import argparse
import cv2


def generate_rectangle_outline(pts_min, pts_max, z, color, padding=0.1):
    # Convert to padded coordinates
    xmin = pts_min[0] - padding
    xmax = pts_max[0] + padding

    ymin = pts_min[1] - padding
    ymax = pts_max[1] + padding

    # Line coordinates
    pts = np.asarray(
        [
            [xmin, ymin, z],
            [xmin, ymax, z],
            [xmax, ymax, z],
            [xmax, ymin, z],
        ],
        dtype=float,
    )
    lines = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ],
        dtype=int,
    )

    colors = np.asarray([color] * 4, dtype=float)

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(pts)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset


def extract_points_for_slice(pts, height, thickness):
    idxs = (pts[:, 2] > height - thickness) * (pts[:, 2] < height + thickness)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[idxs])
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    return pcd


def pcd_to_image(pcd, grid_size, px_padding=3):
    pts = np.asarray(pcd.points)
    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)

    n_cols = int(np.ceil((pts_max[0] - pts_min[0]) / grid_size) + px_padding * 2)
    n_rows = int(np.ceil((pts_max[1] - pts_min[1]) / grid_size) + px_padding * 2)

    img = np.ones((n_cols, n_rows), dtype=np.uint8) * 255

    for pt in pts:
        x = int(np.ceil((pt[0] - pts_min[0]) / grid_size)) + px_padding
        y = int(np.ceil((pt[1] - pts_min[1]) / grid_size)) + px_padding
        img[x, y] = 0

    # Correct the alignment and scale
    img = cv2.resize(
        cv2.flip(img.T, 0), (n_cols * 2, n_rows * 2), interpolation=cv2.INTER_NEAREST
    )
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the point cloud"
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.02, help="Grid size for projections"
    )
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.path)
    pcd.estimate_normals()

    # Select lower and upper
    pts = np.asarray(pcd.points)
    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)

    zmin = pts_min[2]
    zmax = pts_max[2]
    zrange = zmax - zmin
    h_lower = zmin + zrange * 0.2
    h_upper = zmin + zrange * 0.8

    # Points
    pcd_lower = extract_points_for_slice(pts, h_lower, args.grid_size)
    pcd_upper = extract_points_for_slice(pts, h_upper, args.grid_size)

    # A bounding rectangle for each
    lines_lower = generate_rectangle_outline(pts_min, pts_max, h_lower, [0.8, 0.0, 0.0])
    lines_upper = generate_rectangle_outline(pts_min, pts_max, h_upper, [0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries(
        [pcd, pcd_lower, pcd_upper, lines_lower, lines_upper]
    )

    img_lower = pcd_to_image(pcd_lower, args.grid_size)
    img_upper = pcd_to_image(pcd_upper, args.grid_size)

    # cv2.imshow("lower", img_lower)
    # cv2.imshow("upper", img_upper)
    # cv2.waitKey(0)

    cv2.imwrite("lower.png", img_lower)
    cv2.imwrite("upper.png", img_upper)


if __name__ == "__main__":
    main()
