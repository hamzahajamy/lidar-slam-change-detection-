"""
RMSE and Axis-wise Deviation Analysis
======================================
Evaluates the accuracy of the SLAM-generated map against a high-precision
reference dataset using KD-tree nearest-neighbour search, with axis-wise
(X/Y/Z) deviation breakdown and a colour-coded deviation heatmap.

Source: Project Seminar 2025, "Classification, Localization and Mapping
of LiDAR 3D Point Clouds", IKG, Leibniz University Hannover.
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Load point clouds ----
ref_path = "data/reference_labelled_map.pcd"
slam_path = "data/final_map_registration.pcd"

pcd_ref = o3d.io.read_point_cloud(ref_path)
pcd_slam = o3d.io.read_point_cloud(slam_path)

if len(pcd_ref.points) == 0 or len(pcd_slam.points) == 0:
    raise ValueError("One of the point clouds is empty. Check file paths and formats.")

print(f"Reference points: {len(pcd_ref.points):,}")
print(f"SLAM points: {len(pcd_slam.points):,}")

# ---- 2. RMSE calculation via nearest-neighbour search ----
ref_kdtree = o3d.geometry.KDTreeFlann(pcd_ref)
slam_points = np.asarray(pcd_slam.points)

distances = []
for pt in slam_points:
    _, idx, dists = ref_kdtree.search_knn_vector_3d(pt, 1)
    distances.append(np.sqrt(dists[0]))

distances = np.array(distances)
rmse = np.sqrt(np.mean(distances ** 2))
print(f"\nRMSE (SLAM to Reference): {rmse:.4f} meters")

# ---- 3. Deviation heatmap ----
max_dist = np.percentile(distances, 99)
colors = plt.cm.jet(np.clip(distances / max_dist, 0, 1))[:, :3]
pcd_slam.colors = o3d.utility.Vector3dVector(colors)

output_file = "output/slam_colored_deviation.ply"
o3d.io.write_point_cloud(output_file, pcd_slam)
print(f"Saved color-coded SLAM point cloud to: {output_file}")

# ---- 4. Axis-wise deviation analysis ----
ref_points = np.asarray(pcd_ref.points)
x_dev, y_dev, z_dev = [], [], []

for pt in slam_points:
    _, idx, _ = ref_kdtree.search_knn_vector_3d(pt, 1)
    ref_pt = ref_points[idx[0]]
    x_dev.append(pt[0] - ref_pt[0])
    y_dev.append(pt[1] - ref_pt[1])
    z_dev.append(pt[2] - ref_pt[2])

x_dev, y_dev, z_dev = np.array(x_dev), np.array(y_dev), np.array(z_dev)


def print_axis_stats(name, values):
    print(f"{name}-Axis Deviation:")
    print(f"  Mean: {np.mean(values):.4f}")
    print(f"  Std : {np.std(values):.4f}")
    print(f"  Min : {np.min(values):.4f}")
    print(f"  Max : {np.max(values):.4f}")
    print(f"  RMSE: {np.sqrt(np.mean(values ** 2)):.4f}\n")


print_axis_stats("X", x_dev)
print_axis_stats("Y", y_dev)
print_axis_stats("Z", z_dev)

mean_abs_error = np.mean(distances)
print(f"Mean Absolute Error: {mean_abs_error:.4f} meters")
