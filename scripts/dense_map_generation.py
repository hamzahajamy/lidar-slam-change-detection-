"""
Dense Map Generation from LiDAR Scans and KISS-SLAM Poses
============================================================
Constructs a global 3D point cloud map by applying 6-DoF transformations
(from KISS-SLAM pose estimates) to individual LiDAR scans, accumulating
them into a single unified dense map.

Source: Project Seminar 2025, "Classification, Localization and Mapping
of LiDAR 3D Point Clouds", IKG, Leibniz University Hannover.
"""

import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import glob
from kiss_icp.datasets import dataset_factory

# ---- Paths ----
pose_file_path = Path("data/poses_kitti.txt")
bag_folder = Path("data/measurement_long")
pcd_output_dir = Path("output/pcd_scans")
pcd_output_dir.mkdir(parents=True, exist_ok=True)

NUM_SCANS = 2029

# ---- Step 1: Load poses ----
poses = np.loadtxt(pose_file_path)
if poses.ndim == 1:
    poses = poses.reshape(-1, 12)
if poses.shape[0] != NUM_SCANS:
    raise ValueError(f"Expected {NUM_SCANS} poses, got {poses.shape[0]}")
print("Loaded poses:", poses.shape[0])

pose_list = [np.vstack([row.reshape(3, 4), [0, 0, 0, 1]]) for row in poses]

# ---- Step 2: Load scans from ROS bag and save as individual PCD files ----
dataset = dataset_factory(dataloader="rosbag", data_dir=bag_folder, topic="/ouster/points")
scans = [scan for scan, _ in dataset]
if len(scans) < NUM_SCANS:
    raise RuntimeError(f"Expected at least {NUM_SCANS} scans, got {len(scans)}")
print("Loaded scans from rosbag:", len(scans))

for i, scan in enumerate(tqdm(scans[:NUM_SCANS], desc="Saving PCD scans")):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan)
    o3d.io.write_point_cloud(str(pcd_output_dir / f"{i:06d}.pcd"), pcd)
print("Saved PCD scans to:", pcd_output_dir)

# ---- Step 3: Transform each scan into the global frame and accumulate ----
pcd_files = sorted(glob.glob(str(pcd_output_dir / "*.pcd")))
if len(pcd_files) != NUM_SCANS:
    raise RuntimeError(f"Expected {NUM_SCANS} .pcd files, got {len(pcd_files)}")

all_points = []
total_points = 0
for i in tqdm(range(NUM_SCANS), desc="Transforming scans"):
    pcd = o3d.io.read_point_cloud(pcd_files[i])
    scan = np.asarray(pcd.points)

    pose = pose_list[i]
    R = pose[:3, :3]
    t = pose[:3, 3]
    transformed = (R @ scan.T).T + t

    all_points.append(transformed)
    total_points += transformed.shape[0]

print("All scans transformed:", len(all_points))
print("Total points accumulated:", total_points)

# ---- Step 4: Save the final dense map ----
final_points = np.vstack(all_points)
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(final_points)

output_path = Path("output/final_dense_map.pcd")
o3d.io.write_point_cloud(str(output_path), final_pcd)
print("Final map saved to:", output_path)
