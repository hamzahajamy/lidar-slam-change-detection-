"""
Point Cloud Registration: RANSAC (coarse) + ICP (fine)
========================================================
Registers the KISS-SLAM-generated map (source) to a labelled reference
map (target) using a coarse-to-fine pipeline:
  1. Preprocessing (downsampling, normal estimation, outlier removal)
  2. FPFH feature computation
  3. RANSAC-based global registration
  4. ICP refinement (point-to-point and point-to-plane)
  5. Quality evaluation (fitness, inlier RMSE)

This was my primary contribution to the project: parameterising and
running the full registration pipeline used as the basis for all
downstream change-detection analysis.

Source: Project Seminar 2025, "Classification, Localization and Mapping
of LiDAR 3D Point Clouds", IKG, Leibniz University Hannover.
"""

import numpy as np
import open3d as o3d
import copy
import os

source_file = "data/slam_map.ply"
target_file = "data/reference_labelled_map.pcd"

if not os.path.exists(source_file):
    raise FileNotFoundError(f"Source file not found: {source_file}")
if not os.path.exists(target_file):
    raise FileNotFoundError(f"Target file not found: {target_file}")

source = o3d.io.read_point_cloud(source_file)
target = o3d.io.read_point_cloud(target_file)

print("Source point cloud points:", len(source.points))
print("Target point cloud points:", len(target.points))


def preprocess_point_cloud(pcd, voxel_size=0.5):
    """Downsample, estimate normals, and remove statistical outliers."""
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    if not pcd_down.has_normals():
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
        )

    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.8)
    return pcd_down.select_by_index(ind)


source_original = copy.deepcopy(source)

# Coarse registration uses a larger voxel size; fine registration uses a smaller one
ransac_voxel_size = 0.40
icp_voxel_size = 0.10

source_down_coarse = preprocess_point_cloud(source, ransac_voxel_size)
target_down_coarse = preprocess_point_cloud(target, ransac_voxel_size)

source_down_fine = preprocess_point_cloud(source, icp_voxel_size)
target_down_fine = preprocess_point_cloud(target, icp_voxel_size)


def compute_fpfh_features(pcd, voxel_size):
    """Compute Fast Point Feature Histogram (FPFH) descriptors for global registration."""
    radius_normal = voxel_size * 4
    radius_features = voxel_size * 10

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50)
        )

    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_features, max_nn=200)
    )


source_fpfh = compute_fpfh_features(source_down_coarse, ransac_voxel_size)
target_fpfh = compute_fpfh_features(target_down_coarse, ransac_voxel_size)

# ---- Coarse global registration via RANSAC on FPFH correspondences ----
distance_threshold = ransac_voxel_size * 2.5
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down_coarse, target_down_coarse, source_fpfh, target_fpfh, True, distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    4,
    [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    o3d.pipelines.registration.RANSACConvergenceCriteria(20000000, 0.9999)
)

print("\nGlobal RANSAC registration result:")
print(f"Fitness: {result_ransac.fitness}")
print(f"Inlier RMSE: {result_ransac.inlier_rmse}")


def perform_icp(source, target, initial_transform, method="point_to_point", max_distance=0.1):
    """Run ICP refinement (point-to-point or point-to-plane) from an initial transform."""
    if method == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel_size * 3, max_nn=30)
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel_size * 3, max_nn=30)
            )

    return o3d.pipelines.registration.registration_icp(
        source, target, max_distance, initial_transform, estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, relative_fitness=1e-7)
    )


# ---- Fine refinement: compare point-to-point and point-to-plane ICP ----
result_icp_p2p = perform_icp(
    source_down_fine, target_down_fine, result_ransac.transformation,
    method="point_to_point", max_distance=icp_voxel_size * 1.5
)
print("\nPoint-to-point ICP fitness:", result_icp_p2p.fitness, "| inlier RMSE:", result_icp_p2p.inlier_rmse)

result_icp_p2l = perform_icp(
    source_down_fine, target_down_fine, result_ransac.transformation,
    method="point_to_plane", max_distance=icp_voxel_size * 1.5
)
print("Point-to-plane ICP fitness:", result_icp_p2l.fitness, "| inlier RMSE:", result_icp_p2l.inlier_rmse)

if result_icp_p2p.fitness > result_icp_p2l.fitness:
    final_result = result_icp_p2p
    method_name = "point-to-point"
else:
    final_result = result_icp_p2l
    method_name = "point-to-plane"

print(f"\nBest method: {method_name} ICP | Fitness: {final_result.fitness} | Inlier RMSE: {final_result.inlier_rmse}")

# ---- Apply the final transformation to the full-resolution source cloud ----
combined_transformation = final_result.transformation
source_original_aligned = copy.deepcopy(source_original)
source_original_aligned.transform(combined_transformation)

aligned_file = "output/final_map_registration.ply"
o3d.io.write_point_cloud(aligned_file, source_original_aligned)
print(f"Aligned map saved to: {aligned_file}")

np.savetxt("output/final_transformation_matrix.txt", combined_transformation)
print("Transformation matrix saved to: output/final_transformation_matrix.txt")
