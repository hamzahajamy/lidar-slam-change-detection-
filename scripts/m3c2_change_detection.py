"""
M3C2 Change Detection
=======================
Computes Multiscale Model-to-Model Cloud Comparison (M3C2) distances
between a registered SLAM map and a reference map, classifying each
core point as Unchanged, Added, Removed, or No significant change
based on a 95% confidence Level of Detection (LOD) and an operational
distance threshold.

This was my primary contribution to the project, alongside the
registration pipeline: implementing the full M3C2 workflow used to
produce the final change classification and statistics.

Source: Project Seminar 2025, "Classification, Localization and Mapping
of LiDAR 3D Point Clouds", IKG, Leibniz University Hannover.
"""

import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)


def load_and_preprocess(file_path, voxel_size=None, remove_outliers=True):
    """Load a point cloud, optionally remove outliers and downsample, then estimate normals."""
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Loaded {file_path}: {len(pcd.points)} points")

    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"After outlier removal: {len(pcd.points)} points")

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"After downsampling: {len(pcd.points)} points")

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return pcd


reference_map = load_and_preprocess("data/reference_labelled_map.pcd", voxel_size=0.05)
registered_map = load_and_preprocess("output/final_map_registration.ply", voxel_size=0.05)


def select_core_points(pcd, voxel_size=0.2):
    """Select a sparse, evenly-spaced set of core points via voxel downsampling."""
    core_points = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Selected {len(core_points.points)} core points")
    return core_points


def calculate_multiscale_normals(pcd, core_points, scales):
    """Estimate the most stable surface normal at each core point across multiple radii."""
    core_points_np = np.asarray(core_points.points)
    pcd_points_np = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = np.zeros((len(core_points_np), 3))

    for i, point in enumerate(tqdm(core_points_np, desc="Estimating normals")):
        scale_eigenvalues, scale_eigenvectors = [], []

        for scale in scales:
            [_, idx, _] = pcd_tree.search_radius_vector_3d(point, scale)
            if len(idx) < 3:
                continue

            neighbors = pcd_points_np[idx, :]
            cov = np.cov(neighbors.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx_sort = eigenvalues.argsort()
            scale_eigenvalues.append(eigenvalues[idx_sort])
            scale_eigenvectors.append(eigenvectors[:, idx_sort])

        if not scale_eigenvalues:
            normals[i] = [0, 0, 1]
            continue

        # Pick the scale with the most stable (lowest eigenvalue ratio) normal
        stability = [ev[0] / (ev[1] + 1e-10) for ev in scale_eigenvalues]
        best_scale_idx = np.argmin(stability)
        normals[i] = scale_eigenvectors[best_scale_idx][:, 0]

        centroid = np.mean(pcd_points_np, axis=0)
        if np.dot(normals[i], point - centroid) < 0:
            normals[i] = -normals[i]

    return normals


def calculate_m3c2_distances(reference_pcd, registered_pcd, core_points, normals,
                              cylinder_radius=0.3, max_distance=1.5):
    """Project neighbouring points from both epochs onto the local normal and compute
    the signed distance, standard deviation, and 95%-confidence significance flag."""
    core_points_np = np.asarray(core_points.points)
    reference_points_np = np.asarray(reference_pcd.points)
    registered_points_np = np.asarray(registered_pcd.points)

    reference_tree = o3d.geometry.KDTreeFlann(reference_pcd)
    registered_tree = o3d.geometry.KDTreeFlann(registered_pcd)

    distances = np.zeros(len(core_points_np))
    std_devs = np.zeros(len(core_points_np))
    significant_change = np.zeros(len(core_points_np), dtype=bool)

    for i, (point, normal) in enumerate(tqdm(zip(core_points_np, normals), desc="Computing M3C2 distances")):
        [_, ref_idx, _] = reference_tree.search_radius_vector_3d(point, cylinder_radius)
        [_, reg_idx, _] = registered_tree.search_radius_vector_3d(point, cylinder_radius)

        if len(ref_idx) < 5 or len(reg_idx) < 5:
            continue

        ref_cylinder_points = reference_points_np[ref_idx]
        reg_cylinder_points = registered_points_np[reg_idx]

        ref_projections = np.dot(ref_cylinder_points - point, normal)
        reg_projections = np.dot(reg_cylinder_points - point, normal)

        ref_avg, reg_avg = np.mean(ref_projections), np.mean(reg_projections)
        distance = reg_avg - ref_avg

        combined_std = np.sqrt(np.std(ref_projections) ** 2 + np.std(reg_projections) ** 2)

        distances[i] = distance
        std_devs[i] = combined_std
        # 95% confidence Level of Detection (LOD95)
        significant_change[i] = abs(distance) > 1.96 * combined_std and abs(distance) < max_distance

    return distances, std_devs, significant_change


def classify_changes(distances, significant_change, threshold=0.1):
    """Classify each core point as Unchanged, Added, Removed, or No significant change."""
    classification = np.zeros(len(distances), dtype=int)

    for i, (distance, significant) in enumerate(zip(distances, significant_change)):
        if not significant:
            classification[i] = 3  # No significant change
        elif abs(distance) <= threshold:
            classification[i] = 0  # Unchanged
        elif distance > threshold:
            classification[i] = 2  # Removed
        else:
            classification[i] = 1  # Added

    counts = np.bincount(classification, minlength=4)
    labels = ["Unchanged", "Added", "Removed", "No significant change"]
    for label, count in zip(labels, counts):
        print(f"  {label}: {count} points")

    return classification


# ---- M3C2 parameters ----
core_point_voxel_size = 0.2
normal_scales = [0.2, 0.5]
cylinder_radius = 0.3
max_distance = 1.5
change_threshold = 0.1

core_points = select_core_points(reference_map, voxel_size=core_point_voxel_size)
normals = calculate_multiscale_normals(reference_map, core_points, normal_scales)
core_points.normals = o3d.utility.Vector3dVector(normals)

distances, std_devs, significant_change = calculate_m3c2_distances(
    reference_map, registered_map, core_points, normals, cylinder_radius, max_distance
)

classification = classify_changes(distances, significant_change, threshold=change_threshold)

# ---- Colour-coded visualisation ----
result_cloud = o3d.geometry.PointCloud(core_points)
colors = np.zeros((len(classification), 3))
colors[classification == 0] = [0, 1, 0]        # Unchanged: green
colors[classification == 1] = [1.0, 0.5, 0.0]  # Added: orange
colors[classification == 2] = [0.0, 0.4, 1.0]  # Removed: blue
colors[classification == 3] = [0.7, 0.7, 0.7]  # No significant change: grey
result_cloud.colors = o3d.utility.Vector3dVector(colors)

# ---- Summary statistics and plots ----
results_df = pd.DataFrame({
    "Distance": distances, "StdDev": std_devs,
    "Significant": significant_change, "Classification": classification
})
print("\nSummary statistics:\n", results_df.describe())

plt.figure(figsize=(10, 6))
category_counts = results_df["Classification"].value_counts().sort_index()
category_names = ["Unchanged", "Added", "Removed", "No significant change"]
plt.pie(category_counts, labels=[category_names[i] for i in category_counts.index],
        autopct="%1.1f%%", colors=["green", "orange", "blue", "gray"])
plt.title("Distribution of Change Categories")
plt.savefig("output/change_category_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 6))
plt.hist(distances, bins=50, alpha=0.7, color="skyblue")
plt.axvline(x=0, color="black", linestyle="--")
plt.axvline(x=change_threshold, color="green", linestyle="--", label=f"Threshold (+{change_threshold})")
plt.axvline(x=-change_threshold, color="red", linestyle="--", label=f"Threshold (-{change_threshold})")
plt.xlabel("M3C2 Distance")
plt.ylabel("Frequency")
plt.title("Distribution of M3C2 Distances")
plt.legend()
plt.savefig("output/m3c2_distance_histogram.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Save outputs ----
o3d.io.write_point_cloud("output/change_detection_results.ply", result_cloud)
results_df.to_csv("output/change_detection_results.csv", index=False)
print("\nSaved classified point cloud and results CSV to output/")
