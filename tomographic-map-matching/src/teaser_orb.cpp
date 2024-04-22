#include <algorithm>
#include <iterator>
#include <memory>
#include <opencv2/core/types.hpp>
#include <spdlog/spdlog.h>
#include <teaser/registration.h>
#include <tomographic_map_matching/teaser_orb.hpp>

namespace map_matcher {

TeaserORB::TeaserORB(TeaserORBParameters parameters)
    : MapMatcherBase(static_cast<Parameters>(parameters)),
      parameters_(parameters) {}

void TeaserORB::PrintParameters() const {
  spdlog::info("[PARAMS] grid_size: {} threshold: {} orb_num_features: {} "
               "orb_scale_factor: {} orb_n_levels: {} orb_edge_threshold: {} "
               "orb_first_level: "
               "{} orb_wta_k: {} orb_patch_size: {} orb_fast_threshold: {} "
               "gms_threshold_factor: {} lsh_num_tables: {} lsh_key_size: {} "
               "lsh_multiprobe_level: {} minimum_z_overlap_percentage: {} "
               "teaser_noise_bound: "
               "{} teaser_maximum_correspondences: {}",
               parameters_.grid_size, parameters_.slice_z_height,
               parameters_.orb_num_features, parameters_.orb_scale_factor,
               parameters_.orb_n_levels, parameters_.orb_edge_threshold,
               parameters_.orb_first_level, parameters_.orb_wta_k,
               parameters_.orb_patch_size, parameters_.orb_fast_threshold,
               parameters_.gms_threshold_factor, parameters_.lsh_num_tables,
               parameters_.lsh_key_size, parameters_.lsh_multiprobe_level,
               parameters_.minimum_z_overlap_percentage,
               parameters_.teaser_noise_bound,
               parameters_.teaser_maximum_correspondences);
  spdlog::info("[FLAGS] cross_match: {} median_filter: {} gms_matching: {} "
               "approximate_neighbors: {} teaser_3d: {}",
               parameters_.cross_match, parameters_.median_filter,
               parameters_.gms_matching, parameters_.approximate_neighbors,
               parameters_.teaser_3d);
}

HypothesisPtr
TeaserORB::RegisterPointCloudMaps(const PointCloud::Ptr map1_pcd,
                                  const PointCloud::Ptr map2_pcd) const {
  spdlog::info("Number of points map1_size: {} map2_size: {}", map1_pcd->size(),
               map2_pcd->size());

  if (map1_pcd->size() == 0 or map2_pcd->size() == 0) {
    spdlog::critical("Pointcloud(s) are empty. Aborting");
    return HypothesisPtr(new Hypothesis());
  }

  // Timing
  std::chrono::steady_clock::time_point total, indiv;
  total = std::chrono::steady_clock::now();
  indiv = std::chrono::steady_clock::now();

  // Calculate & store all possible binary images (determined by grid size)
  std::vector<SlicePtr> map1_slice = ComputeSliceImages(map1_pcd),
                        map2_slice = ComputeSliceImages(map2_pcd);

  spdlog::info(
      "[TIMING] Binary occupancy image generation t_image_generation: {}",
      CalculateTimeSince(indiv));
  spdlog::info("Number of slices map1_num_slices: {} map2_num_slices: {}",
               map1_slice.size(), map2_slice.size());

  indiv = std::chrono::steady_clock::now();

  // Convert binary images to feature slices
  ComputeSliceFeatures(map1_slice);
  ComputeSliceFeatures(map2_slice);

  // Calculate number of features in each map
  size_t map1_nfeat = 0, map2_nfeat = 0;
  for (const auto &slice : map1_slice)
    map1_nfeat += slice->kp.size();
  for (const auto &slice : map2_slice)
    map2_nfeat += slice->kp.size();

  spdlog::info("[TIMING] ORB Feature extraction t_feature_extraction: {}",
               CalculateTimeSince(indiv));
  spdlog::info("Number of features map1_num_features: {} map2_num_features: {}",
               map1_nfeat, map2_nfeat);
  indiv = std::chrono::steady_clock::now();

  HypothesisPtr result_unrefined;

  if (parameters_.teaser_3d) {
    // Method 2: Compare features across all slices (in 3D)
    result_unrefined = RunTeaserWith3DMatches(map1_slice, map2_slice);
    spdlog::info("[TIMING] Teaser++ on 3D matches t_pose_estimation: {}",
                 CalculateTimeSince(indiv));
  } else {
    // Method 1: Similar to correlations before, using
    std::vector<HypothesisPtr> correlation_results =
        CorrelateSlices(map1_slice, map2_slice);
    result_unrefined = correlation_results[0];
    spdlog::info("[TIMING] Slice correlation t_pose_estimation: {}",
                 CalculateTimeSince(indiv));
  }

  spdlog::info("Computed pose x: {} y: {} z: {} t: {}", result_unrefined->x,
               result_unrefined->y, result_unrefined->z,
               result_unrefined->theta);
  spdlog::info("[HYPOTHESES] Inlier hypothesis count num_inliers: {}",
               result_unrefined->n_inliers);
  indiv = std::chrono::steady_clock::now();

  // Storing the same pointer to the unrefined result, if there is no refinement
  // to be performed
  HypothesisPtr result_refined(result_unrefined);

  if (result_unrefined->n_inliers == 0) {
    spdlog::warn("Pose cannot be calculated");
  } else {
    // Spread analysis
    PointT spread = ComputeResultSpread(result_unrefined);
    spdlog::info("[TIMING] Spread analysis t_spread_analysis: {}",
                 CalculateTimeSince(indiv));
    spdlog::info("[SPREAD] spread_ax1: {} spread_ax2: {} spread_axz: {}",
                 spread.x, spread.y, spread.z);
    spdlog::info("[HYPOTHESES] Num. of features agreeing with result "
                 "num_inlier_features: {}",
                 result_unrefined->inlier_points_1->size());
    indiv = std::chrono::steady_clock::now();

    // Skip refinement and do not print timing info related to it
    if (parameters_.icp_refinement) {
      result_refined = RefineResult(result_unrefined);
      spdlog::info("[TIMING] ICP refinement: {}", CalculateTimeSince(indiv));
      spdlog::info("Refined pose x: {} y: {} z: {} t: {} icp: {}",
                   result_refined->x, result_refined->y, result_refined->z,
                   result_refined->theta, parameters_.icp_refinement);
    }
  }
  spdlog::info("[TIMING] Total runtime t_total: {}", CalculateTimeSince(total));

  // Measure memory use
  size_t peak = GetPeakRSS();
  spdlog::info("[MEMORY] RSS memory_usage_cpu: {}", peak);

  return result_refined;
}

std::vector<HypothesisPtr>
TeaserORB::CorrelateSlices(const std::vector<SlicePtr> &map1_features,
                           const std::vector<SlicePtr> &map2_features) const {
  // Number of possibilities (unless restricted) for slice pairings is n1 + n2 -
  // 1 Starting from bottom slice of m2 and top slice of m1 only, all the way to
  // the other way around. Manipulate index ranges
  size_t map1_index = 0, map2_index = map2_features.size() - 1;
  const size_t map1_size = map1_features.size(),
               map2_size = map2_features.size();

  // Only consider overlaps of particular percentage
  const size_t minimum_overlap = static_cast<size_t>(
      std::round(parameters_.minimum_z_overlap_percentage *
                 static_cast<double>(std::min(map1_size, map2_size))));

  std::vector<HypothesisPtr> correlated_results;
  size_t count = 0;

  while (!(map1_index == map1_size && map2_index == 0)) {
    // Height is determined by whichever has the smaller number of slices after
    // the index remaining between the two
    size_t height = std::min(map1_size - map1_index, map2_size - map2_index);

    if (height >= minimum_overlap) {
      HeightIndices indices{map1_index, map1_index + height, map2_index,
                            map2_index + height};

      HypothesisPtr hypothesis =
          RegisterForGivenInterval(map1_features, map2_features, indices);
      correlated_results.push_back(hypothesis);
    }

    // Update indices
    count++;
    if (map2_index != 0)
      --map2_index;
    else
      ++map1_index;
  }

  spdlog::info("Number of correlations num_correlation: {}", count);

  // Providing a lambda sorting function to deal with the use of smart
  // pointers. Otherwise sorted value is not exactly accurate
  std::sort(
      correlated_results.rbegin(), correlated_results.rend(),
      [](HypothesisPtr val1, HypothesisPtr val2) { return *val1 < *val2; });
  return correlated_results;
}

HypothesisPtr
TeaserORB::RegisterForGivenInterval(const std::vector<SlicePtr> &map1,
                                    const std::vector<SlicePtr> &map2,
                                    HeightIndices indices) const {
  if (indices.m2_max - indices.m2_min != indices.m1_max - indices.m1_min) {
    spdlog::critical("Different number of slices are sent for calculation");
    throw std::runtime_error("Different number of slices");
  }

  size_t window = std::min(indices.m2_max - indices.m2_min,
                           indices.m1_max - indices.m1_min);

  // Aggregate matching features from all slices in the given range
  PointCloud::Ptr map1_points(new PointCloud()), map2_points(new PointCloud());
  std::vector<float> distances;

  for (size_t i = 0; i < window; ++i) {
    // Extract correct slice & assign the weak ptrs
    size_t m1_idx = indices.m1_min + i, m2_idx = indices.m2_min + i;
    const Slice &slice1 = *map1[m1_idx], slice2 = *map2[m2_idx];

    if (slice1.kp.size() < 2 || slice2.kp.size() < 2) {
      spdlog::debug(
          "Not enough keypoints in slices: m1_idx: {} kp: {} m2_idx: {} kp: {}",
          m1_idx, slice1.kp.size(), m2_idx, slice2.kp.size());
      continue;
    }

    // Extract matching keypoints
    MatchingResultPtr matched_keypoints;
    if (parameters_.gms_matching) {
      matched_keypoints = MatchKeyPointsGMS(slice1, slice2);
    } else {
      matched_keypoints = MatchKeyPoints(slice1, slice2);
    }

    // Convert to image coordinates
    std::vector<cv::Point2f> points1img, points2img;
    cv::KeyPoint::convert(matched_keypoints->map1_keypoints, points1img);
    cv::KeyPoint::convert(matched_keypoints->map2_keypoints, points2img);

    // Convert to real coordinates (3D)
    PointCloud::Ptr points1 = img2real(points1img, slice1.slice_bounds,
                                       slice1.height),
                    points2 = img2real(points2img, slice2.slice_bounds,
                                       slice2.height);

    // Append to collective cloud
    *map1_points += *points1;
    *map2_points += *points2;
    distances.insert(distances.end(), matched_keypoints->distances.begin(),
                     matched_keypoints->distances.end());
  }

  // Teaser++ registration on top N matches
  SelectTopNMatches(map1_points, map2_points, distances);
  spdlog::debug("Number of correspondences: {}", map1_points->size());

  if (map1_points->size() < 5) {
    spdlog::debug("Not enough correspondences");
    return HypothesisPtr(new Hypothesis());
  }

  std::shared_ptr<teaser::RobustRegistrationSolver> solver =
      RegisterPointsWithTeaser(map1_points, map2_points);

  // Construct solution
  HypothesisPtr result =
      ConstructSolutionFromSolverState(solver, map1_points, map2_points);

  return result;
}

std::shared_ptr<teaser::RobustRegistrationSolver>
TeaserORB::RegisterPointsWithTeaser(const PointCloud::Ptr pcd1,
                                    const PointCloud::Ptr pcd2) const {
  // Convert to Eigen
  size_t N = pcd1->size();
  Eigen::Matrix<double, 3, Eigen::Dynamic> pcd1eig(3, N), pcd2eig(3, N);
  for (size_t i = 0; i < N; ++i) {
    const PointT &pt1 = pcd1->points[i], pt2 = pcd2->points[i];
    pcd1eig.col(i) << pt1.x, pt1.y, pt1.z;
    pcd2eig.col(i) << pt2.x, pt2.y, pt2.z;
  }

  // Prepare solver based on system parameters
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = parameters_.teaser_noise_bound;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = 0.005;
  params.inlier_selection_mode =
      teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;

  std::unique_ptr<teaser::RobustRegistrationSolver> solver(
      new teaser::RobustRegistrationSolver(params));

  // Disable verbose output to stdout
  if (!parameters_.teaser_verbose)
    std::cout.setstate(std::ios_base::failbit);
  solver->solve(pcd2eig, pcd1eig);
  std::cout.clear();

  return solver;
}

HypothesisPtr TeaserORB::RunTeaserWith3DMatches(
    const std::vector<SlicePtr> &map1_features,
    const std::vector<SlicePtr> &map2_features) const {
  // Extract all matches, slice by slice, in parallel
  PointCloud::Ptr map1_points(new PointCloud()), map2_points(new PointCloud());
  std::vector<float> distances;

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (size_t j = 0; j < map2_features.size(); ++j) {
    for (size_t i = 0; i < map1_features.size(); ++i) {
      const Slice &slice_map1 = *map1_features[i],
                  slice_map2 = *map2_features[j];

      MatchingResultPtr matches = MatchKeyPoints(slice_map1, slice_map2);

      std::vector<cv::Point2f> points1img, points2img;
      cv::KeyPoint::convert(matches->map1_keypoints, points1img);
      cv::KeyPoint::convert(matches->map2_keypoints, points2img);

      // Convert to real coordinates (3D)
      PointCloud::Ptr points1 = img2real(points1img, slice_map1.slice_bounds,
                                         slice_map1.height),
                      points2 = img2real(points2img, slice_map2.slice_bounds,
                                         slice_map2.height);

#pragma omp critical
      {
        *map1_points += *points1;
        *map2_points += *points2;
        distances.insert(distances.end(),
                         std::make_move_iterator(matches->distances.begin()),
                         std::make_move_iterator(matches->distances.end()));
      }
    }
  }

  // Retain only the top N, if larger than the teaser_maximum_correspondences
  SelectTopNMatches(map1_points, map2_points, distances);
  spdlog::debug("Number of correspondences: {}", map1_points->size());
  if (map1_points->size() < 5) {
    spdlog::debug("Not enough correspondences");
    return HypothesisPtr(new Hypothesis());
  }

  // Register
  std::shared_ptr<teaser::RobustRegistrationSolver> solver =
      RegisterPointsWithTeaser(map1_points, map2_points);

  // Process result from state
  HypothesisPtr result =
      ConstructSolutionFromSolverState(solver, map1_points, map2_points);

  return result;
}

void TeaserORB::SelectTopNMatches(PointCloud::Ptr &map1_points,
                                  PointCloud::Ptr &map2_points,
                                  const std::vector<float> &distances) const {
  // Return as is if there are less matches than maximum
  if (distances.size() <= parameters_.teaser_maximum_correspondences)
    return;

  // Sort by indices
  std::vector<size_t> indices(distances.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Shortest distance (best match) at the front
  std::stable_sort(indices.begin(), indices.end(),
                   [&distances](size_t i1, size_t i2) {
                     return distances[i1] < distances[i2];
                   });

  // Collate
  PointCloud::Ptr map1_topN(new PointCloud()), map2_topN(new PointCloud());
  for (size_t i = 0; i < parameters_.teaser_maximum_correspondences; ++i) {
    map1_topN->push_back(map1_points->points[indices[i]]);
    map2_topN->push_back(map2_points->points[indices[i]]);
  }

  map1_points = map1_topN;
  map2_points = map2_topN;
}

HypothesisPtr TeaserORB::ConstructSolutionFromSolverState(
    const std::shared_ptr<teaser::RobustRegistrationSolver> &solver,
    const PointCloud::Ptr &map1_points,
    const PointCloud::Ptr &map2_points) const {
  HypothesisPtr result(new Hypothesis());
  result->inlier_points_1 = PointCloud::Ptr(new PointCloud());
  result->inlier_points_2 = PointCloud::Ptr(new PointCloud());

  std::vector<int> inlier_mask = solver->getInlierMaxClique();
  for (const auto &idx : inlier_mask) {
    const auto &pt1 = map1_points->points[idx], &pt2 = map2_points->points[idx];
    result->inlier_points_1->push_back(pt1);
    result->inlier_points_2->push_back(pt2);
  }

  result->n_inliers = result->inlier_points_1->size();

  // Decompose pose into parameters
  teaser::RegistrationSolution solution = solver->getSolution();
  result->x = solution.translation.x();
  result->y = solution.translation.y();
  result->z = solution.translation.z();

  Eigen::Matrix4d solution_mat = Eigen::Matrix4d::Identity();
  solution_mat.topLeftCorner(3, 3) = solution.rotation;
  solution_mat.topRightCorner(3, 1) = solution.translation;
  result->pose = solution_mat;

  Eigen::Vector3d eulAng = solution.rotation.eulerAngles(2, 1, 0);

  // Identify if the rotation axis is pointing downwards. In that case, the
  // rotation will be pi rad apart
  Eigen::AngleAxisd angle_axis(solution.rotation);
  if (angle_axis.axis()(2) < 0.0)
    result->theta = eulAng(0) - M_PI;
  else
    result->theta = eulAng(0);

  return result;
}

} // namespace map_matcher
