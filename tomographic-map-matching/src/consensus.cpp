#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/common/transforms.h>
#include <spdlog/spdlog.h>
#include <tomographic_map_matching/consensus.hpp>

namespace map_matcher {

Consensus::Consensus(ConsensusParameters parameters)
    : MapMatcherBase(static_cast<Parameters>(parameters)),
      parameters_(parameters) {}

void Consensus::PrintParameters() const {
  spdlog::info("[PARAMS] grid_size: {} threshold: {} orb_num_features: {} "
               "orb_scale_factor: {} orb_n_levels: {} orb_edge_threshold: {} "
               "orb_first_level: "
               "{} orb_wta_k: {} orb_patch_size: {} orb_fast_threshold: {} "
               "gms_threshold_factor: {} lsh_num_tables: {} lsh_key_size: {} "
               "lsh_multiprobe_level: {} minimum_z_overlap_percentage: {} "
               "consensus_ransac_factor: "
               "{}",
               parameters_.grid_size, parameters_.slice_z_height,
               parameters_.orb_num_features, parameters_.orb_scale_factor,
               parameters_.orb_n_levels, parameters_.orb_edge_threshold,
               parameters_.orb_first_level, parameters_.orb_wta_k,
               parameters_.orb_patch_size, parameters_.orb_fast_threshold,
               parameters_.gms_threshold_factor, parameters_.lsh_num_tables,
               parameters_.lsh_key_size, parameters_.lsh_multiprobe_level,
               parameters_.minimum_z_overlap_percentage,
               parameters_.ransac_gsize_factor);
  spdlog::info("[FLAGS] cross_match: {} median_filter: {} gms_matching: {} "
               "approximate_neighbors: {} use_rigid: {}",
               parameters_.cross_match, parameters_.median_filter,
               parameters_.gms_matching, parameters_.approximate_neighbors,
               parameters_.use_rigid);
}

HypothesisPtr
Consensus::RegisterPointCloudMaps(const PointCloud::Ptr map1_pcd,
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

  // VisualizeImageSlices(map1_image);
  // VisualizeImageSlices(map2_image);
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

  std::vector<HypothesisPtr> slice_correlations =
      CorrelateSlices(map1_slice, map2_slice);

  // Storing the same pointer to the unrefined result, if there is no refinement
  // to be performed
  HypothesisPtr result_unrefined = slice_correlations[0],
                result_refined = result_unrefined;

  spdlog::info("[TIMING] Slice correlation t_pose_estimation: {}",
               CalculateTimeSince(indiv));
  indiv = std::chrono::steady_clock::now();

  spdlog::info("Computed pose x: {} y: {} z: {} t: {}", result_unrefined->x,
               result_unrefined->y, result_unrefined->z,
               result_unrefined->theta);
  spdlog::info("[HYPOTHESES] Inlier hypothesis count num_inliers: {}",
               result_unrefined->n_inliers);
  indiv = std::chrono::steady_clock::now();

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

    // ICP refinement
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
Consensus::CorrelateSlices(const std::vector<SlicePtr> &map1_features,
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

      std::vector<SliceTransformPtr> results,
          results_interim = ComputeMapTf(map1_features, map2_features, indices);

      // Eliminate poses with zero inliers
      for (auto res : results_interim) {
        if (res->inliers.size())
          results.push_back(res);
      }

      HypothesisPtr agreed_result = VoteBetweenSlices(results);
      correlated_results.push_back(agreed_result);
    }

    // Update indices
    ++count;
    if (map2_index != 0)
      --map2_index;
    else
      ++map1_index;
  }
  spdlog::info("Num. correlations num_correlation: {}", count);

  // Providing a lambda sorting function to deal with the use of smart
  // pointers. Otherwise sorted value is not exactly accurate
  std::sort(
      correlated_results.rbegin(), correlated_results.rend(),
      [](HypothesisPtr val1, HypothesisPtr val2) { return *val1 < *val2; });
  return correlated_results;
}

std::vector<SliceTransformPtr>
Consensus::ComputeMapTf(const std::vector<SlicePtr> &map1,
                        const std::vector<SlicePtr> &map2,
                        HeightIndices indices) const {
  if (indices.m2_max - indices.m2_min != indices.m1_max - indices.m1_min) {
    spdlog::critical("Different number of slices are sent for calculation");
    throw std::runtime_error("Different number of slices");
  }

  size_t window = std::min(indices.m2_max - indices.m2_min,
                           indices.m1_max - indices.m1_min);
  std::vector<SliceTransformPtr> res(window);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < window; ++i) {
    // Initialize index
    res[i] = SliceTransformPtr(new SliceTransform());

    // Extract correct slice & assign the weak ptrs
    size_t m1_idx = indices.m1_min + i, m2_idx = indices.m2_min + i;
    const Slice &slice1 = *map1[m1_idx], slice2 = *map2[m2_idx];

    res[i]->slice1 = map1[m1_idx];
    res[i]->slice2 = map2[m2_idx];

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

    std::vector<cv::KeyPoint> kp1match = matched_keypoints->map1_keypoints,
                              kp2match = matched_keypoints->map2_keypoints;

    if (kp1match.size() <= 4) {
      spdlog::debug("Not enough matches. m1_idx: {} m2_idx: {} num_matches: {}",
                    m1_idx, m2_idx, kp1match.size());
      continue;
    }

    // Convert to image coordinates
    std::vector<cv::Point2f> points1img, points2img;
    cv::KeyPoint::convert(kp1match, points1img);
    cv::KeyPoint::convert(kp2match, points2img);

    // Convert to real coordinates
    std::vector<cv::Point2f> points1 =
        img2real(points1img, slice1.slice_bounds);
    std::vector<cv::Point2f> points2 =
        img2real(points2img, slice2.slice_bounds);

    cv::Mat inliers;

    // Coordinates used in estimation are in real coordinates. Making the RANSAC
    // threshold to be dependent on the grid size (resolution) of the maps,
    // instead of a fixed value, so that the threshold is to an extent uniform
    // across different resolution maps
    double ransacReprojThresh = parameters_.grid_size * 3.0; // Default: 3.0
    size_t maxIters = 2000;                                  // Default: 2000
    double confidence = 0.999;                               // Default: 0.99
    size_t refineIters = 10;                                 // Default: 10

    cv::Mat tf;
    if (parameters_.use_rigid)
      tf = cv::estimateRigid2D(points2, points1, inliers, cv::RANSAC,
                               ransacReprojThresh, maxIters, confidence,
                               refineIters);
    else
      tf = cv::estimateAffinePartial2D(points2, points1, inliers, cv::RANSAC,
                                       ransacReprojThresh, maxIters, confidence,
                                       refineIters);

    // Extract inlier corresponding points
    std::vector<std::pair<cv::Point2f, cv::Point2f>> inliers_vec;
    size_t idx_count = 0;
    for (cv::MatIterator_<uchar> it = inliers.begin<uchar>();
         it != inliers.end<uchar>(); ++it) {
      if (*it == 1) {
        inliers_vec.push_back(
            std::make_pair(points1[idx_count], points2[idx_count]));
      }
      ++idx_count;
    }

    if (inliers_vec.empty()) {
      spdlog::debug("Not enough inliers. m1_idx: {} m2_idx: {}", m1_idx,
                    m2_idx);
      continue;
    }

    // Extract components
    double theta = std::atan2(tf.at<double>(1, 0), tf.at<double>(0, 0));
    double x = tf.at<double>(0, 2), y = tf.at<double>(1, 2);
    double scale = tf.at<double>(0, 0) / std::cos(theta);
    spdlog::debug("estimate*2D scale: {}", scale);

    // Height difference between matching slices is sufficient and should be
    // consistent between different maps since the grid size is the same
    double z = map1[m1_idx]->height - map2[m2_idx]->height;

    Eigen::Affine3d pose = Eigen::Affine3d::Identity();
    pose.translation() << x, y, z;
    pose.rotate(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));

    res[i]->inliers = inliers_vec;
    res[i]->x = x;
    res[i]->y = y;
    res[i]->z = z;
    res[i]->theta = theta;
    res[i]->pose = pose.matrix();
  }
  return res;
}

HypothesisPtr Consensus::VoteBetweenSlices(
    const std::vector<SliceTransformPtr> &results) const {
  std::vector<HypothesisPtr> voted_results(results.size());
  double dist_thresh = parameters_.grid_size * parameters_.ransac_gsize_factor,
         tThresh = 0.015; // ~ 15 deg

  // Skip if there are no slices to vote for
  if (results.size() == 0) {
    return HypothesisPtr(new Hypothesis());
  }

  // Since the number of slices is small, check RANSAC over all slice hypotheses
  for (size_t i = 0; i < results.size(); ++i) {
    size_t n_inliers = 0;
    double xAvg = 0.0, yAvg = 0.0, tSinAvg = 0.0, tCosAvg = 0.0;
    std::vector<SliceTransformPtr> inlier_slices;

    // Collective update
    for (size_t j = 0; j < results.size(); ++j) {
      const SliceTransformPtr k = results[j];
      double dx = results[i]->x - k->x, dy = results[i]->y - k->y,
             dist = dx * dx + dy * dy;
      if (dist < dist_thresh * dist_thresh &&
          std::abs(std::cos(results[i]->theta) - std::cos(k->theta)) <
              tThresh) {
        ++n_inliers;
        xAvg += k->x;
        yAvg += k->y;
        tSinAvg += std::sin(k->theta);
        tCosAvg += std::cos(k->theta);
        inlier_slices.push_back(k);
      }
    }
    double n_inliers_d = static_cast<double>(n_inliers);

    xAvg /= n_inliers_d;
    yAvg /= n_inliers_d;
    tSinAvg /= n_inliers_d;
    tCosAvg /= n_inliers_d;
    double tAvg = std::atan2(tSinAvg, tCosAvg);

    std::vector<double> cur_hyp{xAvg, yAvg, tAvg};

    voted_results[i] = HypothesisPtr(new Hypothesis);
    voted_results[i]->n_inliers = n_inliers;
    voted_results[i]->x = xAvg;
    voted_results[i]->y = yAvg;
    voted_results[i]->z = results[i]->z;
    voted_results[i]->theta = tAvg;
    // Move vector to avoid copying
    voted_results[i]->inlier_slices = std::move(inlier_slices);
    voted_results[i]->pose =
        ConstructTransformFromParameters(xAvg, yAvg, results[i]->z, tAvg);
  }

  std::sort(
      voted_results.rbegin(), voted_results.rend(),
      [](HypothesisPtr val1, HypothesisPtr val2) { return *val1 < *val2; });
  return voted_results[0];
}

} // namespace map_matcher
