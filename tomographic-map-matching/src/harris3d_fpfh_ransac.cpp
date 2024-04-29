#include <pcl/common/transforms.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <spdlog/spdlog.h>
#include <tomographic_map_matching/harris3d_fpfh_ransac.hpp>

namespace map_matcher {

void to_json(json &j, const Harris3DFPFHRansacParameters &p) {
  to_json(j, static_cast<Parameters>(p));
  j["normal_radius"] = p.normal_radius;
  j["fpfh_radius"] = p.fpfh_radius;
  j["harris_radius"] = p.harris_radius;
  j["harris_corner_threshold"] = p.harris_corner_threshold;
  j["ransac_inlier_threshold"] = p.ransac_inlier_threshold;
  j["ransac_refine_model"] = p.ransac_refine_model;
}

void from_json(const json &j, Harris3DFPFHRansacParameters &p) {
  Parameters p_base;
  from_json(j, p_base);
  p = Harris3DFPFHRansacParameters(p_base);
  if (j.contains("normal_radius"))
    j.at("normal_radius").get_to(p.normal_radius);

  if (j.contains("fpfh_radius"))
    j.at("fpfh_radius").get_to(p.fpfh_radius);

  if (j.contains("harris_radius"))
    j.at("harris_radius").get_to(p.harris_radius);

  if (j.contains("harris_corner_threshold"))
    j.at("harris_corner_threshold").get_to(p.harris_corner_threshold);

  if (j.contains("ransac_inlier_threshold"))
    j.at("ransac_inlier_threshold").get_to(p.ransac_inlier_threshold);

  if (j.contains("ransac_refine_model"))
    j.at("ransac_refine_model").get_to(p.ransac_refine_model);
}

Harris3DFPFHRansac::Harris3DFPFHRansac() : MapMatcherBase() {}

Harris3DFPFHRansac::Harris3DFPFHRansac(Harris3DFPFHRansacParameters parameters)
    : MapMatcherBase(static_cast<Parameters>(parameters)),
      parameters_(parameters) {}

json Harris3DFPFHRansac::GetParameters() const {
  json retval = parameters_;
  return retval;
}

void Harris3DFPFHRansac::SetParameters(const json &parameters) {
  parameters_ = parameters.template get<Harris3DFPFHRansacParameters>();
}

HypothesisPtr
Harris3DFPFHRansac::RegisterPointCloudMaps(const PointCloud::Ptr map1_pcd,
                                           const PointCloud::Ptr map2_pcd,
                                           json &stats) const {

  spdlog::debug("PCD size: {}, {}", map1_pcd->size(), map2_pcd->size());

  if (map1_pcd->size() == 0 or map2_pcd->size() == 0) {
    spdlog::critical("Pointcloud(s) are empty. Aborting");
    return HypothesisPtr(new Hypothesis());
  }

  // Timing
  std::chrono::steady_clock::time_point total, indiv;
  total = std::chrono::steady_clock::now();
  indiv = std::chrono::steady_clock::now();

  // Compute normals once
  NormalCloud::Ptr map1_normals(new NormalCloud), map2_normals(new NormalCloud);

  pcl::NormalEstimationOMP<PointT, NormalT> ne1, ne2;
  ne1.setRadiusSearch(parameters_.harris_radius);
  ne2.setRadiusSearch(parameters_.harris_radius);

  ne1.setInputCloud(map1_pcd);
  ne1.compute(*map1_normals);

  ne2.setInputCloud(map2_pcd);
  ne2.compute(*map2_normals);

  // stats["t_normal_estimation"] = CalculateTimeSince(indiv);
  spdlog::debug("Normals estimation completed");
  // indiv = std::chrono::steady_clock::now();

  // Harris3D with normals
  KeypointCloud::Ptr map1_kp(new KeypointCloud), map2_kp(new KeypointCloud);

  Harris3D harris_3d_1, harris_3d_2;
  harris_3d_1.setRadius(parameters_.harris_radius);
  harris_3d_1.setThreshold(parameters_.harris_corner_threshold);

  harris_3d_2.setRadius(parameters_.harris_radius);
  harris_3d_2.setThreshold(parameters_.harris_corner_threshold);

  harris_3d_1.setInputCloud(map1_pcd);
  harris_3d_1.setSearchMethod(ne1.getSearchMethod());
  harris_3d_1.setNormals(map1_normals);
  harris_3d_1.compute(*map1_kp);

  harris_3d_2.setInputCloud(map2_pcd);
  harris_3d_2.setSearchMethod(ne2.getSearchMethod());
  harris_3d_2.setNormals(map2_normals);
  harris_3d_2.compute(*map2_kp);

  // Extract to new pointcloud
  PointCloud::Ptr map1_kp_coords(new PointCloud),
      map2_kp_coords(new PointCloud);
  {
    pcl::ExtractIndices<KeypointT> selector;
    KeypointCloud::Ptr map1_kp_extracted(new KeypointCloud),
        map2_kp_extracted(new KeypointCloud);
    selector.setInputCloud(map1_kp);
    selector.setIndices(harris_3d_1.getKeypointsIndices());
    selector.filter(*map1_kp_extracted);
    pcl::copyPointCloud(*map1_kp_extracted, *map1_kp_coords);

    selector.setInputCloud(map2_kp);
    selector.setIndices(harris_3d_2.getKeypointsIndices());
    selector.filter(*map2_kp_extracted);
    pcl::copyPointCloud(*map2_kp_extracted, *map2_kp_coords);
  }

  spdlog::debug("Keypoint extraction completed. Took {}s. Num kpts: {}, {}",
                CalculateTimeSince(indiv), map1_kp->size(), map2_kp->size());

  // FPFH for the keypoints
  pcl::FPFHEstimationOMP<PointT, NormalT, FeatureT> fpfh;
  FeatureCloud::Ptr map1_features(new FeatureCloud),
      map2_features(new FeatureCloud);
  fpfh.setRadiusSearch(parameters_.fpfh_radius);

  fpfh.setInputCloud(map1_kp_coords);
  fpfh.setInputNormals(map1_normals);
  fpfh.setSearchSurface(map1_pcd);
  fpfh.setSearchMethod(ne1.getSearchMethod());
  fpfh.compute(*map1_features);

  fpfh.setInputCloud(map2_kp_coords);
  fpfh.setInputNormals(map2_normals);
  fpfh.setSearchSurface(map2_pcd);
  fpfh.setSearchMethod(ne2.getSearchMethod());
  fpfh.compute(*map2_features);

  stats["t_feature_extraction"] = CalculateTimeSince(indiv);
  spdlog::debug("Feature extraction time: {}",
                stats["t_feature_extraction"].template get<double>());

  stats["map1_num_features"] = map1_features->size();
  stats["map2_num_features"] = map2_features->size();

  indiv = std::chrono::steady_clock::now();

  // Matching
  pcl::registration::CorrespondenceEstimation<FeatureT, FeatureT>
      correspondence_estimator;
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
  correspondence_estimator.setInputSource(map2_features);
  correspondence_estimator.setInputTarget(map1_features);
  correspondence_estimator.determineCorrespondences(*correspondences);
  spdlog::debug("Matching completed");

  // Limit to one-to-one matches
  pcl::CorrespondencesPtr correspondences_one_to_one(new pcl::Correspondences);
  pcl::registration::CorrespondenceRejectorOneToOne rejector_one_to_one;
  rejector_one_to_one.setInputCorrespondences(correspondences);
  rejector_one_to_one.getCorrespondences(*correspondences_one_to_one);
  spdlog::debug("One-to-one rejection completed");

  // Correspondance rejection with RANSAC
  pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>
      rejector_ransac;
  pcl::CorrespondencesPtr correspondences_inlier(new pcl::Correspondences);
  rejector_ransac.setInputSource(map2_kp_coords);
  rejector_ransac.setInputTarget(map1_kp_coords);

  rejector_ransac.setInlierThreshold(parameters_.ransac_inlier_threshold);
  rejector_ransac.setRefineModel(parameters_.ransac_refine_model);

  rejector_ransac.setInputCorrespondences(correspondences_one_to_one);
  rejector_ransac.getCorrespondences(*correspondences_inlier);
  Eigen::Matrix4f transform = rejector_ransac.getBestTransformation();
  stats["t_pose_estimation"] = CalculateTimeSince(indiv);

  spdlog::debug("Pose estimation completed in {}. Num. inliers: {}",
                stats["t_pose_estimation"].template get<double>(),
                correspondences_inlier->size());

  HypothesisPtr result(new Hypothesis);
  result->n_inliers = correspondences_inlier->size();

  result->x = transform(0, 3);
  result->y = transform(1, 3);
  result->z = transform(2, 3);

  Eigen::Matrix3f rotm = transform.block<3, 3>(0, 0);
  Eigen::AngleAxisf axang(rotm);
  float angle = axang.angle() * axang.axis()(2);
  result->theta = angle;

  result->pose = ConstructTransformFromParameters(result->x, result->y,
                                                  result->z, result->theta);

  // TODO: Extract inliers to be added to result
  result->inlier_points_1 = map1_kp_coords;
  result->inlier_points_2 = map2_kp_coords;

  stats["t_total"] = CalculateTimeSince(total);

  // Measure memory use
  stats["mem_cpu"] = GetPeakRSS();

  // // Visualize features
  // VisualizeKeypoints(map1_pcd, map1_kp_coords);
  // VisualizeKeypoints(map2_pcd, map2_kp_coords);

  return result;
}

void Harris3DFPFHRansac::VisualizeKeypoints(
    const PointCloud::Ptr points, const PointCloud::Ptr keypoints) const {

  pcl::visualization::PCLVisualizer viewer("Keypoints");
  PointCloudColor points_color(points, 155, 0, 0),
      keypoints_color(keypoints, 0, 155, 0);

  viewer.addPointCloud(points, points_color, "points");
  viewer.addPointCloud(keypoints, keypoints_color, "keypoints");

  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");

  viewer.spin();
}

} // namespace map_matcher
