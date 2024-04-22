#include <cstdlib>
#include <exception>
#include <filesystem>
#include <gflags/gflags.h>
#include <memory>
#include <opencv2/core/utility.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <random>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tomographic_map_matching/consensus.hpp>
#include <tomographic_map_matching/map_matcher_base.hpp>
#include <tomographic_map_matching/teaser_orb.hpp>

// Flags
DEFINE_string(map1_path, "", "Path to base map");
DEFINE_string(map2_path, "", "Path to map to be matched");
DEFINE_double(grid_size, 0.1, "Grid size used in generating maps");
// DEFINE_uint64(
//     num_slices,
//     1,
//     "Number of grids to be incorporated in a single slice in vertical
//     direction");
DEFINE_double(
    random_roll_pitch_magnitude, 0.0,
    "Std. deviation of random roll-pitch applied  (if positive) to maps "
    "before registration, in degrees");
DEFINE_uint64(
    icp_refinement, 0,
    "ICP Refinement mode: 0:none, 1:point-to-point icp, 2:icp_nl, 3:gicp");
DEFINE_double(gms_threshold_factor, 0.5,
              "Threshold factor for GMS matching. Higher means less matches");
DEFINE_double(
    minimum_z_overlap_percentage, 0.0,
    "Minimum expected overlap percentage in z-height. Larger means ignoring "
    "z hypotheses with less overlap and speed up correlation step");

// Check
// https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/lsh_index.h
// for description of parameters
DEFINE_uint64(lsh_num_tables, 12, "FLANN number of hash table");
DEFINE_uint64(lsh_key_size, 20, "FLANN key size");
DEFINE_uint64(lsh_multiprobe_level, 2, "FLANN multiprobe level");

DEFINE_bool(cross_match, false,
            "Only keep features that are two-way best matches");
DEFINE_bool(median_filter, false,
            "Apply median blur to 2D image before feature computation");
DEFINE_bool(
    approximate_neighbors, false,
    "Use approximate nearest neighbor for matching instead of brute force");
DEFINE_bool(gms_matching, false, "Use GMS matching scheme");
DEFINE_bool(visualize, false, "Enable visualizations");
DEFINE_bool(debug, false, "Set logging level to debug");
DEFINE_string(log_folder, "", "Folder to generate logs");
DEFINE_uint64(algorithm, 0, "Algorithm selection. 0: Consensus, 1: TEASER++");

// Check OpenCV ORB definition for parameter descriptions
DEFINE_uint64(orb_num_features, 1000, "");
DEFINE_double(orb_scale_factor, 1.2, "");
DEFINE_uint64(orb_n_levels, 8, "");
DEFINE_uint64(orb_edge_threshold, 31, "");
DEFINE_uint64(orb_first_level, 0, "");
DEFINE_uint64(orb_wta_k, 2, "");
DEFINE_uint64(orb_patch_size, 31, "");
DEFINE_uint64(orb_fast_threshold, 20, "");

// Specific to Consensus
DEFINE_double(
    consensus_ransac_factor, 5.0,
    "RANSAC distance threshold factor, with respect to the grid size (thresh "
    "= grid_size * ransac_factor). Applies only to Consensus algorithm");
DEFINE_bool(consensus_use_rigid, false,
            "Use estimateRigid2D instead of estimateAffine2D for registration");

// Specific to Teaser++
DEFINE_double(teaser_noise_bound, 0.02,
              "Noise bound for TEASER++ robust registrator");
DEFINE_uint64(
    teaser_maximum_correspondences, 10000,
    "Maximum number of correspondences to use in TEASER++. Higher number "
    "increases computation cost significantly");
DEFINE_bool(teaser_verbose, false, "Display teaser++ stdout");
DEFINE_bool(teaser_3d, false, "Use 3D matches instead of correlation");

template <typename ParameterType>
void ParseFlagsToCommonParameters(ParameterType &parameters) {
  parameters->grid_size = FLAGS_grid_size;
  parameters->slice_z_height = parameters->grid_size;
  parameters->orb_num_features = FLAGS_orb_num_features;
  parameters->orb_scale_factor = FLAGS_orb_scale_factor;
  parameters->orb_n_levels = FLAGS_orb_n_levels;
  parameters->orb_edge_threshold = FLAGS_orb_edge_threshold;
  parameters->orb_first_level = FLAGS_orb_first_level;
  parameters->orb_wta_k = FLAGS_orb_wta_k;
  parameters->orb_patch_size = FLAGS_orb_patch_size;
  parameters->orb_fast_threshold = FLAGS_orb_fast_threshold;
  parameters->icp_refinement = FLAGS_icp_refinement;
  parameters->cross_match = FLAGS_cross_match;
  parameters->median_filter = FLAGS_median_filter;
  parameters->gms_matching = FLAGS_gms_matching;
  parameters->gms_threshold_factor = FLAGS_gms_threshold_factor;
  parameters->approximate_neighbors = FLAGS_approximate_neighbors;
  parameters->lsh_num_tables = FLAGS_lsh_num_tables;
  parameters->lsh_key_size = FLAGS_lsh_key_size;
  parameters->lsh_multiprobe_level = FLAGS_lsh_multiprobe_level;
}

Eigen::Matrix4d ReadGTPose(std::string fname) {
  // Load poses
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  try {
    std::string line;
    std::fstream poseFile;
    poseFile.open(fname, std::ios::in);
    getline(poseFile, line);
    poseFile.close();

    double val;
    int itemCount = 0, row = 0, col = 0;

    while (itemCount++ < 12) { // Each row stores 3x4 tf matrix entries

      val = std::stod(line.substr(0, line.find_first_of(" ")));
      pose(row, col++) = val;

      // Prepare for the next
      line = line.substr(line.find_first_of(" ") + 1);

      // Move to next row if all cols are filled
      if (col == 4) {
        ++row;
        col = 0;
      }
    }
  } catch (std::exception &ex) {
    spdlog::warn("File {} does not exist. Assuming identity pose", fname);
  }

  return pose;
}

map_matcher::PointCloud::Ptr
ApplyRandomRollPitch(const map_matcher::PointCloud::Ptr &pcd,
                     Eigen::Matrix4d pose) {
  // Apply roll-pitch random pert. on the untransformed map, if asked
  if (FLAGS_random_roll_pitch_magnitude <= 0.0)
    return pcd;

  map_matcher::PointCloud::Ptr perturbed_pcd(new map_matcher::PointCloud());
  Eigen::Matrix4d posei = pose.inverse();
  pcl::transformPointCloud(*pcd, *perturbed_pcd, posei);

  std::random_device rd{};
  std::mt19937 rgen{rd()};

  // Sample radian, even though degree is specified
  std::normal_distribution<double> sampler{
      0.0, FLAGS_random_roll_pitch_magnitude * M_PI / 180.0};

  double roll = sampler(rgen), pitch = sampler(rgen);
  spdlog::info("[RP Pert] Roll(rad): {} Pitch(rad): {}", roll, pitch);

  Eigen::Matrix4d perturbed_pose = Eigen::Matrix4d::Identity();
  Eigen::AngleAxisd roll_mat(roll, Eigen::Vector3d::UnitX()),
      pitch_mat(pitch, Eigen::Vector3d::UnitY());

  perturbed_pose.topLeftCorner(3, 3) = pitch_mat.matrix() * roll_mat.matrix();

  perturbed_pose.matrix() = pose * perturbed_pose.matrix();
  pcl::transformPointCloud(*perturbed_pcd, *perturbed_pcd, perturbed_pose);
  return perturbed_pcd;
}

double ComputeAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est) {
  return std::abs(std::acos(
      fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)));
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_algorithm != 0 and FLAGS_algorithm != 1) {
    spdlog::critical("Algorithm {} not implemented", FLAGS_algorithm);
    exit(EXIT_FAILURE);
  }

  // Log to file if specified
  if (!(std::string(FLAGS_log_folder).empty())) {
    // Generate log name from input parameters
    // Dataset names: From last slash to excluding .pcd (assuming all ends with
    // .pcd here)
    // TODO: Implement with filesystem
    // std::filesystem::path map1_path (FLAGS_map1_path), map2_path
    // (FLAGS_map2_path);

    std::string data1_name = FLAGS_map1_path.substr(
                    FLAGS_map1_path.find_last_of("/") + 1,
                    FLAGS_map1_path.size() -
                        (FLAGS_map1_path.find_last_of("/") + 5)),
                data2_name = FLAGS_map2_path.substr(
                    FLAGS_map2_path.find_last_of("/") + 1,
                    FLAGS_map2_path.size() -
                        (FLAGS_map2_path.find_last_of("/") + 5));

    std::stringstream logfile_name;
    logfile_name.precision(2);
    logfile_name << data1_name << "_" << data2_name << "_g" << FLAGS_grid_size
                 << "_rp" << FLAGS_random_roll_pitch_magnitude << "_cm"
                 << FLAGS_cross_match << "_mf" << FLAGS_median_filter << "_icp"
                 << FLAGS_icp_refinement << "_onf" << FLAGS_orb_num_features
                 << "_osf" << FLAGS_orb_scale_factor << "_onl"
                 << FLAGS_orb_n_levels << "_oet" << FLAGS_orb_edge_threshold
                 << "_ofl" << FLAGS_orb_first_level << "_owk" << FLAGS_orb_wta_k
                 << "_ops" << FLAGS_orb_patch_size << "_oft"
                 << FLAGS_orb_fast_threshold;
    if (FLAGS_algorithm == 0) {
      logfile_name << "_consensus"
                   << "_crf" << FLAGS_consensus_ransac_factor;
    } else if (FLAGS_algorithm == 1) {
      logfile_name << "_teaser"
                   << "_tnb" << FLAGS_teaser_noise_bound << "_tmc"
                   << FLAGS_teaser_maximum_correspondences << "_t3d"
                   << FLAGS_teaser_3d;
    }

    // Mark time for each log file
    auto now = std::time(nullptr);
    auto now_manip = *std::localtime(&now);
    logfile_name << std::put_time(&now_manip, "_%d-%m-%Y_%H-%M-%S") << ".log";

    std::filesystem::path base("result_logs"), folder(FLAGS_log_folder),
        file(logfile_name.str()), combined = base / folder / file;

    // If the folders do not exist, create them
    std::filesystem::create_directories(base / folder);

    // Register combined stdout + file logger as a combined, default logger
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    sinks.push_back(
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(combined.string()));
    auto combined_logger = std::make_shared<spdlog::logger>(
        "combined_logger", begin(sinks), end(sinks));
    spdlog::set_default_logger(combined_logger);
  }

  // Set logging pattern
  spdlog::set_pattern("%Y-%m-%d-%H:%M:%S.%e|%^%l%$| %v");

  // Set logging level to debug if specified
  if (FLAGS_debug) {
    spdlog::set_level(spdlog::level::debug);
    spdlog::debug("OpenCV version: {}", cv::getVersionString());
  }

  // Read common parameters from flags
  std::unique_ptr<map_matcher::MapMatcherBase> matcher_object;

  if (FLAGS_algorithm == 0) {
    spdlog::info("Algorithm: Consensus");
    auto parameters = std::unique_ptr<map_matcher::ConsensusParameters>(
        new map_matcher::ConsensusParameters());
    ParseFlagsToCommonParameters(parameters);
    parameters->minimum_z_overlap_percentage =
        FLAGS_minimum_z_overlap_percentage;
    parameters->ransac_gsize_factor = FLAGS_consensus_ransac_factor;
    parameters->use_rigid = FLAGS_consensus_use_rigid;
    matcher_object = std::unique_ptr<map_matcher::Consensus>(
        new map_matcher::Consensus(*parameters));

  } else {
    spdlog::info("Algorithm: TEASER++-ORB");
    auto parameters = std::unique_ptr<map_matcher::TeaserORBParameters>(
        new map_matcher::TeaserORBParameters());
    ParseFlagsToCommonParameters(parameters);
    parameters->teaser_maximum_correspondences =
        FLAGS_teaser_maximum_correspondences;
    parameters->teaser_noise_bound = FLAGS_teaser_noise_bound;
    parameters->teaser_verbose = FLAGS_teaser_verbose;
    parameters->teaser_3d = FLAGS_teaser_3d;
    matcher_object = std::unique_ptr<map_matcher::TeaserORB>(
        new map_matcher::TeaserORB(*parameters));
  };

  // Print (log) parameters
  matcher_object->PrintParameters();
  spdlog::info("[PARAMS] random_roll_pitch_magnitude: {}",
               FLAGS_random_roll_pitch_magnitude);

  // Load clouds
  map_matcher::PointCloud::Ptr map1_pcd(new map_matcher::PointCloud()),
      map2_pcd(new map_matcher::PointCloud());
  pcl::io::loadPCDFile(FLAGS_map1_path, *map1_pcd);
  pcl::io::loadPCDFile(FLAGS_map2_path, *map2_pcd);
  spdlog::info("[MAPNAME] Map 1: {} Map 2: {}", FLAGS_map1_path,
               FLAGS_map2_path);

  // Calculate ground truth poses
  Eigen::Matrix4d pose1 = ReadGTPose(
                      FLAGS_map1_path.substr(0, FLAGS_map1_path.size() - 4) +
                      "-gtpose.txt"),
                  pose2 = ReadGTPose(
                      FLAGS_map2_path.substr(0, FLAGS_map2_path.size() - 4) +
                      "-gtpose.txt");

  // Apply random roll-pitch if specified
  map1_pcd = ApplyRandomRollPitch(map1_pcd, pose1);
  map2_pcd = ApplyRandomRollPitch(map2_pcd, pose2);

  // Determine target pose based on gt info
  Eigen::Matrix4d target = pose1 * pose2.inverse();

  {
    Eigen::Matrix3d rotm = target.block<3, 3>(0, 0);
    Eigen::AngleAxisd axang(rotm);
    double angle = axang.angle() * axang.axis()(2);
    spdlog::info("Target pose: x: {} y: {}: z: {} t: {}", target(0, 3),
                 target(1, 3), target(2, 3), angle);
  }

  // Calculate & print errors
  map_matcher::HypothesisPtr result =
      matcher_object->RegisterPointCloudMaps(map1_pcd, map2_pcd);

  spdlog::info(
      "[RESULT] Translation error (m) translation_error: {}",
      (target.topRightCorner(3, 1) - result->pose.topRightCorner(3, 1)).norm());
  spdlog::info("[RESULT] Rotation error (rad) rotation_error: {}",
               ComputeAngularError(target.topLeftCorner(3, 3),
                                   result->pose.topLeftCorner(3, 3)));

  // Visualization
  if (FLAGS_visualize) {
    // matcher_object->VisualizeHypothesisSlices(result);
    matcher_object->VisualizeHypothesis(map1_pcd, map2_pcd, result);
  }

  return 0;
}
