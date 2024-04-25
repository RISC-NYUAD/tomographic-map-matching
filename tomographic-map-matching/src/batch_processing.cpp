// #include <cstdlib>
// #include <exception>
// #include <filesystem>
#include <gflags/gflags.h>
// #include <memory>
// #include <opencv2/core/utility.hpp>
// #include <pcl/common/transforms.h>
// #include <pcl/io/pcd_io.h>
// #include <random>
// #include <spdlog/sinks/basic_file_sink.h>
// #include <spdlog/sinks/stdout_color_sinks.h>
// #include <spdlog/spdlog.h>
// #include <sstream>
// #include <stdexcept>
// #include <string>
#include <tomographic_map_matching/consensus.hpp>
#include <tomographic_map_matching/map_matcher_base.hpp>
#include <tomographic_map_matching/teaser_orb.hpp>

#include <iostream>

// Flags
DEFINE_string(config_path, "",
              "JSON file with parameters. Results are appended");
DEFINE_string(data_path, "",
              "Scenario file that delineates pairs to be tested");

// Eigen::Matrix4d ReadGTPose(std::string fname) {
//   // Load poses
//   Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
//   try {
//     std::string line;
//     std::fstream poseFile;
//     poseFile.open(fname, std::ios::in);
//     getline(poseFile, line);
//     poseFile.close();

//     double val;
//     int itemCount = 0, row = 0, col = 0;

//     while (itemCount++ < 12) { // Each row stores 3x4 tf matrix entries

//       val = std::stod(line.substr(0, line.find_first_of(" ")));
//       pose(row, col++) = val;

//       // Prepare for the next
//       line = line.substr(line.find_first_of(" ") + 1);

//       // Move to next row if all cols are filled
//       if (col == 4) {
//         ++row;
//         col = 0;
//       }
//     }
//   } catch (std::exception &ex) {
//     spdlog::warn("File {} does not exist. Assuming identity pose", fname);
//   }

//   return pose;
// }

// map_matcher::PointCloud::Ptr
// ApplyRandomRollPitch(const map_matcher::PointCloud::Ptr &pcd,
//                      Eigen::Matrix4d pose) {
//   // Apply roll-pitch random pert. on the untransformed map, if asked
//   if (FLAGS_random_roll_pitch_magnitude <= 0.0)
//     return pcd;

//   map_matcher::PointCloud::Ptr perturbed_pcd(new map_matcher::PointCloud());
//   Eigen::Matrix4d posei = pose.inverse();
//   pcl::transformPointCloud(*pcd, *perturbed_pcd, posei);

//   std::random_device rd{};
//   std::mt19937 rgen{rd()};

//   // Sample radian, even though degree is specified
//   std::normal_distribution<double> sampler{
//       0.0, FLAGS_random_roll_pitch_magnitude * M_PI / 180.0};

//   double roll = sampler(rgen), pitch = sampler(rgen);
//   spdlog::info("[RP Pert] Roll(rad): {} Pitch(rad): {}", roll, pitch);

//   Eigen::Matrix4d perturbed_pose = Eigen::Matrix4d::Identity();
//   Eigen::AngleAxisd roll_mat(roll, Eigen::Vector3d::UnitX()),
//       pitch_mat(pitch, Eigen::Vector3d::UnitY());

//   perturbed_pose.topLeftCorner(3, 3) = pitch_mat.matrix() *
//   roll_mat.matrix();

//   perturbed_pose.matrix() = pose * perturbed_pose.matrix();
//   pcl::transformPointCloud(*perturbed_pcd, *perturbed_pcd, perturbed_pose);
//   return perturbed_pcd;
// }

// double ComputeAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est) {
//   return std::abs(std::acos(
//       fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)));
// }

using json = map_matcher::json;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  // Read config and data pairs
  std::ifstream config_file(FLAGS_config_path);
  json config = json::parse(config_file);

  map_matcher::Parameters params;
  json params_json = params;

  for (auto &[key, value] : config.items()) {
    params_json[key] = value;
  }

  // map_matcher::Consensus consensus;
  // map_matcher::TeaserORB teaser;

  map_matcher::TeaserORB teaser;
  std::cout << teaser.GetParameters().dump(2) << std::endl << std::endl;

  teaser.SetParameters(params_json);
  std::cout << teaser.GetParameters().dump(2) << std::endl << std::endl;

  map_matcher::Consensus consensus;
  std::cout << consensus.GetParameters().dump(2) << std::endl << std::endl;

  consensus.SetParameters(params_json);
  std::cout << consensus.GetParameters().dump(2) << std::endl << std::endl;

  // // Base
  // std::cout << "## Base:" << std::endl;
  // {
  //   auto &params = params_base;
  //   json params_json = params;
  //   std::cout << "Before:" << std::endl;
  //   std::cout << params_json.dump(2) << std::endl;

  //   for (auto &[key, value] : config.items()) {
  //     params_json[key] = value;
  //   }

  //   std::cout << "After:" << std::endl;
  //   std::cout << params_json.dump(2) << std::endl;
  // }

  // // Consensus
  // std::cout << std::endl << "## Consensus:" << std::endl;
  // {
  //   map_matcher::Consensus consensus;
  //   auto params = consensus.GetParameters();
  //   json params_json = params;
  //   std::cout << "Before:" << std::endl;
  //   std::cout << params_json.dump(2) << std::endl;

  //   for (auto &[key, value] : config.items()) {
  //     params_json[key] = value;
  //   }

  //   std::cout << "After:" << std::endl;
  //   std::cout << params_json.dump(2) << std::endl;
  // }

  // // Teaser
  // std::cout << std::endl << "## TeaserORB:" << std::endl;
  // {
  //   map_matcher::TeaserORB teaser;
  //   auto params = teaser.GetParameters();
  //   json params_json = params;
  //   std::cout << "Before:" << std::endl;
  //   std::cout << params_json.dump(2) << std::endl;

  //   for (auto &[key, value] : config.items()) {
  //     params_json[key] = value;
  //   }

  //   std::cout << "After:" << std::endl;
  //   std::cout << params_json.dump(2) << std::endl;
  // }

  // Patch default params with config

  // if (FLAGS_algorithm != 0 and FLAGS_algorithm != 1) {
  //   spdlog::critical("Algorithm {} not implemented", FLAGS_algorithm);
  //   exit(EXIT_FAILURE);
  // }

  // // Log to file if specified
  // if (!(std::string(FLAGS_log_folder).empty())) {
  //   // Generate log name from input parameters
  //   // Dataset names: From last slash to excluding .pcd (assuming all ends
  //   with
  //   // .pcd here)
  //   // TODO: Implement with filesystem
  //   // std::filesystem::path map1_path (FLAGS_map1_path), map2_path
  //   // (FLAGS_map2_path);

  //   std::string data1_name = FLAGS_map1_path.substr(
  //                   FLAGS_map1_path.find_last_of("/") + 1,
  //                   FLAGS_map1_path.size() -
  //                       (FLAGS_map1_path.find_last_of("/") + 5)),
  //               data2_name = FLAGS_map2_path.substr(
  //                   FLAGS_map2_path.find_last_of("/") + 1,
  //                   FLAGS_map2_path.size() -
  //                       (FLAGS_map2_path.find_last_of("/") + 5));

  //   std::stringstream logfile_name;
  //   logfile_name.precision(2);
  //   logfile_name << data1_name << "_" << data2_name << "_g" <<
  //   FLAGS_grid_size
  //                << "_rp" << FLAGS_random_roll_pitch_magnitude << "_cm"
  //                << FLAGS_cross_match << "_mf" << FLAGS_median_filter <<
  //                "_icp"
  //                << FLAGS_icp_refinement << "_onf" << FLAGS_orb_num_features
  //                << "_osf" << FLAGS_orb_scale_factor << "_onl"
  //                << FLAGS_orb_n_levels << "_oet" << FLAGS_orb_edge_threshold
  //                << "_ofl" << FLAGS_orb_first_level << "_owk" <<
  //                FLAGS_orb_wta_k
  //                << "_ops" << FLAGS_orb_patch_size << "_oft"
  //                << FLAGS_orb_fast_threshold;
  //   if (FLAGS_algorithm == 0) {
  //     logfile_name << "_consensus"
  //                  << "_crf" << FLAGS_consensus_ransac_factor;
  //   } else if (FLAGS_algorithm == 1) {
  //     logfile_name << "_teaser"
  //                  << "_tnb" << FLAGS_teaser_noise_bound << "_tmc"
  //                  << FLAGS_teaser_maximum_correspondences << "_t3d"
  //                  << FLAGS_teaser_3d;
  //   }

  //   // Mark time for each log file
  //   auto now = std::time(nullptr);
  //   auto now_manip = *std::localtime(&now);
  //   logfile_name << std::put_time(&now_manip, "_%d-%m-%Y_%H-%M-%S") <<
  //   ".log";

  //   std::filesystem::path base("result_logs"), folder(FLAGS_log_folder),
  //       file(logfile_name.str()), combined = base / folder / file;

  //   // If the folders do not exist, create them
  //   std::filesystem::create_directories(base / folder);

  //   // Register combined stdout + file logger as a combined, default logger
  //   std::vector<spdlog::sink_ptr> sinks;
  //   sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  //   sinks.push_back(
  //       std::make_shared<spdlog::sinks::basic_file_sink_mt>(combined.string()));
  //   auto combined_logger = std::make_shared<spdlog::logger>(
  //       "combined_logger", begin(sinks), end(sinks));
  //   spdlog::set_default_logger(combined_logger);
  // }

  // // Set logging pattern
  // spdlog::set_pattern("%Y-%m-%d-%H:%M:%S.%e|%^%l%$| %v");

  // // Set logging level to debug if specified
  // if (FLAGS_debug) {
  //   spdlog::set_level(spdlog::level::debug);
  //   spdlog::debug("OpenCV version: {}", cv::getVersionString());
  // }

  // map_matcher::json stats;

  // // Read common parameters from flags
  // std::unique_ptr<map_matcher::MapMatcherBase> matcher_object;

  // if (FLAGS_algorithm == 0) {
  //   spdlog::info("Algorithm: Consensus");
  //   auto parameters = std::unique_ptr<map_matcher::ConsensusParameters>(
  //       new map_matcher::ConsensusParameters());
  //   ParseFlagsToCommonParameters(parameters);
  //   parameters->minimum_z_overlap_percentage =
  //       FLAGS_minimum_z_overlap_percentage;
  //   parameters->ransac_gsize_factor = FLAGS_consensus_ransac_factor;
  //   parameters->use_rigid = FLAGS_consensus_use_rigid;
  //   matcher_object = std::unique_ptr<map_matcher::Consensus>(
  //       new map_matcher::Consensus(*parameters));

  // } else {
  //   spdlog::info("Algorithm: TEASER++-ORB");
  //   auto parameters = std::unique_ptr<map_matcher::TeaserORBParameters>(
  //       new map_matcher::TeaserORBParameters());
  //   ParseFlagsToCommonParameters(parameters);
  //   parameters->teaser_maximum_correspondences =
  //       FLAGS_teaser_maximum_correspondences;
  //   parameters->teaser_noise_bound = FLAGS_teaser_noise_bound;
  //   parameters->teaser_verbose = FLAGS_teaser_verbose;
  //   parameters->teaser_3d = FLAGS_teaser_3d;
  //   matcher_object = std::unique_ptr<map_matcher::TeaserORB>(
  //       new map_matcher::TeaserORB(*parameters));
  // };

  // // Print (log) parameters
  // matcher_object->PrintParameters();
  // spdlog::info("[PARAMS] random_roll_pitch_magnitude: {}",
  //              FLAGS_random_roll_pitch_magnitude);

  // // Load clouds
  // map_matcher::PointCloud::Ptr map1_pcd(new map_matcher::PointCloud()),
  //     map2_pcd(new map_matcher::PointCloud());
  // pcl::io::loadPCDFile(FLAGS_map1_path, *map1_pcd);
  // pcl::io::loadPCDFile(FLAGS_map2_path, *map2_pcd);
  // spdlog::info("[MAPNAME] Map 1: {} Map 2: {}", FLAGS_map1_path,
  //              FLAGS_map2_path);

  // // Calculate ground truth poses
  // Eigen::Matrix4d pose1 = ReadGTPose(
  //                     FLAGS_map1_path.substr(0, FLAGS_map1_path.size() - 4) +
  //                     "-gtpose.txt"),
  //                 pose2 = ReadGTPose(
  //                     FLAGS_map2_path.substr(0, FLAGS_map2_path.size() - 4) +
  //                     "-gtpose.txt");

  // // Apply random roll-pitch if specified
  // map1_pcd = ApplyRandomRollPitch(map1_pcd, pose1);
  // map2_pcd = ApplyRandomRollPitch(map2_pcd, pose2);

  // // Determine target pose based on gt info
  // Eigen::Matrix4d target = pose1 * pose2.inverse();

  // {
  //   Eigen::Matrix3d rotm = target.block<3, 3>(0, 0);
  //   Eigen::AngleAxisd axang(rotm);
  //   double angle = axang.angle() * axang.axis()(2);
  //   spdlog::info("Target pose: x: {} y: {}: z: {} t: {}", target(0, 3),
  //                target(1, 3), target(2, 3), angle);
  // }

  // // Calculate & print errors
  // map_matcher::HypothesisPtr result =
  //     matcher_object->RegisterPointCloudMaps(map1_pcd, map2_pcd, stats);

  // spdlog::info(
  //     "[RESULT] Translation error (m) translation_error: {}",
  //     (target.topRightCorner(3, 1) - result->pose.topRightCorner(3,
  //     1)).norm());
  // spdlog::info("[RESULT] Rotation error (rad) rotation_error: {}",
  //              ComputeAngularError(target.topLeftCorner(3, 3),
  //                                  result->pose.topLeftCorner(3, 3)));
  // spdlog::info("Stats: {}", stats.dump());

  // // Visualization
  // if (FLAGS_visualize) {
  //   // matcher_object->VisualizeHypothesisSlices(result);
  //   matcher_object->VisualizeHypothesis(map1_pcd, map2_pcd, result);
  // }

  return 0;
}
