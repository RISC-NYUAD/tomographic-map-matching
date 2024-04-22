#pragma once

#include <teaser/registration.h>
#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

struct TeaserORBParameters : public Parameters {
  // minimum_z_overlap_percentage may be obsolete if feature matching is good
  // across different slice heights
  double minimum_z_overlap_percentage, teaser_noise_bound;
  size_t teaser_maximum_correspondences;
  bool teaser_verbose, teaser_3d;
};

class TeaserORB : public MapMatcherBase {
public:
  TeaserORB(TeaserORBParameters parameters);
  HypothesisPtr
  RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                         const PointCloud::Ptr pcd2) const override;
  void PrintParameters() const override;

private:
  TeaserORBParameters parameters_;
  std::vector<HypothesisPtr>
  CorrelateSlices(const std::vector<SlicePtr> &map1_features,
                  const std::vector<SlicePtr> &map2_features) const;
  HypothesisPtr
  RunTeaserWith3DMatches(const std::vector<SlicePtr> &map1_features,
                         const std::vector<SlicePtr> &map2_features) const;
  HypothesisPtr RegisterForGivenInterval(const std::vector<SlicePtr> &map1,
                                         const std::vector<SlicePtr> &map2,
                                         HeightIndices indices) const;
  std::shared_ptr<teaser::RobustRegistrationSolver>
  RegisterPointsWithTeaser(const PointCloud::Ptr pcd1,
                           const PointCloud::Ptr pcd2) const;
  void SelectTopNMatches(PointCloud::Ptr &map1_points,
                         PointCloud::Ptr &map2_points,
                         const std::vector<float> &distances) const;
  HypothesisPtr ConstructSolutionFromSolverState(
      const std::shared_ptr<teaser::RobustRegistrationSolver> &solver,
      const PointCloud::Ptr &map1_points,
      const PointCloud::Ptr &map2_points) const;
};
} // namespace map_matcher
