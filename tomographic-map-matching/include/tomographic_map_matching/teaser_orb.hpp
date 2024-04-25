#pragma once

#include <teaser/registration.h>
#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

struct TeaserORBParameters : public Parameters {

  TeaserORBParameters() = default;
  TeaserORBParameters(const TeaserORBParameters &) = default;
  TeaserORBParameters(const Parameters &p) : Parameters(p) {}
  double teaser_noise_bound = 0.02;
  size_t teaser_num_correspondences_max = 10000;
  bool teaser_verbose = false;
  bool teaser_3d = false;
};

void to_json(json &j, const TeaserORBParameters &p);
void from_json(const json &j, TeaserORBParameters &p);

class TeaserORB : public MapMatcherBase {
public:
  TeaserORB();
  TeaserORB(TeaserORBParameters parameters);
  json GetParameters() const override;
  void SetParameters(const json &parameters);
  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                                       const PointCloud::Ptr pcd2,
                                       json &stats) const override;

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
