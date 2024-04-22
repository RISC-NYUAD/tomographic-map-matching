#pragma once

#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

struct ConsensusParameters : public Parameters {
  double ransac_gsize_factor, minimum_z_overlap_percentage;
  bool use_rigid;
};

class Consensus : public MapMatcherBase {
public:
  Consensus(ConsensusParameters parameters);
  HypothesisPtr
  RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                         const PointCloud::Ptr pcd2) const override;
  void PrintParameters() const override;

private:
  ConsensusParameters parameters_;
  std::vector<HypothesisPtr>
  CorrelateSlices(const std::vector<SlicePtr> &map1_features,
                  const std::vector<SlicePtr> &map2_features) const;
  std::vector<SliceTransformPtr> ComputeMapTf(const std::vector<SlicePtr> &map1,
                                              const std::vector<SlicePtr> &map2,
                                              HeightIndices indices) const;
  HypothesisPtr
  VoteBetweenSlices(const std::vector<SliceTransformPtr> &results) const;
};

} // namespace map_matcher
